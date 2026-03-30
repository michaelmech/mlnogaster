from typing import Any, Optional

import numpy as np
import polars as pl
from deap import gp, tools

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class BackendMixin:
    def fit(self, X: Any, y: Any, target_col: Optional[str] = None):
        df = self._to_polars(X)
        self.numeric_cols = self._infer_numeric_cols(df)

        y_np = np.asarray(y).reshape(-1).astype(float)
        if y_np.shape[0] != df.height:
            raise ValueError("y length != number of rows in X")

        if target_col is None:
            target_col = "__target__"
        self._te_target_col = target_col

        if self.categorical_cols:
            df_te = df.with_columns(pl.Series(target_col, y_np))
            self._fit_target_encoders(df_te, target_col=target_col)
            df = self._apply_target_encoders(df)

        self._pset = None
        self._toolbox = None
        self._ensure_deap()

        allowed_modes = {"ga", "hill_climb"}
        if self.search_mode not in allowed_modes:
            raise ValueError(f"search_mode must be one of {sorted(allowed_modes)}")

        if self.search_mode == "hill_climb":
            if self.hc_metric is None:
                raise ValueError("hc_metric must be provided when search_mode is hill_climb.")
            try:
                _ = float(self.hc_metric(np.zeros((len(y_np), 1)), y_np))
            except TypeError as e:
                raise TypeError("In hill_climb mode, hc_metric must have signature hc_metric(X, y).") from e

        if self.search_mode == "hill_climb":
            return self._fit_hill_climb(df, y_np)

        if self.metric is not None:
            try:
                _ = float(self.metric(np.zeros(3), np.zeros(3)))
            except TypeError as e:
                raise TypeError("In ga mode, metric must have signature metric(y_true, y_pred).") from e

        def _eval_ga(individual):
            compiled = gp.compile(expr=individual, pset=self._pset)
            out = compiled() if callable(compiled) else compiled

            lf = df
            if out.pre_cols:
                for name, e in out.pre_cols:
                    lf = lf.with_columns(e.alias(name))

            pred = lf.select(out.expr.alias("__pred__")).to_series().to_numpy().astype(float, copy=False)

            mask = np.isfinite(pred) & np.isfinite(y_np)
            if mask.sum() < max(5, int(0.2 * len(y_np))):
                return (1e18,)

            metric_val = self._compute_single_metric(y_np[mask], pred[mask])
            fitness_val = -metric_val if self.maximize_metric else metric_val
            return (fitness_val + self.parsimony_coefficient * len(individual),)

        self._toolbox.register("evaluate", _eval_ga)

        pop = self._toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(maxsize=self.hall_of_fame)

        for ind, fit in zip(pop, map(self._toolbox.evaluate, pop)):
            ind.fitness.values = fit
        hof.update(pop)

        best_hist = [hof[0].fitness.values[0]]
        no_improve = 0

        for _gen in range(1, self.generations + 1):
            offspring = list(map(self._toolbox.clone, self._toolbox.select(pop, len(pop))))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if self._rng.random() < self.cxpb:
                    self._toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mut in offspring:
                if self._rng.random() < self.mutpb:
                    self._toolbox.mutate(mut)
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(self._toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            cur_best = hof[0].fitness.values[0]
            best_hist.append(cur_best)

            if self.early_stop_rounds is not None:
                if len(best_hist) >= 2 and (best_hist[-2] - best_hist[-1]) <= self.early_stop_tol:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self.early_stop_rounds:
                    break

        self.hof_ = hof
        self.best_programs_ = [str(ind) for ind in hof]
        self.best_feature_names_ = self._dedupe_names([
            self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in hof
        ])
        self.best_fitness_ = [float(ind.fitness.values[0]) for ind in hof]

        self._fitted = True
        return self

    def transform(self, X: Any) -> pl.DataFrame:
        df = self._to_polars(X)
        if self._te_maps and self._te_target_col is not None:
            df = self._apply_target_encoders(df)

        self._ensure_deap()

        if self.search_mode == "hill_climb":
            if not getattr(self, "_hc_selected_inds_", None):
                raise RuntimeError("Call fit(X, y) before transform().")

            names = getattr(self, "selected_feature_names_", None)
            if not names:
                raw = [self._individual_to_symbolic(ind) for ind in self._hc_selected_inds_]
                names = self._dedupe_names([self._sanitize_feature_name(s) for s in raw])

            lf = df.lazy()
            for ind, nm in zip(self._hc_selected_inds_, names):
                out = self._compile_to_numexpr(ind)
                if out.pre_cols:
                    for name, e in out.pre_cols:
                        lf = lf.with_columns(e.alias(name))
                lf = lf.with_columns(out.expr.alias(nm))

            out_df = lf.collect()
            base_cols = list(self.index_cols) + [c for c in self.numeric_cols if c in out_df.columns]
            return out_df.select(base_cols + list(names))

        if self.hof_ is None:
            raise RuntimeError("Call fit(X, y) before transform().")

        names = getattr(self, "best_feature_names_", None)
        if not names:
            raw = [self._individual_to_symbolic(ind) for ind in self.hof_]
            names = self._dedupe_names([self._sanitize_feature_name(s) for s in raw])

        lf = df.lazy()
        for ind, nm in zip(self.hof_, names):
            compiled = gp.compile(expr=ind, pset=self._pset)
            out = compiled() if callable(compiled) else compiled
            if getattr(out, "pre_cols", None):
                for name, e in out.pre_cols:
                    lf = lf.with_columns(e.alias(name))
            lf = lf.with_columns(out.expr.alias(nm))

        out_df = lf.collect()
        base_cols = list(self.index_cols) + [c for c in self.numeric_cols if c in out_df.columns]
        return out_df.select(base_cols + list(names))

    def _to_polars(self, X: Any) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            df = X
        elif pd is not None and isinstance(X, pd.DataFrame):
            df_pd = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
            df = pl.from_pandas(df_pd)
        elif isinstance(X, np.ndarray):
            if not self.numeric_cols:
                self.numeric_cols = [f"num_{i}" for i in range(X.shape[1])]
            if X.shape[1] != len(self.numeric_cols):
                raise ValueError("numpy array column count != inferred numeric column count.")
            df = pl.DataFrame({c: X[:, i] for i, c in enumerate(self.numeric_cols)})
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        for c in self.index_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required index column '{c}'.")
        return df

    def _infer_numeric_cols(self, df: pl.DataFrame) -> list[str]:
        if self.numeric_cols:
            return list(self.numeric_cols)
        forbidden = set(self.index_cols) | set(self.categorical_cols)
        return [c for c in df.columns if c not in forbidden]

    def _fit_target_encoders(self, df: pl.DataFrame, target_col: str) -> None:
        self._te_global_mean = float(df.select(pl.col(target_col).mean()).item())
        smooth = float(self.target_encoding_smoothing)
        for cat in self.categorical_cols:
            if cat not in df.columns:
                continue
            tec = self._te_colname(cat, target_col)
            agg = (
                df.group_by(cat, maintain_order=True)
                .agg(pl.col(target_col).mean().alias("__m"), pl.len().alias("__n"))
                .with_columns(
                    ((pl.col("__m") * pl.col("__n") + smooth * self._te_global_mean) / (pl.col("__n") + smooth)).alias(tec)
                )
                .select([cat, tec])
            )
            self._te_maps[cat] = agg

    def _apply_target_encoders(self, df: pl.DataFrame) -> pl.DataFrame:
        assert self._te_target_col is not None
        out = df
        for cat, mdf in self._te_maps.items():
            tec = self._te_colname(cat, self._te_target_col)
            out = out.join(mdf, on=cat, how="left")
        return out
