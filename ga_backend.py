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
        self._debug_log("Starting fit().")
        df = self._to_polars(X)
        self._debug_df("Input dataframe after conversion", df)
        self.numeric_cols = self._infer_numeric_cols(df)
        self._debug_log(f"Inferred numeric columns: {self.numeric_cols}")
        self._debug_log(f"Configured categorical columns: {self.categorical_cols}")

        y_np = np.asarray(y).reshape(-1).astype(float)
        if y_np.shape[0] != df.height:
            raise ValueError("y length != number of rows in X")
        self._debug_log(
            f"Target vector prepared: len={len(y_np)}, finite={int(np.isfinite(y_np).sum())}"
        )

        if target_col is None:
            target_col = "__target__"
        self._te_target_col = target_col

        if self.categorical_cols:
            df_te = df.with_columns(pl.Series(target_col, y_np))
            self._fit_target_encoders(df_te, target_col=target_col)
            df = self._apply_target_encoders(df)
            self._debug_df("Dataframe after target encoding", df)

        self._pset = None
        self._toolbox = None
        self._ensure_deap()
        self._debug_log("DEAP structures initialized.")

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

        def _new_lineage_id() -> int:
            self._lineage_counter += 1
            return self._lineage_counter

        def _assign_lineage(individual, parents: Optional[list[tuple[int, ...]]] = None) -> None:
            parent_lineages = [tuple(p) for p in (parents or [])]
            flat = []
            for ln in parent_lineages:
                flat.extend(ln)
            flat.append(_new_lineage_id())
            individual.lineage = tuple(flat[-self.lineage_depth :])

            if len(parent_lineages) >= 2:
                overlap = self._lineage_overlap_ratio(parent_lineages[0], parent_lineages[1])
                individual.inbreeding_coeff = overlap
            else:
                individual.inbreeding_coeff = 0.0

        def _first_objective(ind) -> float:
            vals = ind.fitness.values
            return float(vals[0] if len(vals) else np.inf)

        def _impostor_ratio(population) -> float:
            if not population:
                return 0.0
            hits = sum(1 for ind in population if "impostor_noise" in str(ind))
            return float(hits / len(population))

        def _eval_ga(individual):
            if not hasattr(individual, "birth_generation"):
                individual.birth_generation = generation_state["generation"]
            if not hasattr(individual, "lineage"):
                _assign_lineage(individual)

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
            age_pen = self._age_penalty(
                birth_generation=getattr(individual, "birth_generation", generation_state["generation"]),
                current_generation=generation_state["generation"],
            )
            novelty_pen = self._novelty_penalty(individual, elite_program_memory)
            incest_pen = self._incest_penalty(individual)
            adjusted = fitness_val + self.parsimony_coefficient * len(individual) + age_pen + novelty_pen + incest_pen
            if self.enable_multi_objective:
                return (adjusted, float(len(individual)))
            return (adjusted,)

        self._toolbox.register("evaluate", _eval_ga)

        pop = self._toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(maxsize=self.hall_of_fame)
        generation_state = {"generation": 0}
        elite_program_memory: list[str] = []
        impostor_disabled = False

        for ind in pop:
            ind.birth_generation = 0
            _assign_lineage(ind)

        for ind, fit in zip(pop, map(self._toolbox.evaluate, pop)):
            ind.fitness.values = fit
        hof.update(pop)
        self._debug_log(
            f"Initial GA population evaluated: population={len(pop)}, hof_size={len(hof)}"
        )
        elite_program_memory.extend(str(ind) for ind in hof)
        elite_program_memory = elite_program_memory[-self.fitness_sharing_memory_size :]

        best_hist = [_first_objective(hof[0])]
        no_improve = 0

        for _gen in range(1, self.generations + 1):
            generation_state["generation"] = _gen
            mutpb_cur, tourn_cur = self._adaptive_rates(pop)
            self._debug_log(
                f"Generation {_gen}: mutpb={mutpb_cur:.4f}, tournament_size={tourn_cur}, best={_first_objective(hof[0]):.6f}"
            )

            if self.enable_multi_objective:
                offspring = list(map(self._toolbox.clone, tools.selNSGA2(pop, len(pop))))
            else:
                offspring = list(map(self._toolbox.clone, tools.selTournament(pop, len(pop), tournsize=tourn_cur)))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if self._rng.random() < self.cxpb:
                    p1_lineage = tuple(getattr(c1, "lineage", ()))
                    p2_lineage = tuple(getattr(c2, "lineage", ()))
                    self._toolbox.mate(c1, c2)
                    c1.birth_generation = _gen
                    c2.birth_generation = _gen
                    _assign_lineage(c1, parents=[p1_lineage, p2_lineage])
                    _assign_lineage(c2, parents=[p2_lineage, p1_lineage])
                    del c1.fitness.values, c2.fitness.values

            for mut in offspring:
                if self._rng.random() < mutpb_cur:
                    prev_lineage = tuple(getattr(mut, "lineage", ()))
                    self._toolbox.mutate(mut)
                    mut.birth_generation = _gen
                    _assign_lineage(mut, parents=[prev_lineage])
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(self._toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            elite_program_memory.extend(str(ind) for ind in hof)
            elite_program_memory = elite_program_memory[-self.fitness_sharing_memory_size :]

            if (
                self.enable_impostor_op
                and self.disable_impostor_if_dominant
                and not impostor_disabled
                and _impostor_ratio(pop) >= self.impostor_dominance_threshold
            ):
                impostor_disabled = True
                pop = [ind for ind in pop if "impostor_noise" not in str(ind)]
                while len(pop) < self.population_size:
                    fresh = self._toolbox.individual()
                    fresh.birth_generation = _gen
                    _assign_lineage(fresh)
                    fresh.fitness.values = self._toolbox.evaluate(fresh)
                    pop.append(fresh)
                hof.update(pop)

            cur_best = _first_objective(hof[0])
            best_hist.append(cur_best)

            if self.early_stop_rounds is not None:
                if len(best_hist) >= 2 and (best_hist[-2] - best_hist[-1]) <= self.early_stop_tol:
                    no_improve += 1
                else:
                    no_improve = 0
                if self.enable_extinction and no_improve >= self.extinction_stagnation_rounds:
                    n_reseed = max(1, int(self.population_size * self.extinction_reseed_fraction))
                    pop_sorted = sorted(pop, key=_first_objective, reverse=True)
                    survivors = pop_sorted[:-n_reseed] if n_reseed < len(pop_sorted) else []
                    reseeded = []
                    for _ in range(n_reseed):
                        ind = self._toolbox.individual()
                        ind.birth_generation = _gen
                        _assign_lineage(ind)
                        ind.fitness.values = self._toolbox.evaluate(ind)
                        reseeded.append(ind)
                    pop = survivors + reseeded
                    if self.enable_multi_objective:
                        pop = tools.selNSGA2(pop, k=min(len(pop), self.population_size))
                    hof.update(pop)
                    no_improve = 0
                elif no_improve >= self.early_stop_rounds:
                    self._debug_log(f"Early stopping at generation {_gen}.")
                    break

        if self.enable_program_pruning and len(hof) > 0:
            hof_list = [self._toolbox.clone(ind) for ind in hof]
            top_k = min(self.prune_top_k, len(hof_list))
            for i in range(top_k):
                base_ind = self._toolbox.clone(hof_list[i])
                improved = True
                steps = 0
                while improved and len(base_ind) > 1 and steps < self.prune_max_steps:
                    improved = False
                    base_score = _first_objective(base_ind)
                    for j in range(len(base_ind)):
                        candidate = self._toolbox.clone(base_ind)
                        del candidate[j]
                        if len(candidate) == 0:
                            continue
                        try:
                            fit = self._toolbox.evaluate(candidate)
                        except Exception:
                            continue
                        candidate.fitness.values = fit
                        cand_score = _first_objective(candidate)
                        if cand_score <= base_score + self.hc_tol:
                            base_ind = candidate
                            improved = True
                            steps += 1
                            break
                hof_list[i] = base_ind
            new_hof = tools.HallOfFame(maxsize=self.hall_of_fame)
            new_hof.update(hof_list)
            hof = new_hof

        self.hof_ = hof
        self.best_programs_ = [str(ind) for ind in hof]
        self.best_feature_names_ = self._dedupe_names([
            self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in hof
        ])
        self.best_fitness_ = [float(ind.fitness.values[0]) for ind in hof]
        self._debug_log(
            f"Fit complete. Best programs={len(self.best_programs_)}, best_fitness={self.best_fitness_[0] if self.best_fitness_ else None}"
        )

        self._fitted = True
        return self

    def transform(self, X: Any) -> pl.DataFrame:
        self._debug_log("Starting transform().")
        df = self._to_polars(X)
        self._debug_df("Transform input dataframe after conversion", df)
        if self._te_maps and self._te_target_col is not None:
            df = self._apply_target_encoders(df)
            self._debug_df("Transform dataframe after target encoding", df)

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
            self._debug_df("Transform output dataframe (hill_climb)", out_df)
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
        self._debug_df("Transform output dataframe (ga)", out_df)
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
        self._debug_log(f"Target encoding global mean: {self._te_global_mean:.6f}")
        smooth = float(self.target_encoding_smoothing)
        for cat in self.categorical_cols:
            if cat not in df.columns:
                self._debug_log(f"Skipping target encoding for missing categorical column '{cat}'.")
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
            self._debug_log(
                f"Built target encoder map for '{cat}' with {agg.height} groups."
            )

    def _apply_target_encoders(self, df: pl.DataFrame) -> pl.DataFrame:
        assert self._te_target_col is not None
        out = df
        for cat, mdf in self._te_maps.items():
            tec = self._te_colname(cat, self._te_target_col)
            out = out.join(mdf, on=cat, how="left")
        return out
