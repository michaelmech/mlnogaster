import json
import re
import operator
import random
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

import polars as pl

try:
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
except Exception:  # pragma: no cover
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass
    class RegressorMixin:
        pass

from deap import base, creator, tools, gp

Arg = Union[str, "FeatureNode"]
OpType = Literal["element-wise", "groupby", "time-series", "target_encoding"]
SearchMode = Literal["ga", "hill_climb", "ga_hill_climb"]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    operator_type: OpType
    fn: Callable
    cat_pos: tuple[int, ...]
    num_pos: tuple[int, ...]
    target_pos: Optional[int] = None


@dataclass(frozen=True)
class OperationSpec:
    name: str
    arity: int
    kind: str
    builder: Callable[["GAFeatureEngineerDEAP", List[pl.Expr], Dict[str, Any]], pl.Expr]
    formatter: Callable[[List[str], Dict[str, Any]], str]
    requires_group_col: bool = False
    restrict_group_cols_to_categoricals: bool = False


@dataclass
class FeatureNode:
    op_token: str
    op_name: str
    args: List[Arg]
    params: Dict[str, Any]
    kind: str


@dataclass(frozen=True)
class NumE:
    expr: pl.Expr
    pre_cols: Tuple[Tuple[str, pl.Expr], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CatE:
    expr: pl.Expr


class GAFeatureEngineerDEAP(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(
        self,
        index_cols: Tuple[str, str] = ("ticker", "date"),
        categorical_cols: Optional[Sequence[str]] = None,
        numeric_cols: Optional[Sequence[str]] = None,
        max_colname_len: int = 160,
        random_state: int = 0,
        target_encoding_smoothing: float = 10.0,
        enable_impostor_op: bool = False,

        # GA metric: used ONLY in search_mode="ga"
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        maximize_metric: bool = False,

        # Hill-climb metric: used in search_mode="hill_climb" or "ga_hill_climb"
        hc_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        maximize_hc_metric: Optional[bool] = None,

        # GA knobs
        population_size: int = 300,
        generations: int = 25,
        tournament_size: int = 7,
        cxpb: float = 0.5,
        mutpb: float = 0.3,
        init_min_depth: int = 1,
        init_max_depth: int = 4,
        max_tree_height: int = 10,
        hall_of_fame: int = 5,
        parsimony_coefficient: float = 1e-4,
        early_stop_rounds: Optional[int] = 7,
        early_stop_tol: float = 1e-6,

        # hill-climb knobs (still apply in ga_hill_climb)
        hc_max_steps: int = 25,
        hc_num_gp_candidates: int = 50,
        hc_include_identity_candidates: bool = True,
        hc_start_with_one_identity: bool = True,
        hc_prune: bool = True,
        hc_patience: int = 5,
        hc_tol: float = 1e-12,

        checkpoint_path=None,
        checkpoint_every_accepts =np.inf,

        # new
        search_mode: SearchMode = "ga",
        verbose: bool = False,
    ):


        self.checkpoint_path=checkpoint_path
        self.checkpoint_every_accepts=checkpoint_every_accepts
        self.index_cols = index_cols
        self.categorical_cols = list(categorical_cols) if categorical_cols else []
        self.numeric_cols = list(numeric_cols) if numeric_cols else []
        self.max_colname_len = int(max_colname_len)
        self.random_state = int(random_state)
        self.target_encoding_smoothing = float(target_encoding_smoothing)
        self.enable_impostor_op = bool(enable_impostor_op)
        self._tmp_col_counter = 0
        self.verbose = bool(verbose)

        # GA metric
        self.metric = metric
        self.maximize_metric = bool(maximize_metric)
        if isinstance(self.metric, (list, tuple)):
            raise TypeError("Pass exactly one GA metric callable (not a list/tuple).")
        if self.metric is not None and not callable(self.metric):
            raise TypeError("metric must be a callable or None.")

        # HC metric
        self.hc_metric = hc_metric
        if self.hc_metric is not None and not callable(self.hc_metric):
            raise TypeError("hc_metric must be a callable or None.")
        # If not specified, default to GA maximize flag for HC as well (keeps old behavior workable)
        self.maximize_hc_metric = bool(self.maximize_metric if maximize_hc_metric is None else maximize_hc_metric)

        self.search_mode: SearchMode = search_mode

        # DEAP knobs
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)
        self.init_min_depth = int(init_min_depth)
        self.init_max_depth = int(init_max_depth)
        self.max_tree_height = int(max_tree_height)
        self.hall_of_fame = int(hall_of_fame)
        self.parsimony_coefficient = float(parsimony_coefficient)
        self.early_stop_rounds = None if early_stop_rounds is None else int(early_stop_rounds)
        self.early_stop_tol = float(early_stop_tol)

        self._rng: random.Random = random.Random(self.random_state)

        # registries (keep your existing ones)
        self._custom_ops: Dict[str, OperatorSpec] = {}
        self._ops_elementary: Dict[str, OperationSpec] = {}
        self._ops_groupby: Dict[str, OperationSpec] = {}
        self._ops_ts: Dict[str, OperationSpec] = {}
        self._ops_target: Dict[str, OperationSpec] = {}

        # target encoding
        self._te_maps: Dict[str, pl.DataFrame] = {}
        self._te_global_mean: Optional[float] = None
        self._te_target_col: Optional[str] = None

        # hill climb knobs
        self.hc_max_steps = int(hc_max_steps)
        self.hc_num_gp_candidates = int(hc_num_gp_candidates)
        self.hc_include_identity_candidates = bool(hc_include_identity_candidates)
        self.hc_start_with_one_identity = bool(hc_start_with_one_identity)
        self.hc_prune = bool(hc_prune)
        self.hc_patience = int(hc_patience)
        self.hc_tol = float(hc_tol)

        # outputs
        self.selected_programs_: Optional[List[str]] = None
        self.selected_fitness_: Optional[float] = None
        self.selected_feature_names_: Optional[List[str]] = None

        self.hof_: Optional[tools.HallOfFame] = None
        self.best_programs_: Optional[List[str]] = None
        self.best_fitness_: Optional[List[float]] = None
        self.best_feature_names_: Optional[List[str]] = None

        # internal
        self._pset: Optional[gp.PrimitiveSetTyped] = None
        self._toolbox: Optional[base.Toolbox] = None
        self._fitted: bool = False

        # keep your existing operator registration
        self._register_default_ops()

    # -------------------------
    # You should KEEP all your existing helpers:
    #   _register_default_ops, _build_pset, _ensure_deap, _to_polars, _infer_numeric_cols,
    #   _fit_target_encoders, _apply_target_encoders, _program_to_numpy,
    #   _compile_to_numexpr, _individual_to_symbolic, _sanitize_feature_name, _dedupe_names, etc.
    # -------------------------

    # ---- existing default metric for GA mode ----
    def _default_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def _compute_single_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        fn = self.metric if self.metric is not None else self._default_metric
        val = fn(y_true, y_pred)
        if not isinstance(val, (int, float, np.number)):
            raise TypeError(f"Metric must return a single number, got {type(val)}")
        val = float(val)
        if not np.isfinite(val):
            raise ValueError("Metric must return a finite number.")
        return val

    # ---- HC scoring ----
    def _hc_score(self, Xmat: np.ndarray, y: np.ndarray) -> float:
        if self.hc_metric is None:
            raise ValueError("hc_metric must be provided for hill climbing modes.")
        val = float(self.hc_metric(Xmat, y))
        if not np.isfinite(val):
            raise ValueError("hc_metric must return a finite number.")
        return val

    def _hc_improved(self, new_score: float, old_score: float) -> bool:
        if self.maximize_hc_metric:
            return new_score > (old_score + self.hc_tol)
        return new_score < (old_score - self.hc_tol)

    # -------------------------
    # GA-backed hill climbing
    # -------------------------
    @dataclass
    class _HCState:
        selected_inds: List[Any]
        selected_names: List[str]
        best_X: np.ndarray
        best_score: float
        no_improve: int = 0

    def _make_identity_individual(self, col: str):
        self._ensure_deap()
        expr = f"el__identity(num__{col})"
        return creator.Individual(gp.PrimitiveTree.from_string(expr, self._pset))

    def _fit_ga_hill_climb(
          self,
          df: pl.DataFrame,
          y_np: np.ndarray,

      ) -> "GAFeatureEngineerDEAP":

        """Hybrid mode:
        - GA generates candidates via evolution.
        - Evaluation uses HC metric incrementally (add candidate feature to current selected set).
        - Acceptance updates the HC state (selected feature set).
        - GA's original y/y_pred evaluation is not performed.
        """
        self._ensure_deap()

        def _patch_legacy_constants(expr: str) -> str:
          # Replace bare legacy token "_create_lit" with a callable literal
          return re.sub(
              r'(?<![\w])_create_lit(?!\s*\()',
              "_create_lit(0.0)",
              expr
          )

        # ---- init HC state (supports resume) ----
        def build_X(ind_list: List[Any]) -> np.ndarray:
            cols = [self._program_to_numpy(df, ind) for ind in ind_list]
            return np.column_stack(cols) if cols else np.empty((df.height, 0), dtype=float)


        if self.checkpoint_path is not None:
            # Rebuild accepted features from their string programs
            with open(self.checkpoint_path, "r") as f:
              resume_checkpoint = json.load(f)


            selected_inds = [
                  creator.Individual(
                      gp.PrimitiveTree.from_string(
                          _patch_legacy_constants(expr),
                          self._pset
                      )
                  )
                  for expr in resume_checkpoint["selected_programs"]
              ]



            selected_names = list(resume_checkpoint["selected_programs"])


            best_X = build_X(selected_inds)
            best_score = float(resume_checkpoint["best_score"])

            mask = np.isfinite(y_np) & np.all(np.isfinite(best_X), axis=1)
            best_score = self._hc_score(best_X[mask], y_np[mask])
            print(best_score)


            no_improve0 = int(resume_checkpoint.get("no_improve", 0))
        else:
            remaining_identity = set(self.numeric_cols)
            selected_inds: List[Any] = []
            selected_names: List[str] = []

            if self.hc_start_with_one_identity and remaining_identity:
                c0 = self._rng.choice(list(remaining_identity))
                ind0 = self._make_identity_individual(c0)
                selected_inds.append(ind0)
                selected_names.append(str(ind0))
                remaining_identity.remove(c0)

            if not selected_inds:
                if not remaining_identity:
                    raise ValueError("No numeric columns available for hill-climb.")
                c0 = self._rng.choice(list(remaining_identity))
                ind0 = self._make_identity_individual(c0)
                selected_inds.append(ind0)
                selected_names.append(str(ind0))
                remaining_identity.remove(c0)

            best_X = build_X(selected_inds)
            best_score = self._hc_score(best_X, y_np)
            no_improve0 = 0

        state = GAFeatureEngineerDEAP._HCState(
            selected_inds=selected_inds,
            selected_names=selected_names,
            best_X=best_X,
            best_score=best_score,
            no_improve=no_improve0,
        )


        # ---- DEAP fitness: based on HC metric for THIS individual (incremental) ----
        def _eval_hc(individual):
            try:
                feat = self._program_to_numpy(df, individual)
                X_try = np.column_stack([state.best_X, feat])
            except Exception:
                return (1e18,)  # minimize

            mask = np.isfinite(y_np) & np.all(np.isfinite(X_try), axis=1)
            if mask.sum() < max(5, int(0.2 * len(y_np))):
                return (1e18,)

            sc = self._hc_score(X_try[mask], y_np[mask])

            # accept if improves the running HC solution
            if self._hc_improved(sc, state.best_score):
                state.selected_inds.append(individual)
                state.selected_names.append(str(individual))
                state.best_X = X_try
                state.best_score = sc
                state.no_improve = 0


                # optional prune (same semantics as your original)
                if self.hc_prune and len(state.selected_inds) > 1:
                    # Greedy redundancy pruning: drop any feature whose removal doesn't hurt.
                    changed = True
                    while changed and len(state.selected_inds) > 1:
                        changed = False
                        for j in range(len(state.selected_inds)):
                            trial_inds = state.selected_inds[:j] + state.selected_inds[j + 1 :]
                            X_trial = build_X(trial_inds)
                            m2 = np.isfinite(y_np) & np.all(np.isfinite(X_trial), axis=1)
                            if m2.sum() < max(5, int(0.2 * len(y_np))):
                                continue
                            sc2 = self._hc_score(X_trial[m2], y_np[m2])
                            not_worse = (
                                sc2 >= state.best_score - self.hc_tol
                                if self.maximize_hc_metric
                                else sc2 <= state.best_score + self.hc_tol
                            )
                            if not_worse:
                                state.selected_inds.pop(j)
                                state.selected_names.pop(j)
                                state.best_X = X_trial
                                state.best_score = sc2
                                changed = True
                                break

                accepted_count = len(state.selected_inds)

                # After acceptance + optional prune, sync HC state onto self so get_hc_checkpoint() works
                self._hc_selected_inds_ = list(state.selected_inds)
                self.selected_programs_ = list(state.selected_names)
                self.selected_fitness_ = float(state.best_score)
                self._hc_no_improve_ = int(state.no_improve)


                               # inside _eval_hc, after acceptance + optional prune
                if self.checkpoint_path is not None and (accepted_count % self.checkpoint_every_accepts == 0):
                    with open(self.checkpoint_path, "w") as f:
                        json.dump(self.get_hc_checkpoint(), f)

            # translate to DEAP FitnessMin
            fit_val = -sc if self.maximize_hc_metric else sc
            return (fit_val + self.parsimony_coefficient * len(individual),)

        self._toolbox.register("evaluate", _eval_hc)

        # ---- run GA (unchanged loop structure) ----
        pop = self._toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(maxsize=self.hall_of_fame)

        # initial eval
        for ind, fit in zip(pop, map(self._toolbox.evaluate, pop)):
            ind.fitness.values = fit
        hof.update(pop)

        best_hist = [hof[0].fitness.values[0]]
        no_improve_ga = 0

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
                    no_improve_ga += 1
                else:
                    no_improve_ga = 0
                if no_improve_ga >= self.early_stop_rounds:
                    break

        # ---- store hybrid outputs ----
        self.hof_ = hof
        self.best_programs_ = [str(ind) for ind in hof]
        self.best_fitness_ = [float(ind.fitness.values[0]) for ind in hof]
        self.best_feature_names_ = self._dedupe_names(
            [self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in hof]
        )

        # HC-selected (accepted) programs
        self.selected_programs_ = list(state.selected_names)
        raw = [self._individual_to_symbolic(ind) for ind in state.selected_inds]
        raw = [self._sanitize_feature_name(s) for s in raw]
        self.selected_feature_names_ = self._dedupe_names(raw)
        self.selected_fitness_ = float(state.best_score)
        self._hc_selected_inds_ = list(state.selected_inds)

        self._fitted = True
        if self.checkpoint_path is not None:
          with open(self.checkpoint_path, "w") as f:
              json.dump(self.get_hc_checkpoint(), f)

        return self

    # -------------------------
    # fit(): route modes
    # -------------------------
    def fit(
            self,
            X: Any,
            y: Any,
            target_col: Optional[str] = None,
        ):

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

        # rebuild GP primitives now that columns/TE are known
        self._pset = None
        self._toolbox = None
        self._ensure_deap()

        if self.search_mode in ("hill_climb", "ga_hill_climb"):
            # smoke test: hc_metric signature
            if self.hc_metric is None:
                raise ValueError("hc_metric must be provided when search_mode is hill_climb or ga_hill_climb.")
            try:
                _ = float(self.hc_metric(np.zeros((len(y_np), 1)), y_np))
            except TypeError as e:
                raise TypeError(
                    "In hill climbing modes, hc_metric must have signature hc_metric(X, y)."
                ) from e

        if self.search_mode == "hill_climb":
            return self._fit_hill_climb(df, y_np)  # keep your existing implementation

        if self.search_mode == "ga_hill_climb":

            return self._fit_ga_hill_climb(
                df,
                y_np,

            )

        # --------------------
        # Pure GA mode (unchanged): metric(y, y_pred)
        # --------------------
        if self.metric is not None:
            # smoke test ga metric
            try:
                _ = float(self.metric(np.zeros(3), np.zeros(3)))
            except TypeError as e:
                raise TypeError(
                    "In ga mode, metric must have signature metric(y_true, y_pred)."
                ) from e

        def _eval_ga(individual):
            compiled = gp.compile(expr=individual, pset=self._pset)
            out = compiled() if callable(compiled) else compiled

            lf = df
            if out.pre_cols:
                for name, e in out.pre_cols:
                    lf = lf.with_columns(e.alias(name))

            pred = (
                lf.select(out.expr.alias("__pred__"))
                .to_series()
                .to_numpy()
                .astype(float, copy=False)
            )

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
        self.best_feature_names_ = self._dedupe_names(
            [self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in hof]
        )
        self.best_fitness_ = [float(ind.fitness.values[0]) for ind in hof]

        self._fitted = True
        return self

    # -------------------------
    # transform(): keep your existing logic, but in ga_hill_climb we emit selected HC features
    # -------------------------
    def transform(self, X: Any) -> pl.DataFrame:
        df = self._to_polars(X)
        if self._te_maps and self._te_target_col is not None:
            df = self._apply_target_encoders(df)

        self._ensure_deap()

        if self.search_mode in ("hill_climb", "ga_hill_climb"):
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

        # pure GA: emit HOF
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



    def _compile_to_numexpr(self, individual) -> NumE:
        self._ensure_deap()
        compiled = gp.compile(expr=individual, pset=self._pset)
        out = compiled() if callable(compiled) else compiled
        if not isinstance(out, NumE):
            raise TypeError("Program did not compile to NumE.")
        return out


    def _program_to_numpy(self, df: pl.DataFrame, individual) -> np.ndarray:
        out = self._compile_to_numexpr(individual)

        lf = df.lazy()

        if out.pre_cols:
            lf = lf.with_columns_seq([e.alias(name) for name, e in out.pre_cols])

        lf = lf.select(out.expr.alias("__feat__"))

        arr = (
            lf.collect(cluster_with_columns=False)
              .to_series()
              .to_numpy()
              .astype(float, copy=False)
        )
        return arr


    def _to_polars(self, X: Any) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            df = X
        elif pd is not None and isinstance(X, pd.DataFrame):
            df_pd = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
            df = pl.from_pandas(df_pd)
        elif isinstance(X, np.ndarray):
            if not self.numeric_cols:
                raise ValueError("If passing a numpy array, you must provide numeric_cols in __init__.")
            if X.shape[1] != len(self.numeric_cols):
                raise ValueError("numpy array column count != len(numeric_cols).")
            df = pl.DataFrame({c: X[:, i] for i, c in enumerate(self.numeric_cols)})
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        for c in self.index_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required index column '{c}'.")
        return df


    def _infer_numeric_cols(self, df: pl.DataFrame) -> List[str]:
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
                      ((pl.col("__m") * pl.col("__n") + smooth * self._te_global_mean)
                      / (pl.col("__n") + smooth)).alias(tec)
                  )
                  .select([cat, tec])
            )
            self._te_maps[cat] = agg


    def _apply_target_encoders(self, df: pl.DataFrame) -> pl.DataFrame:
        assert self._te_target_col is not None
        assert self._te_global_mean is not None
        out = df
        for cat, mdf in self._te_maps.items():
            tec = self._te_colname(cat, self._te_target_col)
            out = (
                out.join(mdf, on=cat, how="left")
                  .with_columns(pl.col(tec).fill_null(self._te_global_mean))
            )
        return out


    def _dedupe_names(self, names: list[str]) -> list[str]:
        seen = {}
        out = []
        for n in names:
            k = n
            if k in seen:
                seen[k] += 1
                k = f"{k}__{seen[n]}"
            else:
                seen[k] = 0
            out.append(k)
        return out


    def _sanitize_feature_name(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^0-9a-zA-Z_()\[\],.=+\-*/|:]", "_", s)
        if len(s) > self.max_colname_len:
            suffix = hex(abs(hash(s)) % (16**6))[2:]
            keep = max(8, self.max_colname_len - (2 + len(suffix)))
            s = f"{s[:keep]}__{suffix}"
        return s



    def _individual_to_symbolic(self, individual) -> str:
        """
        Convert a DEAP Individual / PrimitiveTree into a human-readable symbolic string
        using the registered formatter functions.
        """
        # DEAP Individuals are iterable over nodes (Primitive / Terminal)
        nodes = list(individual)
        stack: list[str] = []

        for node in reversed(nodes):  # process prefix tree right-to-left
            if isinstance(node, gp.Terminal):
                # Terminals in typed GP often stringify to their "name" already.
                # If yours stringify to something verbose, you can use node.name when available.
                stack.append(str(node.name))
                continue

            if isinstance(node, gp.Primitive):
                args = [stack.pop() for _ in range(node.arity)]  # already left->right
                op_name = node.name.split("__", 1)[1] if "__" in node.name else node.name

                if node.name.startswith("el__"):
                    spec = self._ops_elementary.get(op_name)
                elif node.name.startswith("gb__"):
                    spec = self._ops_groupby.get(op_name)
                elif node.name.startswith("ts__"):
                    spec = self._ops_ts.get(op_name)
                elif node.name.startswith("te__"):
                    spec = self._ops_target.get(op_name)
                else:
                    spec = None

                if spec is not None:
                    stack.append(spec.formatter(args, {}))
                else:
                    stack.append(f"{op_name}({', '.join(args)})")
                continue

            raise TypeError(f"Unknown node type inside tree: {type(node)}")

        if len(stack) != 1:
            # If this triggers, something is inconsistent in the tree representation.
            raise ValueError(f"Could not render individual; stack={stack}")

        return stack[0]

    def _register_default_ops(self) -> None:
        def fmt(name: str, args: List[str], params: Dict[str, Any]) -> str:
            if not params:
                return f"{name}({', '.join(args)})"
            p = ",".join([f"{k}={params[k]}" for k in sorted(params)])
            return f"{name}[{p}]({', '.join(args)})"

        def safe_div(a: pl.Expr, b: pl.Expr, eps: float = 1e-12) -> pl.Expr:
            return pl.when(b.abs() > eps).then(a / b).otherwise(None)

        def rolling_min_key(method: Callable) -> str:
            params = set(inspect.signature(method).parameters.keys())
            return "min_samples" if "min_samples" in params else "min_periods"

        # ---- elementary ----
        self.add_elementary_op("add", 2, builder=lambda eng, a, p: a[0] + a[1], formatter=lambda args, p: fmt("add", args, p))
        self.add_elementary_op("sub", 2, builder=lambda eng, a, p: a[0] - a[1], formatter=lambda args, p: fmt("sub", args, p))
        self.add_elementary_op("mul", 2, builder=lambda eng, a, p: a[0] * a[1], formatter=lambda args, p: fmt("mul", args, p))
        self.add_elementary_op("div", 2, builder=lambda eng, a, p: safe_div(a[0], a[1]), formatter=lambda args, p: fmt("div", args, p))
        self.add_elementary_op("neg", 1, builder=lambda eng, a, p: -a[0], formatter=lambda args, p: fmt("neg", args, p))
        self.add_elementary_op("abs", 1, builder=lambda eng, a, p: a[0].abs(), formatter=lambda args, p: fmt("abs", args, p))
        self.add_elementary_op("sqrt_abs", 1, builder=lambda eng, a, p: a[0].abs().sqrt(), formatter=lambda args, p: fmt("sqrt_abs", args, p))
        self.add_elementary_op("log1p_abs", 1, builder=lambda eng, a, p: a[0].abs().log1p(), formatter=lambda args, p: fmt("log1p_abs", args, p))
        self.add_elementary_op("clip", 1, builder=lambda eng, a, p: a[0].clip(p.get("lo", -3.0), p.get("hi", 3.0)), formatter=lambda args, p: fmt("clip", args, p))

        self.add_elementary_op(
            "identity", 1,
            builder=lambda eng, a, p: a[0],
            formatter=lambda args, p: fmt("identity", args, p),
        )

        # ---- groupby (defaults to date group) ----
        self.add_groupby_op("cs_rank", 1,
            builder=lambda eng, a, p: a[0].rank(method=p.get("method","average"), descending=p.get("descending",True))
                                  .over(p.get("group_col", eng.index_cols[1])),
            formatter=lambda args, p: fmt("cs_rank", args, p),
        )

        self.add_groupby_op("cs_zscore", 1,
            builder=lambda eng, a, p: safe_div(
                (a[0] - a[0].mean().over(p.get("group_col", eng.index_cols[1]))),
                a[0].std().over(p.get("group_col", eng.index_cols[1]))
            ),
            formatter=lambda args, p: fmt("cs_zscore", args, p),
        )
        self.add_groupby_op("grp_mean", 1,
            builder=lambda eng, a, p: a[0].mean().over(p.get("group_col", eng.index_cols[1])),
            formatter=lambda args, p: fmt("grp_mean", args, p),
        )

        # ---- time-series (over ticker) ----
        min_key = rolling_min_key(pl.Expr.rolling_mean)

        def ts_over_ticker(expr: pl.Expr, eng: "GAFeatureEngineerDEAP") -> pl.Expr:
            return expr.over(eng.index_cols[0])

        self.add_ts_op("ts_lag", 1, builder=lambda eng, a, p: ts_over_ticker(a[0].shift(int(p.get("n",1))), eng),
                      formatter=lambda args, p: fmt("ts_lag", args, p))
        self.add_ts_op("ts_diff", 1, builder=lambda eng, a, p: ts_over_ticker(a[0].diff(int(p.get("n",1))), eng),
                      formatter=lambda args, p: fmt("ts_diff", args, p))
        self.add_ts_op("ts_pct_change", 1, builder=lambda eng, a, p: ts_over_ticker(a[0].pct_change(int(p.get("n",1))), eng),
                      formatter=lambda args, p: fmt("ts_pct_change", args, p))
        self.add_ts_op("ts_mean", 1, builder=lambda eng, a, p: ts_over_ticker(
                          a[0].rolling_mean(window_size=int(p.get("window",5)), **{min_key:int(p.get("min_samples",1))}), eng),
                      formatter=lambda args, p: fmt("ts_mean", args, p))
        self.add_ts_op("ts_std", 1, builder=lambda eng, a, p: ts_over_ticker(
                          a[0].rolling_std(window_size=int(p.get("window",5)), **{min_key:int(p.get("min_samples",2))}), eng),
                      formatter=lambda args, p: fmt("ts_std", args, p))
        self.add_ts_op("ts_ewm_mean", 1, builder=lambda eng, a, p: ts_over_ticker(
                          a[0].ewm_mean(span=float(p.get("span",10.0)), adjust=bool(p.get("adjust",False)), ignore_nulls=bool(p.get("ignore_nulls",False))), eng),
                      formatter=lambda args, p: fmt("ts_ewm_mean", args, p))

        if self.enable_impostor_op:
            self.add_elementary_op(
                "impostor_noise", 0,
                builder=lambda eng, a, p: eng._impostor_expr(seed=int(p.get("seed", eng.random_state))),
                formatter=lambda args, p: fmt("impostor_noise", [], p),
            )


    def _new_tmp_col(self, base: str) -> str:
      self._tmp_col_counter += 1
      return self._safe_ident(f"__tmp__{base}__{self._tmp_col_counter}")

    # -------------------------
    # KEEP create_operator SYNTAX + 4 OP TYPES
    # -------------------------
    def create_operator(
        self,
        operator_function: Callable[..., pl.Expr],
        arity: int,
        name: str,
        operator_type: OpType,
        cat_col_args: Optional[List[int]] = None,
        num_col_args: Optional[List[int]] = None,
        target_col_arg: Optional[int] = None,
    ) -> None:
        cat_col_args = cat_col_args or []
        num_col_args = num_col_args or []

        allowed: set[str] = {"element-wise", "groupby", "time-series", "target_encoding"}
        if operator_type not in allowed:
            raise ValueError(f"operator_type must be one of {sorted(allowed)}")
        if not isinstance(arity, int) or arity < 0:
            raise ValueError("arity must be a non-negative int")

        all_pos = list(cat_col_args) + list(num_col_args)
        if target_col_arg is not None:
            all_pos.append(target_col_arg)

        for p in all_pos:
            if not isinstance(p, int):
                raise TypeError("arg positions must be integers (0-based)")
            if p < 0 or p >= arity:
                raise ValueError(f"arg position {p} out of bounds for arity={arity}")
        if len(set(all_pos)) != len(all_pos):
            raise ValueError("cat/num/target arg positions must be disjoint")

        # guards to preserve your conceptual op types
        if operator_type == "element-wise":
            if cat_col_args:
                raise ValueError("element-wise ops cannot use categorical args")
            if target_col_arg is not None:
                raise ValueError("element-wise ops cannot use target_col_arg")

        if operator_type == "groupby":
            if target_col_arg is not None:
                raise ValueError("groupby ops cannot use target_col_arg")
            if len(cat_col_args) == 0:
                raise ValueError("groupby ops must have at least one categorical arg (group key)")
            if len(num_col_args) == 0:
                raise ValueError("groupby ops must have at least one numeric arg (value)")

        if operator_type == "time-series":
            if cat_col_args:
                raise ValueError("time-series ops cannot use categorical args (initially)")
            if target_col_arg is not None:
                raise ValueError("time-series ops cannot use target_col_arg")

        if operator_type == "target_encoding":
            if target_col_arg is None:
                raise ValueError("target_encoding ops must set target_col_arg")
            if len(cat_col_args) == 0:
                raise ValueError("target_encoding ops must have at least one categorical arg")

        spec = OperatorSpec(
            name=name,
            arity=arity,
            operator_type=operator_type,
            fn=operator_function,
            cat_pos=tuple(cat_col_args),
            num_pos=tuple(num_col_args),
            target_pos=target_col_arg,
        )
        self._custom_ops[name] = spec

        kind_map = {
            "element-wise": "elementary",
            "groupby": "groupby",
            "time-series": "ts",
            "target_encoding": "target",
        }
        kind = kind_map[operator_type]

        def _builder(eng: "GAFeatureEngineerDEAP", arg_exprs: List[pl.Expr], p: Dict[str, Any]) -> pl.Expr:
            kwargs = {k: v for k, v in p.items() if k != "bound_cols"}
            return operator_function(*arg_exprs, **kwargs)

        def _formatter(args: List[str], p: Dict[str, Any]) -> str:
            pp = {k: v for k, v in p.items() if k != "bound_cols"}
            if not pp:
                return f"{name}({', '.join(args)})"
            param_str = ",".join([f"{k}={pp[k]}" for k in sorted(pp)])
            return f"{name}[{param_str}]({', '.join(args)})"

        self.add_custom_operation(
            name=name,
            arity=arity,
            kind=kind,
            builder=_builder,
            formatter=_formatter,
            restrict_group_cols_to_categoricals=(operator_type == "groupby"),
        )

        # new primitives => rebuild pset/toolbox next time
        self._pset = None
        self._toolbox = None

    # registry helpers
    def add_elementary_op(self, name: str, arity: int, builder, formatter) -> None:
        self._ops_elementary[name] = OperationSpec(name=name, arity=arity, kind="elementary", builder=builder, formatter=formatter)

    def add_groupby_op(self, name: str, arity: int, builder, formatter, restrict_group_cols_to_categoricals: bool = False) -> None:
        self._ops_groupby[name] = OperationSpec(name=name, arity=arity, kind="groupby", builder=builder, formatter=formatter,
                                                restrict_group_cols_to_categoricals=restrict_group_cols_to_categoricals)

    def add_ts_op(self, name: str, arity: int, builder, formatter) -> None:
        self._ops_ts[name] = OperationSpec(name=name, arity=arity, kind="ts", builder=builder, formatter=formatter)

    def add_target_op(self, name: str, arity: int, builder, formatter) -> None:
        self._ops_target[name] = OperationSpec(name=name, arity=arity, kind="target", builder=builder, formatter=formatter)

    def add_custom_operation(self, name: str, arity: int, kind: str, builder, formatter=None, restrict_group_cols_to_categoricals: bool = False) -> None:
        if formatter is None:
            formatter = lambda args, p: f"{name}({', '.join(args)})"
        if kind == "elementary":
            self.add_elementary_op(name, arity, builder, formatter)
        elif kind == "groupby":
            self.add_groupby_op(name, arity, builder, formatter, restrict_group_cols_to_categoricals)
        elif kind == "ts":
            self.add_ts_op(name, arity, builder, formatter)
        elif kind == "target":
            self.add_target_op(name, arity, builder, formatter)
        else:
            raise ValueError("kind must be one of: elementary, groupby, ts, target")

    # -------------------------
    # Target encoding: turned into GP primitives automatically
    # -------------------------
    def _te_colname(self, cat_col: str, target_col: str) -> str:
        return f"__te__mean__{cat_col}__{target_col}"

    # -------------------------
    # DEAP primitive set build
    # -------------------------
    def _safe_ident(self, s: str) -> str:
        s2 = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(s))
        if not s2 or s2[0].isdigit():
            s2 = "c_" + s2
        return s2

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        """
        Build a typed DEAP primitive set that produces a NumE (numeric Polars expression).

        Key points:
          - Column references must be *terminals* (pset.addTerminal), not 0-arity primitives.
          - Random constants should be *ephemeral terminals* (pset.addEphemeralConstant).
          - Operators remain primitives returning NumE.
        """
        # No GP input args: compiled programs are called with no parameters
        pset = gp.PrimitiveSetTyped("MAIN", [], NumE)

        # -------------------------
        # Terminals: numeric columns
        # -------------------------
        for c in self.numeric_cols:
            term = NumE(pl.col(c).cast(pl.Float64))
            pset.addTerminal(term, NumE, name=self._safe_ident(f"num__{c}"))


        # -----------------------------
        # Terminals: categorical columns
        # -----------------------------
        for c in self.categorical_cols:
            term = CatE(pl.col(c).cast(pl.Utf8))
            pset.addTerminal(term, CatE, name=self._safe_ident(f"cat__{c}"))


        # 1. Helper function that runs during eval() to create the actual NumE
        def _create_lit(x: float) -> NumE:
            return NumE(pl.lit(x))

        # Register it in the pset context so the compiled code can find it
        pset.context["_create_lit"] = _create_lit

        # 2. Wrapper class to control the string representation
        class LiteralWrapper:
            def __init__(self, val: float):
                self.val = val

            def __str__(self):
                return f"_create_lit({self.val})"

            __repr__ = __str__

        # 3. Add ephemeral constant returning the wrapper, but typed as NumE
        pset.addEphemeralConstant(
            "const",
            lambda: LiteralWrapper(self._rng.uniform(-1.0, 1.0)),
            NumE,
        )

        # -------------------------
        # Helper to add operations
        # -------------------------
        def add_op(
            tag: str,
            op: OperationSpec,
            arg_tys: List[type],
            default_params: Optional[Dict[str, Any]] = None,
        ) -> None:
            dp = dict(default_params or {})

            def _is_literal_expr(e: pl.Expr) -> bool:
                """
                True if this expr references no root columns (e.g. pl.lit(0.5)).
                We use Polars meta introspection when available.
                """
                try:
                    # For literal-only expressions, root_names() is empty
                    return len(e.meta.root_names()) == 0
                except Exception:
                    # If Polars meta isn't available/compatible, fall back to False
                    # (you can switch this to True for maximum safety at the cost of more temp cols)
                    return False

            def _prim(*args):
                # args are NumE/CatE wrappers
                pre_cols: List[Tuple[str, pl.Expr]] = []
                expr_args: List[pl.Expr] = []

                for a in args:
                    if isinstance(a, NumE):
                        pre_cols.extend(list(a.pre_cols))
                        expr_args.append(a.expr)
                    else:  # CatE
                        expr_args.append(a.expr)

                # ------------------------------------------------------------
                # FIX: Polars cannot aggregate/window on a pure literal.
                # If a ts/groupby op gets a literal argument, materialize it as
                # a temporary column first, then pass pl.col(tmp) to the builder.
                # ------------------------------------------------------------
                if op.kind in ("ts", "groupby"):
                    for j, ex in enumerate(expr_args):
                        if _is_literal_expr(ex):
                            tmp_in = self._new_tmp_col(f"{tag}__{op.name}__arg{j}")
                            pre_cols.append((tmp_in, ex))
                            expr_args[j] = pl.col(tmp_in)

                expr = op.builder(self, expr_args, dp)

                # MATERIALIZE: treat ts + groupby ops as staged expressions
                if op.kind in ("ts", "groupby"):
                    tmp = self._new_tmp_col(f"{tag}__{op.name}")
                    pre_cols.append((tmp, expr))
                    return NumE(pl.col(tmp), tuple(pre_cols))

                return NumE(expr, tuple(pre_cols))

            name = self._safe_ident(f"{tag}__{op.name}")
            pset.addPrimitive(_prim, arg_tys, NumE, name=name)

        # -------------------------
        # Element-wise ops (NumE -> NumE)
        # -------------------------
        for op in self._ops_elementary.values():
            add_op("el", op, [NumE] * op.arity)

        # -------------------------
        # Groupby ops
        #   - built-ins numeric-only
        #   - custom groupby ops respect Cat/Num positions
        # -------------------------
        for op in self._ops_groupby.values():
            if (
                op.name in self._custom_ops
                and self._custom_ops[op.name].operator_type == "groupby"
            ):
                spec = self._custom_ops[op.name]
                arg_tys = [(CatE if i in spec.cat_pos else NumE) for i in range(spec.arity)]
                add_op("gb", op, arg_tys)
            else:
                add_op("gb", op, [NumE] * op.arity)

        # -------------------------
        # Time-series ops (numeric-only)
        # -------------------------
        ts_defaults = {
            "ts_lag": {"n": 1},
            "ts_diff": {"n": 1},
            "ts_pct_change": {"n": 1},
            "ts_mean": {"window": 5, "min_samples": 1},
            "ts_std": {"window": 5, "min_samples": 2},
            "ts_ewm_mean": {"span": 10.0, "adjust": False, "ignore_nulls": True},
        }
        for op in self._ops_ts.values():
            add_op("ts", op, [NumE] * op.arity, default_params=ts_defaults.get(op.name, {}))

        # -------------------------
        # Target-encoding terminals + custom TE ops
        # -------------------------
        # These TE columns exist only after fit() has built self._te_maps; at that point
        # we can expose them as numeric terminals.
        if self._te_target_col is not None and self._te_maps:
            for cat in self.categorical_cols:
                tec = self._te_colname(cat, self._te_target_col)
                # Terminal that references the materialized TE column
                pset.addTerminal(
                    NumE(pl.col(tec).cast(pl.Float64)),
                    NumE,
                    name=self._safe_ident(f"te_mean__{cat}"),
                )

        # Custom target_encoding ops: typed via cat/num/target slots
        for op in self._ops_target.values():
            if (
                op.name in self._custom_ops
                and self._custom_ops[op.name].operator_type == "target_encoding"
            ):
                spec = self._custom_ops[op.name]
                arg_tys = [(CatE if i in spec.cat_pos else NumE) for i in range(spec.arity)]
                add_op("te", op, arg_tys)

        return pset


    def _ensure_deap(self):

        import builtins

        def _create_lit(x: float) -> NumE:
            return NumE(pl.lit(x))

        builtins._create_lit = _create_lit

        if self._pset is None:
            self._pset = self._build_pset()

        if self._toolbox is None:
            # global creator registry (avoid redef)
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=self.init_min_depth, max_=self.init_max_depth)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
            toolbox.register("mate", gp.cxOnePoint)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self._pset)

            toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))
            toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))

            self._toolbox = toolbox


    def get_hc_checkpoint(self) -> dict:
        if not hasattr(self, "_hc_selected_inds_"):

            raise RuntimeError("No hill-climb state to checkpoint.")

        return {
            "selected_programs": [str(ind) for ind in self._hc_selected_inds_],
            "best_score": float(self.selected_fitness_),
            "no_improve": getattr(self, "_hc_no_improve_", 0),
        }


    def load_hc_checkpoint(self, checkpoint: dict, df: pl.DataFrame, y_np: np.ndarray):
        self._ensure_deap()

        inds = [
            creator.Individual(
                gp.PrimitiveTree.from_string(expr, self._pset)
            )
            for expr in checkpoint["selected_programs"]
        ]

        def build_X(ind_list):
            cols = [self._program_to_numpy(df, ind) for ind in ind_list]
            return np.column_stack(cols) if cols else np.empty((df.height, 0))

        X = build_X(inds)
        score = checkpoint["best_score"]

        self._hc_selected_inds_ = inds
        self.selected_programs_ = checkpoint["selected_programs"]
        self.selected_feature_names_ = self._dedupe_names(
            [self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in inds]
        )
        self.selected_fitness_ = score

        return X, score
