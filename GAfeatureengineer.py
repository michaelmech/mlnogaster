import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

try:
    from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
except Exception:  # pragma: no cover
    class BaseEstimator:
        pass

    class TransformerMixin:
        pass

    class RegressorMixin:
        pass

from deap import base, gp, tools

from feature_engineering_types import OperationSpec, OperatorSpec, SearchMode
from ga_backend import BackendMixin
from ga_fitness import FitnessMixin
from ga_hill_climb import HillClimbMixin
from ga_operations import OperationsMixin
from ga_organisms import OrganismMixin


class GAFeatureEngineerDEAP(
    FitnessMixin,
    OrganismMixin,
    OperationsMixin,
    HillClimbMixin,
    BackendMixin,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
):
    def __init__(
        self,
        index_cols: Tuple[str, str] = ("ticker", "date"),
        cat_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        max_colname_len: int = 160,
        random_state: int = 0,
        target_encoding_smoothing: float = 10.0,
        enable_impostor_op: bool = False,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        maximize_metric: bool = False,
        hc_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        maximize_hc_metric: Optional[bool] = None,
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
        hc_start_with_one_identity: bool = True,
        hc_prune: bool = True,
        hc_tol: float = 1e-12,
        checkpoint_path=None,
        checkpoint_every_accepts=np.inf,
        search_mode: SearchMode = "ga",
        verbose: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every_accepts = checkpoint_every_accepts
        self.index_cols = index_cols
        if cat_cols is not None and categorical_cols is not None:
            raise ValueError("Pass only one of cat_cols or categorical_cols.")
        resolved_cat_cols = cat_cols if cat_cols is not None else categorical_cols
        self.categorical_cols = list(resolved_cat_cols) if resolved_cat_cols else []
        self.numeric_cols: List[str] = []
        self.max_colname_len = int(max_colname_len)
        self.random_state = int(random_state)
        self.target_encoding_smoothing = float(target_encoding_smoothing)
        self.enable_impostor_op = bool(enable_impostor_op)
        self._tmp_col_counter = 0
        self.verbose = bool(verbose)

        self.metric = metric
        self.maximize_metric = bool(maximize_metric)
        if isinstance(self.metric, (list, tuple)):
            raise TypeError("Pass exactly one GA metric callable (not a list/tuple).")
        if self.metric is not None and not callable(self.metric):
            raise TypeError("metric must be a callable or None.")

        self.hc_metric = hc_metric
        if self.hc_metric is not None and not callable(self.hc_metric):
            raise TypeError("hc_metric must be a callable or None.")
        self.maximize_hc_metric = bool(self.maximize_metric if maximize_hc_metric is None else maximize_hc_metric)

        self.search_mode: SearchMode = search_mode

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

        self._custom_ops: Dict[str, OperatorSpec] = {}
        self._ops_elementary: Dict[str, OperationSpec] = {}
        self._ops_groupby: Dict[str, OperationSpec] = {}
        self._ops_ts: Dict[str, OperationSpec] = {}
        self._ops_target: Dict[str, OperationSpec] = {}

        self._te_maps: Dict[str, pl.DataFrame] = {}
        self._te_global_mean: Optional[float] = None
        self._te_target_col: Optional[str] = None

        self.hc_start_with_one_identity = bool(hc_start_with_one_identity)
        self.hc_prune = bool(hc_prune)
        self.hc_tol = float(hc_tol)

        self.selected_programs_: Optional[List[str]] = None
        self.selected_fitness_: Optional[float] = None
        self.selected_feature_names_: Optional[List[str]] = None

        self.hof_: Optional[tools.HallOfFame] = None
        self.best_programs_: Optional[List[str]] = None
        self.best_fitness_: Optional[List[float]] = None
        self.best_feature_names_: Optional[List[str]] = None

        self._pset: Optional[gp.PrimitiveSetTyped] = None
        self._toolbox: Optional[base.Toolbox] = None
        self._fitted: bool = False

        self._register_default_ops()
