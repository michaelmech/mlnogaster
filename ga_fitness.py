from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from GAfeatureengineer import GAFeatureEngineerDEAP


class FitnessMixin:
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
