from typing import TYPE_CHECKING

import difflib
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

    def _age_penalty(self, birth_generation: int, current_generation: int) -> float:
        if not self.enable_aging or self.aging_penalty <= 0:
            return 0.0
        age = max(0, int(current_generation) - int(birth_generation))
        half_life = max(1e-9, float(self.aging_half_life_generations))
        normalized = age / half_life
        return float(self.aging_penalty * (1.0 - np.exp(-normalized)))

    def _novelty_penalty(self, individual, reference_programs: list[str]) -> float:
        if not self.enable_fitness_sharing or self.fitness_sharing_strength <= 0:
            return 0.0
        if not reference_programs:
            return 0.0

        candidate = str(individual)
        max_similarity = max(
            difflib.SequenceMatcher(a=candidate, b=ref).ratio() for ref in reference_programs
        )
        return float(self.fitness_sharing_strength * max_similarity)

    def _population_diversity_ratio(self, pop) -> float:
        if not pop:
            return 0.0
        unique = len({str(ind) for ind in pop})
        return float(unique / len(pop))

    def _adaptive_rates(self, pop) -> tuple[float, int]:
        if not self.enable_adaptive_rates:
            return float(self.mutpb), int(self.tournament_size)

        diversity = self._population_diversity_ratio(pop)
        mut_span = max(0.0, self.adaptive_mutation_max - self.adaptive_mutation_min)
        p_mut = self.adaptive_mutation_max - mut_span * diversity

        tour_span = max(0, self.adaptive_tournament_max - self.adaptive_tournament_min)
        tourn = self.adaptive_tournament_min + int(round(tour_span * diversity))
        tourn = max(2, min(tourn, len(pop) if len(pop) > 1 else 2))
        return float(p_mut), int(tourn)

    def _lineage_overlap_ratio(self, lineage_a: tuple[int, ...], lineage_b: tuple[int, ...]) -> float:
        if not lineage_a or not lineage_b:
            return 0.0
        sa = set(lineage_a)
        sb = set(lineage_b)
        return float(len(sa & sb) / max(1, min(len(sa), len(sb))))

    def _incest_penalty(self, individual) -> float:
        if not self.enable_incest_penalty or self.incest_penalty_strength <= 0:
            return 0.0
        coeff = float(getattr(individual, "inbreeding_coeff", 0.0))
        if coeff < self.incest_similarity_threshold:
            return 0.0
        return float(self.incest_penalty_strength * coeff)
