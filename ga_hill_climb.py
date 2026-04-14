import json
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import polars as pl
from deap import creator, gp, tools


class HillClimbMixin:
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

    def _checkpoint_write_due(self, accepted_count: int) -> bool:
        if self.checkpoint_path is None:
            return False
        every = self.checkpoint_every_accepts
        if every is None:
            return False
        try:
            every_f = float(every)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(every_f) or every_f <= 0:
            return False
        every_i = int(every_f)
        return accepted_count > 0 and (accepted_count % every_i == 0)

    def _load_checkpoint_file(self) -> Optional[dict]:
        if self.checkpoint_path is None:
            return None
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if "selected_programs" not in data or "best_score" not in data:
            return None
        return data

    def _fit_hill_climb(self, df: pl.DataFrame, y_np: np.ndarray):
        self._ensure_deap()
        self._debug_log("Starting hill-climb fit.")

        def _patch_legacy_constants(expr: str) -> str:
            # Older checkpoints may contain constants rendered as `_create_lit`
            # or `_create_lit(<number>)`. Newer DEAP versions parse terminal tokens
            # via `eval(token)` and will fail on bare function names. Convert both
            # forms into plain numeric literals so they are always parseable.
            out = re.sub(
                r"_create_lit\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)",
                r"\1",
                expr,
            )
            return re.sub(r"(?<![\w.])_create_lit(?![\w(])", "0.0", out)

        def build_X(ind_list: List[Any]) -> np.ndarray:
            cols = [self._program_to_numpy(df, ind) for ind in ind_list]
            return np.column_stack(cols) if cols else np.empty((df.height, 0), dtype=float)

        resume_checkpoint = self._load_checkpoint_file()
        if resume_checkpoint is not None:
            self._debug_log("Loading hill-climb state from checkpoint.")
            selected_inds = []
            selected_names = []
            for expr in resume_checkpoint["selected_programs"]:
                patched = _patch_legacy_constants(expr)
                try:
                    ind = creator.Individual(gp.PrimitiveTree.from_string(patched, self._pset))
                except (TypeError, ValueError):
                    self._debug_log(f"Skipping unparsable checkpoint expression: {expr}")
                    continue
                selected_inds.append(ind)
                selected_names.append(patched)

            if not selected_inds:
                self._debug_log("Checkpoint expressions were invalid; reinitializing from scratch.")
                resume_checkpoint = None

        if resume_checkpoint is not None:

            best_X = build_X(selected_inds)
            best_score = float(resume_checkpoint["best_score"])

            mask = np.isfinite(y_np) & np.all(np.isfinite(best_X), axis=1)
            best_score = self._hc_score(best_X[mask], y_np[mask])

            no_improve0 = int(resume_checkpoint.get("no_improve", 0))
        else:
            self._debug_log("No checkpoint found; initializing fresh hill-climb state.")
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
        self._debug_log(
            f"Initial hill-climb score={best_score:.6f}, selected_features={len(selected_inds)}"
        )

        state = HillClimbMixin._HCState(
            selected_inds=selected_inds,
            selected_names=selected_names,
            best_X=best_X,
            best_score=best_score,
            no_improve=no_improve0,
        )

        def _eval_hc(individual):
            try:
                feat = self._program_to_numpy(df, individual)
                X_try = np.column_stack([state.best_X, feat])
            except Exception:
                return (1e18,)

            mask = np.isfinite(y_np) & np.all(np.isfinite(X_try), axis=1)
            if mask.sum() < max(5, int(0.2 * len(y_np))):
                return (1e18,)

            sc = self._hc_score(X_try[mask], y_np[mask])

            if self._hc_improved(sc, state.best_score):
                self._debug_log(
                    f"Hill-climb accepted individual: score {state.best_score:.6f} -> {sc:.6f}"
                )
                state.selected_inds.append(individual)
                state.selected_names.append(str(individual))
                state.best_X = X_try
                state.best_score = sc
                state.no_improve = 0

                if self.hc_prune and len(state.selected_inds) > 1:
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
                                self._debug_log(
                                    f"Hill-climb pruned feature at index {j}; new score={sc2:.6f}"
                                )
                                break

                accepted_count = len(state.selected_inds)

                self._hc_selected_inds_ = list(state.selected_inds)
                self.selected_programs_ = list(state.selected_names)
                self.selected_fitness_ = float(state.best_score)
                self._hc_no_improve_ = int(state.no_improve)

                if self._checkpoint_write_due(accepted_count):
                    with open(self.checkpoint_path, "w") as f:
                        json.dump(self.get_hc_checkpoint(), f)

            fit_val = -sc if self.maximize_hc_metric else sc
            return (fit_val + self.parsimony_coefficient * len(individual),)

        self._toolbox.register("evaluate", _eval_hc)

        pop = self._toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(maxsize=self.hall_of_fame)

        for ind, fit in zip(pop, map(self._toolbox.evaluate, pop)):
            ind.fitness.values = fit
        hof.update(pop)
        self._debug_log(f"Initial hill-climb GA stage evaluated: population={len(pop)}")

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
            self._debug_log(f"Hill-climb generation {_gen}: best={cur_best:.6f}")

            if self.early_stop_rounds is not None:
                if len(best_hist) >= 2 and (best_hist[-2] - best_hist[-1]) <= self.early_stop_tol:
                    no_improve_ga += 1
                else:
                    no_improve_ga = 0
                if no_improve_ga >= self.early_stop_rounds:
                    self._debug_log(f"Hill-climb early stopping at generation {_gen}.")
                    break

        self.hof_ = hof
        self.best_programs_ = [str(ind) for ind in hof]
        self.best_fitness_ = [float(ind.fitness.values[0]) for ind in hof]
        self.best_feature_names_ = self._dedupe_names([
            self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in hof
        ])

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
            self._debug_log(f"Wrote hill-climb checkpoint to {self.checkpoint_path}.")

        return self

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

        def _patch_legacy_constants(expr: str) -> str:
            out = re.sub(
                r"_create_lit\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)",
                r"\1",
                expr,
            )
            return re.sub(r"(?<![\w.])_create_lit(?![\w(])", "0.0", out)

        inds = [
            creator.Individual(gp.PrimitiveTree.from_string(_patch_legacy_constants(expr), self._pset))
            for expr in checkpoint["selected_programs"]
        ]

        def build_X(ind_list):
            cols = [self._program_to_numpy(df, ind) for ind in ind_list]
            return np.column_stack(cols) if cols else np.empty((df.height, 0))

        X = build_X(inds)
        score = checkpoint["best_score"]

        self._hc_selected_inds_ = inds
        self.selected_programs_ = checkpoint["selected_programs"]
        self.selected_feature_names_ = self._dedupe_names([
            self._sanitize_feature_name(self._individual_to_symbolic(ind)) for ind in inds
        ])
        self.selected_fitness_ = score

        return X, score
