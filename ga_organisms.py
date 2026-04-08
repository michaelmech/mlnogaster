import re
from typing import Any, List

import numpy as np
import polars as pl
from deap import gp

from feature_engineering_types import NumE


class OrganismMixin:
    def _impostor_expr(self, seed: int):
        return pl.int_range(0, pl.len()).shuffle(seed=seed + int(self._rng.randint(0, 10_000))).cast(pl.Float64)

    def _compile_to_numexpr(self, individual) -> NumE:
        self._ensure_deap()
        self._debug_log(f"Compiling individual: {individual}")
        compiled = gp.compile(expr=individual, pset=self._pset)
        out = compiled() if callable(compiled) else compiled
        if not isinstance(out, NumE):
            raise TypeError("Program did not compile to NumE.")
        self._debug_log(
            f"Compiled individual with pre_cols={len(out.pre_cols)} and expr={out.expr}"
        )
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
        self._debug_log(
            f"Program evaluated to numpy array: shape={arr.shape}, finite={int(np.isfinite(arr).sum())}"
        )
        return arr

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
        nodes = list(individual)
        stack: list[str] = []

        for node in reversed(nodes):
            if isinstance(node, gp.Terminal):
                stack.append(str(node.name))
                continue

            if isinstance(node, gp.Primitive):
                args = [stack.pop() for _ in range(node.arity)]
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
            raise ValueError(f"Could not render individual; stack={stack}")

        return stack[0]
