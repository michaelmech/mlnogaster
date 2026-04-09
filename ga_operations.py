import operator
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from deap import base, creator, gp, tools

from feature_engineering_custom_operations import (
    add_custom_operation as _add_custom_operation,
    add_elementary_op as _add_elementary_op,
    add_groupby_op as _add_groupby_op,
    add_target_op as _add_target_op,
    add_ts_op as _add_ts_op,
    create_operator as _create_operator,
)
from feature_engineering_default_operations import register_default_ops
from feature_engineering_types import CatE, NumE, OpType, OperationSpec


class OperationsMixin:
    def _register_default_ops(self) -> None:
        register_default_ops(self)

    def _new_tmp_col(self, base: str) -> str:
        self._tmp_col_counter += 1
        return self._safe_ident(f"__tmp__{base}__{self._tmp_col_counter}")

    def create_operator(
        self,
        operator_function: Callable[..., pl.Expr],
        arity: int,
        name: str,
        operator_type: OpType,
        cat_col_args: Optional[List[int]] = None,
        target_col_arg: Optional[int] = None,
        online: bool = False,
    ) -> None:
        _create_operator(
            self,
            operator_function=operator_function,
            arity=arity,
            name=name,
            operator_type=operator_type,
            cat_col_args=cat_col_args,
            target_col_arg=target_col_arg,
            online=online,
        )

    def add_elementary_op(self, name: str, arity: int, builder, formatter) -> None:
        _add_elementary_op(self, name=name, arity=arity, builder=builder, formatter=formatter)

    def add_groupby_op(self, name: str, arity: int, builder, formatter, restrict_group_cols_to_categoricals: bool = False) -> None:
        _add_groupby_op(
            self,
            name=name,
            arity=arity,
            builder=builder,
            formatter=formatter,
            restrict_group_cols_to_categoricals=restrict_group_cols_to_categoricals,
        )

    def add_ts_op(self, name: str, arity: int, builder, formatter) -> None:
        _add_ts_op(self, name=name, arity=arity, builder=builder, formatter=formatter)

    def add_target_op(self, name: str, arity: int, builder, formatter) -> None:
        _add_target_op(self, name=name, arity=arity, builder=builder, formatter=formatter)

    def add_custom_operation(self, name: str, arity: int, kind: str, builder, formatter=None, restrict_group_cols_to_categoricals: bool = False) -> None:
        _add_custom_operation(
            self,
            name=name,
            arity=arity,
            kind=kind,
            builder=builder,
            formatter=formatter,
            restrict_group_cols_to_categoricals=restrict_group_cols_to_categoricals,
        )

    def _te_colname(self, cat_col: str, target_col: str) -> str:
        return f"__te__mean__{cat_col}__{target_col}"

    def _safe_ident(self, s: str) -> str:
        s2 = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(s))
        if not s2 or s2[0].isdigit():
            s2 = "c_" + s2
        return s2

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        pset = gp.PrimitiveSetTyped("MAIN", [], NumE)
        self._debug_log("Building primitive set.")

        for c in self.numeric_cols:
            term = NumE(pl.col(c).cast(pl.Float64))
            pset.addTerminal(term, NumE, name=self._safe_ident(f"num__{c}"))
        self._debug_log(f"Registered numeric terminals: {len(self.numeric_cols)}")

        for c in self.categorical_cols:
            term = CatE(pl.col(c).cast(pl.Utf8))
            pset.addTerminal(term, CatE, name=self._safe_ident(f"cat__{c}"))
        self._debug_log(f"Registered categorical terminals: {len(self.categorical_cols)}")

        def _create_lit(x: float) -> NumE:
            return NumE(pl.lit(x))

        pset.context["_create_lit"] = _create_lit

        class LiteralWrapper:
            def __init__(self, val: float):
                self.val = val

            def __str__(self):
                return f"_create_lit({self.val})"

            __repr__ = __str__

        pset.addEphemeralConstant(
            "const",
            lambda: LiteralWrapper(self._rng.uniform(-1.0, 1.0)),
            NumE,
        )

        def add_op(tag: str, op: OperationSpec, arg_tys: List[type], default_params: Optional[Dict[str, Any]] = None) -> None:
            dp = dict(default_params or {})

            def _is_literal_expr(e: pl.Expr) -> bool:
                try:
                    return len(e.meta.root_names()) == 0
                except Exception:
                    return False

            def _prim(*args):
                pre_cols: List[tuple[str, pl.Expr]] = []
                expr_args: List[pl.Expr] = []

                for a in args:
                    if isinstance(a, NumE):
                        pre_cols.extend(list(a.pre_cols))
                        expr_args.append(a.expr)
                    else:
                        expr_args.append(a.expr)

                if op.kind in ("ts", "groupby"):
                    for j, ex in enumerate(expr_args):
                        if _is_literal_expr(ex):
                            tmp_in = self._new_tmp_col(f"{tag}__{op.name}__arg{j}")
                            pre_cols.append((tmp_in, ex))
                            expr_args[j] = pl.col(tmp_in)

                expr = op.builder(self, expr_args, dp)

                if op.kind in ("ts", "groupby"):
                    tmp = self._new_tmp_col(f"{tag}__{op.name}")
                    pre_cols.append((tmp, expr))
                    return NumE(pl.col(tmp), tuple(pre_cols))

                return NumE(expr, tuple(pre_cols))

            name = self._safe_ident(f"{tag}__{op.name}")
            pset.addPrimitive(_prim, arg_tys, NumE, name=name)

        for op in self._ops_elementary.values():
            add_op("el", op, [NumE] * op.arity)

        for op in self._ops_groupby.values():
            if op.name in self._custom_ops and self._custom_ops[op.name].operator_type == "groupby":
                spec = self._custom_ops[op.name]
                arg_tys = [(CatE if i in spec.cat_pos else NumE) for i in range(spec.arity)]
                add_op("gb", op, arg_tys)
            else:
                add_op("gb", op, [NumE] * op.arity)

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

        if self._te_target_col is not None and self._te_maps:
            for cat in self.categorical_cols:
                tec = self._te_colname(cat, self._te_target_col)
                pset.addTerminal(
                    NumE(pl.col(tec).cast(pl.Float64)),
                    NumE,
                    name=self._safe_ident(f"te_mean__{cat}"),
                )

        for op in self._ops_target.values():
            if op.name in self._custom_ops and self._custom_ops[op.name].operator_type == "target_encoding":
                spec = self._custom_ops[op.name]
                arg_tys = [(CatE if i in spec.cat_pos else NumE) for i in range(spec.arity)]
                add_op("te", op, arg_tys)

        self._debug_log(
            "Primitive set complete: "
            f"elementary={len(self._ops_elementary)}, groupby={len(self._ops_groupby)}, "
            f"timeseries={len(self._ops_ts)}, target={len(self._ops_target)}"
        )
        return pset

    def _ensure_deap(self):
        if self._pset is None:
            self._pset = self._build_pset()
            self._debug_log("Primitive set initialized.")

        if self._toolbox is None:
            def _reset_creator_type(name: str) -> None:
                if hasattr(creator, name):
                    delattr(creator, name)

            if self.enable_multi_objective:
                reset_mo = False
                if hasattr(creator, "FitnessMinMO") and len(creator.FitnessMinMO.weights) != 2:
                    reset_mo = True
                if hasattr(creator, "IndividualMO"):
                    probe = creator.IndividualMO([])
                    if len(probe.fitness.weights) != 2:
                        reset_mo = True
                if reset_mo:
                    _reset_creator_type("IndividualMO")
                    _reset_creator_type("FitnessMinMO")
                if not hasattr(creator, "FitnessMinMO"):
                    creator.create("FitnessMinMO", base.Fitness, weights=(-1.0, -1.0))
                if not hasattr(creator, "IndividualMO"):
                    creator.create("IndividualMO", gp.PrimitiveTree, fitness=creator.FitnessMinMO)
                individual_cls = creator.IndividualMO
            else:
                reset_single = False
                if hasattr(creator, "FitnessMin") and len(creator.FitnessMin.weights) != 1:
                    reset_single = True
                if hasattr(creator, "Individual"):
                    probe = creator.Individual([])
                    if len(probe.fitness.weights) != 1:
                        reset_single = True
                if reset_single:
                    _reset_creator_type("Individual")
                    _reset_creator_type("FitnessMin")
                if not hasattr(creator, "FitnessMin"):
                    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                if not hasattr(creator, "Individual"):
                    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
                individual_cls = creator.Individual

            toolbox = base.Toolbox()
            toolbox.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=self.init_min_depth, max_=self.init_max_depth)
            toolbox.register("individual", tools.initIterate, individual_cls, toolbox.expr)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
            toolbox.register("mate", gp.cxOnePoint)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self._pset)

            toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))
            toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_height))

            self._toolbox = toolbox
            self._debug_log(
                "DEAP toolbox initialized "
                f"(multi_objective={self.enable_multi_objective}, init_depth=[{self.init_min_depth}, {self.init_max_depth}])."
            )
