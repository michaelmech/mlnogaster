from typing import Callable, Dict, List, Optional

import polars as pl

from feature_engineering_types import OpType, OperatorSpec, OperationSpec


def create_operator(
    engine,
    operator_function: Callable[..., pl.Expr],
    arity: int,
    name: str,
    operator_type: OpType,
    cat_col_args: Optional[List[int]] = None,
    target_col_arg: Optional[int] = None,
    online: bool = False,
) -> None:
    cat_col_args = cat_col_args or []
    online = bool(online)

    allowed: set[str] = {"element-wise", "groupby", "time-series", "target_encoding"}
    if operator_type not in allowed:
        raise ValueError(f"operator_type must be one of {sorted(allowed)}")
    if not isinstance(arity, int) or arity < 0:
        raise ValueError("arity must be a non-negative int")

    all_pos = list(cat_col_args)
    if target_col_arg is not None:
        all_pos.append(target_col_arg)

    for p in all_pos:
        if not isinstance(p, int):
            raise TypeError("arg positions must be integers (0-based)")
        if p < 0 or p >= arity:
            raise ValueError(f"arg position {p} out of bounds for arity={arity}")
    if len(set(all_pos)) != len(all_pos):
        raise ValueError("cat/target arg positions must be disjoint")

    num_col_args = [i for i in range(arity) if i not in set(all_pos)]

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
    elif online:
        raise ValueError("online can only be used with target_encoding operators")

    spec = OperatorSpec(
        name=name,
        arity=arity,
        operator_type=operator_type,
        fn=operator_function,
        cat_pos=tuple(cat_col_args),
        num_pos=tuple(num_col_args),
        target_pos=target_col_arg,
        online=online,
    )
    engine._custom_ops[name] = spec

    kind_map = {
        "element-wise": "elementary",
        "groupby": "groupby",
        "time-series": "ts",
        "target_encoding": "target",
    }
    kind = kind_map[operator_type]

    def _builder(eng, arg_exprs: List[pl.Expr], p: Dict[str, object]) -> pl.Expr:
        kwargs = {k: v for k, v in p.items() if k != "bound_cols"}
        return operator_function(*arg_exprs, **kwargs)

    def _formatter(args: List[str], p: Dict[str, object]) -> str:
        pp = {k: v for k, v in p.items() if k != "bound_cols"}
        if not pp:
            return f"{name}({', '.join(args)})"
        param_str = ",".join([f"{k}={pp[k]}" for k in sorted(pp)])
        return f"{name}[{param_str}]({', '.join(args)})"

    add_custom_operation(
        engine,
        name=name,
        arity=arity,
        kind=kind,
        builder=_builder,
        formatter=_formatter,
        restrict_group_cols_to_categoricals=(operator_type == "groupby"),
    )

    engine._pset = None
    engine._toolbox = None


def add_elementary_op(engine, name: str, arity: int, builder, formatter) -> None:
    engine._ops_elementary[name] = OperationSpec(name=name, arity=arity, kind="elementary", builder=builder, formatter=formatter)


def add_groupby_op(engine, name: str, arity: int, builder, formatter, restrict_group_cols_to_categoricals: bool = False) -> None:
    engine._ops_groupby[name] = OperationSpec(
        name=name,
        arity=arity,
        kind="groupby",
        builder=builder,
        formatter=formatter,
        restrict_group_cols_to_categoricals=restrict_group_cols_to_categoricals,
    )


def add_ts_op(engine, name: str, arity: int, builder, formatter) -> None:
    engine._ops_ts[name] = OperationSpec(name=name, arity=arity, kind="ts", builder=builder, formatter=formatter)


def add_target_op(engine, name: str, arity: int, builder, formatter) -> None:
    engine._ops_target[name] = OperationSpec(name=name, arity=arity, kind="target", builder=builder, formatter=formatter)


def add_custom_operation(engine, name: str, arity: int, kind: str, builder, formatter=None, restrict_group_cols_to_categoricals: bool = False) -> None:
    if formatter is None:
        formatter = lambda args, p: f"{name}({', '.join(args)})"
    if kind == "elementary":
        add_elementary_op(engine, name, arity, builder, formatter)
    elif kind == "groupby":
        add_groupby_op(engine, name, arity, builder, formatter, restrict_group_cols_to_categoricals)
    elif kind == "ts":
        add_ts_op(engine, name, arity, builder, formatter)
    elif kind == "target":
        add_target_op(engine, name, arity, builder, formatter)
    else:
        raise ValueError("kind must be one of: elementary, groupby, ts, target")
