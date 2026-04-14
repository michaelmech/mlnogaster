import inspect
from typing import Any, Callable, Dict, List

import polars as pl


def register_default_ops(engine) -> None:
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

    def maybe_over(expr: pl.Expr, group_col: Any) -> pl.Expr:
        return expr.over(group_col) if group_col else expr

    def default_group_col(eng) -> Any:
        return eng.categorical_cols[0] if getattr(eng, "categorical_cols", None) else None

    engine.add_elementary_op("add", 2, builder=lambda eng, a, p: a[0] + a[1], formatter=lambda args, p: fmt("add", args, p))
    engine.add_elementary_op("sub", 2, builder=lambda eng, a, p: a[0] - a[1], formatter=lambda args, p: fmt("sub", args, p))
    engine.add_elementary_op("mul", 2, builder=lambda eng, a, p: a[0] * a[1], formatter=lambda args, p: fmt("mul", args, p))
    engine.add_elementary_op("div", 2, builder=lambda eng, a, p: safe_div(a[0], a[1]), formatter=lambda args, p: fmt("div", args, p))
    engine.add_elementary_op("neg", 1, builder=lambda eng, a, p: -a[0], formatter=lambda args, p: fmt("neg", args, p))
    engine.add_elementary_op("abs", 1, builder=lambda eng, a, p: a[0].abs(), formatter=lambda args, p: fmt("abs", args, p))
    engine.add_elementary_op("sqrt_abs", 1, builder=lambda eng, a, p: a[0].abs().sqrt(), formatter=lambda args, p: fmt("sqrt_abs", args, p))
    engine.add_elementary_op("log1p_abs", 1, builder=lambda eng, a, p: a[0].abs().log1p(), formatter=lambda args, p: fmt("log1p_abs", args, p))
    engine.add_elementary_op("clip", 1, builder=lambda eng, a, p: a[0].clip(p.get("lo", -3.0), p.get("hi", 3.0)), formatter=lambda args, p: fmt("clip", args, p))

    engine.add_elementary_op("identity", 1, builder=lambda eng, a, p: a[0], formatter=lambda args, p: fmt("identity", args, p))

    engine.add_groupby_op(
        "cs_rank",
        1,
        builder=lambda eng, a, p: maybe_over(
            a[0].rank(method=p.get("method", "average"), descending=p.get("descending", True)),
            p.get("group_col", default_group_col(eng)),
        ),
        formatter=lambda args, p: fmt("cs_rank", args, p),
    )

    engine.add_groupby_op(
        "cs_zscore",
        1,
        builder=lambda eng, a, p: safe_div(
            a[0] - maybe_over(a[0].mean(), p.get("group_col", default_group_col(eng))),
            maybe_over(a[0].std(), p.get("group_col", default_group_col(eng))),
        ),
        formatter=lambda args, p: fmt("cs_zscore", args, p),
    )
    engine.add_groupby_op(
        "grp_mean",
        1,
        builder=lambda eng, a, p: maybe_over(a[0].mean(), p.get("group_col", default_group_col(eng))),
        formatter=lambda args, p: fmt("grp_mean", args, p),
    )

    min_key = rolling_min_key(pl.Expr.rolling_mean)

    def ts_over_entity(expr: pl.Expr, p: Dict[str, Any]) -> pl.Expr:
        return maybe_over(expr, p.get("entity_col"))

    engine.add_ts_op(
        "ts_lag",
        1,
        builder=lambda eng, a, p: ts_over_entity(a[0].shift(int(p.get("n", 1))), p),
        formatter=lambda args, p: fmt("ts_lag", args, p),
    )
    engine.add_ts_op(
        "ts_diff",
        1,
        builder=lambda eng, a, p: ts_over_entity(a[0].diff(int(p.get("n", 1))), p),
        formatter=lambda args, p: fmt("ts_diff", args, p),
    )
    engine.add_ts_op(
        "ts_pct_change",
        1,
        builder=lambda eng, a, p: ts_over_entity(a[0].pct_change(int(p.get("n", 1))), p),
        formatter=lambda args, p: fmt("ts_pct_change", args, p),
    )
    engine.add_ts_op(
        "ts_mean",
        1,
        builder=lambda eng, a, p: ts_over_entity(
            a[0].rolling_mean(window_size=int(p.get("window", 5)), **{min_key: int(p.get("min_samples", 1))}), p
        ),
        formatter=lambda args, p: fmt("ts_mean", args, p),
    )
    engine.add_ts_op(
        "ts_std",
        1,
        builder=lambda eng, a, p: ts_over_entity(
            a[0].rolling_std(window_size=int(p.get("window", 5)), **{min_key: int(p.get("min_samples", 2))}), p
        ),
        formatter=lambda args, p: fmt("ts_std", args, p),
    )
    engine.add_ts_op(
        "ts_ewm_mean",
        1,
        builder=lambda eng, a, p: ts_over_entity(
            a[0].ewm_mean(
                span=float(p.get("span", 10.0)),
                adjust=bool(p.get("adjust", False)),
                ignore_nulls=bool(p.get("ignore_nulls", False)),
            ),
            p,
        ),
        formatter=lambda args, p: fmt("ts_ewm_mean", args, p),
    )

    if engine.enable_impostor_op:
        engine.add_elementary_op(
            "impostor_noise",
            0,
            builder=lambda eng, a, p: eng._impostor_expr(seed=int(p.get("seed", eng.random_state))),
            formatter=lambda args, p: fmt("impostor_noise", [], p),
        )
