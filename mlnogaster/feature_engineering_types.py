from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import polars as pl

Arg = Union[str, "FeatureNode"]
OpType = Literal["element-wise", "groupby", "time-series", "target_encoding"]
SearchMode = Literal["ga", "hill_climb"]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    operator_type: OpType
    fn: Callable
    cat_pos: tuple[int, ...]
    num_pos: tuple[int, ...]
    target_pos: Optional[int] = None
    online: bool = False


@dataclass(frozen=True)
class OperationSpec:
    name: str
    arity: int
    kind: str
    builder: Callable[[Any, List[pl.Expr], Dict[str, Any]], pl.Expr]
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
