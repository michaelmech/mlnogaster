import json

import numpy as np
import polars as pl
from deap import creator, gp

from GAfeatureengineer import GAFeatureEngineerDEAP


def _tiny_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "date": [1, 2, 1, 2, 1, 2],
            "f1": [1.0, 2.0, 1.5, 2.5, 0.5, 3.0],
            "f2": [2.0, 1.0, 2.5, 1.5, 3.0, 0.5],
            "sector": ["x", "x", "y", "y", "z", "z"],
        }
    )


def test_ga_mode_uses_custom_metric():
    df = _tiny_df()
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)
    seen = {"calls": 0}

    def metric(y_true, y_pred):
        seen["calls"] += 1
        return float(np.mean(np.abs(y_true - y_pred)))

    eng = GAFeatureEngineerDEAP(
        metric=metric,
        search_mode="ga",
        population_size=8,
        generations=1,
        hall_of_fame=2,
        random_state=7,
    )

    eng.fit(df, y)
    out = eng.transform(df)

    assert seen["calls"] > 0
    assert out.height == df.height
    assert set(["ticker", "date", "f1", "f2"]).issubset(set(out.columns))
    assert len(eng.best_programs_) == 2


def test_custom_operator_can_be_compiled_and_evaluated():
    df = _tiny_df()
    y = np.array([1.0, 1.3, 1.8, 2.1, 2.5, 2.9], dtype=float)

    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=6,
        generations=1,
        hall_of_fame=2,
        random_state=11,
    )

    eng.create_operator(
        operator_function=lambda x: x * x,
        arity=1,
        name="square",
        operator_type="element-wise",
        num_col_args=[0],
    )
    eng.fit(df, y)

    ind = creator.Individual(gp.PrimitiveTree.from_string("el__square(num__f1)", eng._pset))
    vals = eng._program_to_numpy(df, ind)
    np.testing.assert_allclose(vals, np.array(df["f1"]) ** 2)


def test_hill_climb_checkpoint_write_and_resume(tmp_path):
    df = _tiny_df()
    y = np.array([1.0, 1.4, 1.9, 2.2, 2.3, 2.7], dtype=float)
    ckpt = tmp_path / "hc_checkpoint.json"

    def hc_metric(X, y_true):
        # Minimize squared error against linear target proxy
        return float(np.mean((X.sum(axis=1) - y_true) ** 2))

    eng = GAFeatureEngineerDEAP(
        search_mode="hill_climb",
        hc_metric=hc_metric,
        population_size=6,
        generations=1,
        hall_of_fame=2,
        checkpoint_path=str(ckpt),
        checkpoint_every_accepts=1,
        random_state=23,
    )
    eng.fit(df, y)

    assert ckpt.exists()
    payload = json.loads(ckpt.read_text())
    assert "selected_programs" in payload
    assert "best_score" in payload
    assert len(payload["selected_programs"]) >= 1

    # Warm-start from existing checkpoint file should run cleanly.
    eng2 = GAFeatureEngineerDEAP(
        search_mode="hill_climb",
        hc_metric=hc_metric,
        population_size=6,
        generations=1,
        hall_of_fame=2,
        checkpoint_path=str(ckpt),
        checkpoint_every_accepts=1,
        random_state=23,
    )
    eng2.fit(df, y)
    out = eng2.transform(df)
    assert out.height == df.height
    assert len(eng2.selected_programs_) >= 1
