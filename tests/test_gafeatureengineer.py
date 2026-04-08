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
    )
    eng.fit(df, y)

    ind = creator.Individual(gp.PrimitiveTree.from_string("el__square(num__f1)", eng._pset))
    vals = eng._program_to_numpy(df, ind)
    np.testing.assert_allclose(vals, np.array(df["f1"]) ** 2)


def test_numeric_inference_excludes_string_columns_when_cat_cols_not_set():
    df = _tiny_df()
    y = np.array([1.0, 1.3, 1.8, 2.1, 2.5, 2.9], dtype=float)

    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=6,
        generations=1,
        hall_of_fame=2,
        random_state=11,
    )
    eng.fit(df, y)

    assert "f1" in eng.numeric_cols
    assert "f2" in eng.numeric_cols
    assert "sector" not in eng.numeric_cols


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


def test_target_encoding_unseen_categories_are_null_on_transform():
    df_train = _tiny_df()
    y_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)

    eng = GAFeatureEngineerDEAP(
        cat_cols=["sector"],
        search_mode="ga",
        population_size=6,
        generations=1,
        hall_of_fame=1,
        random_state=13,
    )
    eng.fit(df_train, y_train)

    df_new = pl.DataFrame(
        {
            "ticker": ["A", "D"],
            "date": [3, 3],
            "f1": [1.1, 2.2],
            "f2": [2.2, 1.1],
            "sector": ["x", "new_sector"],
        }
    )
    out = eng.transform(df_new)

    te_col = eng._te_colname("sector", eng._te_target_col)
    te_vals = out.select(te_col).to_series().to_list()

    assert te_vals[0] is not None
    assert te_vals[1] is None


def test_create_operator_inferrs_numeric_args_and_online_flag():
    eng = GAFeatureEngineerDEAP(search_mode="ga", population_size=4, generations=1, hall_of_fame=1)

    eng.create_operator(
        operator_function=lambda c, x, y: x + y,
        arity=3,
        name="grp_custom",
        operator_type="groupby",
        cat_col_args=[0],
    )
    spec = eng._custom_ops["grp_custom"]
    assert spec.cat_pos == (0,)
    assert spec.num_pos == (1, 2)

    eng.create_operator(
        operator_function=lambda c, y: y,
        arity=2,
        name="te_custom",
        operator_type="target_encoding",
        cat_col_args=[0],
        target_col_arg=1,
        online=True,
    )
    te_spec = eng._custom_ops["te_custom"]
    assert te_spec.online is True


def test_inspiration_mapped_aging_penalty_grows_with_generation():
    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=4,
        generations=1,
        hall_of_fame=1,
        enable_aging=True,
        aging_penalty=0.2,
        aging_half_life_generations=4,
    )

    p0 = eng._age_penalty(birth_generation=0, current_generation=0)
    p4 = eng._age_penalty(birth_generation=0, current_generation=4)
    p8 = eng._age_penalty(birth_generation=0, current_generation=8)

    assert p0 == 0.0
    assert p8 > p4 > p0


def test_inspiration_mapped_fitness_sharing_penalizes_duplicates():
    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=4,
        generations=1,
        hall_of_fame=1,
        enable_fitness_sharing=True,
        fitness_sharing_strength=0.5,
    )

    candidate = "el__identity(num__f1)"
    same = eng._novelty_penalty(candidate, [candidate])
    different = eng._novelty_penalty(candidate, ["el__identity(num__f2)"])

    assert same > 0.45
    assert same > different >= 0.0


def test_adaptive_rates_raise_mutation_when_diversity_is_low():
    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=4,
        generations=1,
        hall_of_fame=1,
        enable_adaptive_rates=True,
        adaptive_mutation_min=0.1,
        adaptive_mutation_max=0.9,
        adaptive_tournament_min=2,
        adaptive_tournament_max=6,
    )

    low_div_pop = ["p", "p", "p", "p"]
    high_div_pop = ["p1", "p2", "p3", "p4"]
    low_mut, low_tourn = eng._adaptive_rates(low_div_pop)
    high_mut, high_tourn = eng._adaptive_rates(high_div_pop)

    assert low_mut > high_mut
    assert low_tourn < high_tourn


def test_incest_penalty_kicks_in_only_above_threshold():
    eng = GAFeatureEngineerDEAP(
        search_mode="ga",
        population_size=4,
        generations=1,
        hall_of_fame=1,
        enable_incest_penalty=True,
        incest_penalty_strength=0.7,
        incest_similarity_threshold=0.5,
    )

    class Dummy:
        inbreeding_coeff = 0.6

    class DummyLow:
        inbreeding_coeff = 0.4

    assert eng._incest_penalty(Dummy()) > 0.0
    assert eng._incest_penalty(DummyLow()) == 0.0
