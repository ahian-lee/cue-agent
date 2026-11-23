"""CUE-v0 implementation tests; synthetic rows only, no research evidence."""
from __future__ import annotations

import pytest

from cue_agent.metrics import Decision
from cue_agent.selector import (CUEv0, FEATURE_NAMES, MODEL_FEATURE_NAMES, _feature_vector, _split_conformal_radius, bootstrap_placeholder,
                    fit_label_ablation, project_features)


def test_too_few_rows_refuses_to_fit_and_abstains():
    rows = [{"features": {}, "delta": 1.0}, {"features": {}, "delta": 0.0}]
    policy = CUEv0().fit(rows)
    assert policy.status == "insufficient"
    assert policy.decide({})["decision"] == Decision.ABSTAIN.value
    assert bootstrap_placeholder(rows)["lower"] is None


def test_constant_delta_is_a_valid_observed_target():
    rows = [{"pair_id": f"constant-{i}", "features": {}, "delta": 0.0} for i in range(8)]
    policy = CUEv0().fit(rows)
    assert policy.status == "fit_uncalibrated"
    assert policy.model is not None
    assert policy.predict_interval({})["delta_hat"] == pytest.approx(0.0)
    assert policy.decide({})["decision"] == Decision.ABSTAIN.value


def test_fit_without_independent_calibration_still_abstains():
    rows = []
    for i in range(4):
        rows.append({"features": {"experience_covers_all_actions": 1.0, "visible_order_count": 3}, "delta": 1.0})
        rows.append({"features": {"experience_covers_all_actions": 0.0, "visible_order_count": 2}, "delta": 0.0})
    policy = CUEv0().fit(rows)
    assert policy.status == "fit_uncalibrated"
    out = policy.decide({"experience_covers_all_actions": 1.0, "visible_order_count": 3})
    assert out["decision"] == Decision.ABSTAIN.value
    assert out["interval"]["lower"] is None


def test_feature_projection_requires_full_action_coverage():
    features = project_features(
        requested_actions=["modify_address", "modify_items"],
        experience_covered_actions=["modify_address", "modify_user_address"],
        experience_kind="positive_mechanism",
        visible_order_count=2,
        has_cheapest_request=True,
        address_withheld=True,
    )
    assert features["experience_covers_all_actions"] == 0.0
    assert features["has_cheapest_request"] == 1.0
    assert features["address_withheld"] == 1.0
    assert features["same_user_as_source"] is None


def test_unknown_feature_is_not_the_same_input_as_observed_zero():
    missing = _feature_vector({"same_user_as_source": None})
    zero = _feature_vector({"same_user_as_source": 0})
    omitted = _feature_vector({})
    index = FEATURE_NAMES.index("same_user_as_source")
    assert len(missing) == len(MODEL_FEATURE_NAMES) == 2 * len(FEATURE_NAMES)
    assert missing[index] == zero[index] == 0.0
    assert missing[len(FEATURE_NAMES) + index] == 1.0
    assert zero[len(FEATURE_NAMES) + index] == 0.0
    assert missing == omitted


def test_missingness_can_be_learned_without_claiming_a_valid_bound():
    rows = []
    for i in range(30):
        rows.append({"pair_id": f"known-zero-{i}", "features": {"same_user_as_source": 0}, "delta": -1.0})
        rows.append({"pair_id": f"unknown-{i}", "features": {"same_user_as_source": None}, "delta": 1.0})
    model = CUEv0().fit(rows)
    zero = model.predict_interval({"same_user_as_source": 0})["delta_hat"]
    unknown = model.predict_interval({"same_user_as_source": None})["delta_hat"]
    assert zero < 0 < unknown
    assert model.fit_summary["missingness_indicators"] is True
    assert model.fit_summary["feature_missing_counts"]["same_user_as_source"] == 30
    assert len(model.fit_summary["coefficients"]) == len(MODEL_FEATURE_NAMES)
    assert model.predict_interval({"same_user_as_source": None})["feature_missingness_support"] == "individual_states_seen"
    assert model.decide({"same_user_as_source": None})["decision"] == Decision.ABSTAIN.value


def test_unseen_missingness_is_reported_without_claiming_a_bound():
    rows = [{"pair_id": f"known-{i}", "features": {"same_user_as_source": 0}, "delta": float(i % 2)}
            for i in range(8)]
    model = CUEv0().fit(rows)
    out = model.predict_interval({"same_user_as_source": None})
    assert out["feature_missingness_support"] == "unseen_training_state"
    assert out["unseen_feature_missingness_states"]["same_user_as_source"] == "missing_not_seen_in_training"
    assert out["decision_eligible"] is False
    assert model.decide({"same_user_as_source": None})["decision"] == Decision.ABSTAIN.value


def test_feature_projection_preserves_unknown_inputs():
    features = project_features(requested_actions=None, experience_covered_actions=["return"],
        experience_kind=None, visible_order_count=None, address_withheld=None)
    assert all(value is None for value in features.values())
    known = project_features(requested_actions=[], experience_covered_actions=[], experience_kind="other",
        visible_order_count=0, has_cheapest_request=False, address_withheld=False, same_user_as_source=False)
    assert all(value == 0.0 for value in known.values())


def test_regression_keeps_sign_and_magnitude_not_positive_frequency():
    low = {name: 0.0 for name in FEATURE_NAMES}
    high = {**low, "experience_covers_all_actions": 1.0}
    rows = []
    for index in range(3):
        for label, features, values in (("low", low, (0.1, -1.0)), ("high", high, (1.0, -0.1))):
            for replicate, delta in enumerate(values):
                rows.append(
                    {
                        "pair_id": f"fixture-{label}-{index}-{replicate}",
                        "features": dict(features),
                        "delta": delta,
                    }
                )
    model = CUEv0().fit(rows)
    low_hat = model.predict_interval(low)["delta_hat"]
    high_hat = model.predict_interval(high)["delta_hat"]
    assert low_hat is not None and high_hat is not None
    assert low_hat < 0 < high_hat
    assert high_hat - low_hat > 0.4, "The regression must retain effect magnitude, not only sign."
    assert model.fit_summary["target"] == "delta"
    assert model.fit_summary["regressor"] == "ridge"


def test_unknown_delta_is_skipped_and_all_unknown_refuses():
    rows = []
    for index in range(8):
        rows.append(
            {
                "pair_id": f"known-{index}",
                "features": {"experience_covers_all_actions": float(index % 2)},
                "delta": 1.0 if index % 2 else -1.0,
            }
        )
    for index in range(4):
        rows.append({"pair_id": f"unknown-{index}", "features": {}, "delta": None})
    model = CUEv0().fit(rows)
    assert model.status == "fit_uncalibrated"
    assert model.n_unknown_target == 4
    assert model.predict_interval({})["delta_hat"] is not None
    empty = CUEv0().fit([{"pair_id": f"u-{i}", "features": {}, "delta": None} for i in range(8)])
    assert empty.status == "insufficient"
    assert empty.model is None
    assert "0 known" in empty.reason


def test_failed_refit_clears_old_model_summary_and_estimate():
    rows = [
        {
            "pair_id": f"fit-{i}",
            "features": {"experience_covers_all_actions": float(i % 2)},
            "delta": 1.0 if i % 2 else -1.0,
        }
        for i in range(8)
    ]
    model = CUEv0().fit(rows)
    assert model.model is not None and model.fit_summary
    model.fit([])
    assert model.status == "insufficient"
    assert model.model is None
    assert model.fit_summary == {}
    assert model.calibration is None
    assert model.predict_interval({})["delta_hat"] is None


def test_label_ablation_uses_same_features_split_and_regressor():
    rows = []
    for index in range(20):
        feature = float(index % 2)
        rows.append(
            {
                "pair_id": f"ablation-{index}",
                "features": {"experience_covers_all_actions": feature},
                "delta": (1.0 if feature else -1.0) + (0.1 if index % 4 == 0 else -0.1),
                "y1": 1.0 if feature else 0.0,
            }
        )
    result = fit_label_ablation(rows, test_fraction=0.3, seed=7)
    assert result["same_features"] is True
    assert result["same_split"] is True
    assert result["split"]["strategy"] == "random_rows_development_only"
    assert result["split"]["group_disjointness_verified"] is False
    assert result["same_model_family"] == "ridge"
    assert set(result["targets"]) == {"delta", "y1"}
    for target in ("delta", "y1"):
        assert result["targets"][target]["fit_status"] == "fit_uncalibrated"
        assert result["targets"][target]["metrics"]["n_evaluated"] == result["n_test"]
    assert result["targets"]["delta"]["fit_summary"]["target"] == "delta"
    assert result["targets"]["y1"]["fit_summary"]["target"] == "y1"


def test_ablation_keeps_repeated_tasks_sources_and_templates_on_one_side():
    rows = [
        {"pair_id": f"{group}-{repeat}", "features": {}, "delta": float(group % 2),
         "y1": 1.0, "task_group": f"t{group}", "source_group": f"s{group}",
         "template_group": f"p{group // 2}"}
        for group in range(6) for repeat in range(2)
    ]
    result = fit_label_ablation(rows, group_by=("task_group", "source_group", "template_group"),
                                min_train=4, min_test=2, seed=7)
    split = result["split"]
    train = [row for row in rows if row["pair_id"] in split["train_pair_ids"]]
    test = [row for row in rows if row["pair_id"] in split["test_pair_ids"]]
    for field in split["group_fields"]:
        assert {row[field] for row in train}.isdisjoint({row[field] for row in test})
    assert split["n_components"] == 3
    assert len(train) + len(test) == len(rows)


def test_ablation_does_not_treat_unknown_or_connected_groups_as_independent():
    rows = [{"pair_id": str(i), "features": {}, "delta": 0.0, "y1": 1.0,
             "task_group": f"t{i}", "source_group": "shared"} for i in range(8)]
    with pytest.raises(ValueError, match="no group-disjoint split"):
        fit_label_ablation(rows, group_by=("task_group", "source_group"))
    rows[0]["source_group"] = None
    with pytest.raises(ValueError, match="unknown group"):
        fit_label_ablation(rows, group_by=("task_group", "source_group"))


def test_calibration_requires_disjoint_ids_and_checks_heldout_coverage():
    train = []
    for index in range(12):
        feature = float(index % 2)
        train.append(
            {
                "pair_id": f"train-{index}",
                "features": {"experience_covers_all_actions": feature},
                "delta": 2.0 * feature - 1.0 + (0.2 if index % 3 == 0 else -0.2),
            }
        )
    calibration = []
    for index in range(24):
        feature = float(index % 2)
        calibration.append(
            {
                "pair_id": f"cal-{index}",
                "features": {"experience_covers_all_actions": feature},
                "delta": 2.0 * feature - 1.0 + (0.2 if index % 3 == 0 else -0.2),
                "source": "fixture-only",
            }
        )
    policy = CUEv0().fit(train)
    assert policy.status == "fit_uncalibrated"
    policy.calibrate(calibration, alpha=0.1, min_rows=20, seed=1, source="fixture-only")
    assert policy.status == "fit_uncalibrated"
    assert policy.calibration is not None
    assert policy.calibration["coverage_ok"] is True
    assert policy.calibration["decision_eligible"] is False
    assert policy.calibration["group_disjointness_verified"] is False
    assert policy.calibration["exchangeability_verified"] is False
    assert policy.calibration["conformal_rank"] == 12
    assert policy.calibration["conditional_mean_bound_valid"] is False
    assert policy.calibration["n_calibration"] >= 10
    high = policy.predict_interval({"experience_covers_all_actions": 1.0})
    assert high["decision_eligible"] is False
    assert high["calibrated"] is False
    assert high["lower"] is None and high["upper"] is None
    assert high["diagnostic_marginal_mean_lower"] is not None
    assert policy.decide({"experience_covers_all_actions": 1.0})["decision"] == Decision.ABSTAIN.value
    low = policy.decide({"experience_covers_all_actions": 0.0})
    assert low["decision"] == Decision.ABSTAIN.value

    with pytest.raises(ValueError, match="overlap"):
        policy.calibrate(
            calibration + [{"pair_id": "train-0", "features": {}, "delta": 0.0}],
            min_rows=20,
        )
    with pytest.raises(ValueError, match="pair_id"):
        policy.calibrate([{"features": {}, "delta": 0.0}] * 20, min_rows=20)


def test_conformal_rank_is_finite_sample_corrected_and_can_be_unbounded():
    radius, rank = _split_conformal_radius([-9.0, 1, 2, 3, 4, 5, 6, 7, 8], alpha=0.1)
    assert rank == 9 and radius == 9.0
    radius, rank = _split_conformal_radius([-9.0, 1, 2, 3, 4, 5, 6, 7, 8], alpha=0.05)
    assert rank == 10 and radius is None


def test_unbounded_conformal_interval_does_not_reuse_a_previous_diagnostic():
    train = [{"pair_id": f"fit-{i}", "features": {}, "delta": float(i % 2)} for i in range(8)]
    calibration = [{"pair_id": f"cal-{i}", "features": {}, "delta": float(i % 2)} for i in range(20)]
    model = CUEv0().fit(train).calibrate(calibration, alpha=0.1)
    assert model.calibration is not None
    model.calibrate(calibration, alpha=0.05)
    assert model.calibration is None
    assert "no finite conformal prediction interval" in model.reason
    assert model.predict_interval({})["diagnostic_prediction_lower"] is None
    assert model.decide({})["decision"] == Decision.ABSTAIN.value


def test_marginal_diagnostic_cannot_unlock_a_query_decision():
    common = {name: 0.0 for name in FEATURE_NAMES}
    rare = {**common, "experience_covers_all_actions": 1.0}
    train = [
        {
            "pair_id": f"global-fit-{index}",
            "features": dict(common),
            "delta": 0.1 if index % 2 == 0 else 0.3,
        }
        for index in range(100)
    ]
    calibration = [
        {
            "pair_id": f"global-cal-{index}",
            "features": dict(rare if index % 100 == 0 else common),
            "delta": -1.0 if index % 100 == 0 else 0.2,
        }
        for index in range(1000)
    ]
    policy = CUEv0().fit(train)
    policy.calibrate(calibration, alpha=0.1, seed=0)
    interval = policy.predict_interval(rare)
    assert interval["decision_eligible"] is False
    assert interval["lower"] is None
    assert interval["diagnostic_marginal_mean_lower"] is not None
    assert policy.decide(rare)["decision"] == Decision.ABSTAIN.value
