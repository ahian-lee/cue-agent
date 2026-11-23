"""A benefit probability is not the requested expected incremental utility."""
from cue_agent.selector import CUEv0, FEATURE_NAMES
import pytest


def test_delta_estimate_distinguishes_same_benefit_frequency_but_opposite_value():
    rows = []
    low = {name: 0.0 for name in FEATURE_NAMES}
    high = {**low, "experience_covers_all_actions": 1.0}
    # Both groups have 50% positive deltas. Their expected values have
    # opposite signs, so a P(delta>0) classifier cannot serve as delta_hat.
    for index in range(3):
        for label, features, values in (("low", low, (.1, -1.0)), ("high", high, (1.0, -.1))):
            for replicate, delta in enumerate(values):
                rows.append({"task_id": f"fixture-{label}-{index}-{replicate}",
                    "features": dict(features), "delta": delta})
    model = CUEv0().fit(rows)
    assert model.predict_interval(low)["delta_hat"] < 0, "Expected delta is negative; a positive-class probability is the wrong target."
    assert model.predict_interval(high)["delta_hat"] > 0


def test_constant_y1_target_remains_a_valid_same_model_ablation():
    low = {name: 0.0 for name in FEATURE_NAMES}
    high = {**low, "experience_covers_all_actions": 1.0}
    rows = [{"pair_id": f"ablation-{index}", "features": dict(low if index < 6 else high),
        "delta": 0.0 if index < 6 else 1.0, "y1": 1.0} for index in range(12)]
    delta_model = CUEv0().fit(rows, target_key="delta")
    y1_model = CUEv0().fit(rows, target_key="y1")
    assert delta_model.model is not None
    assert y1_model.model is not None, "Constant Y1 is a valid regression target, not missing data."
    assert y1_model.predict_interval(low)["point_estimate"] == pytest.approx(1.0)
    assert y1_model.predict_interval(high)["point_estimate"] == pytest.approx(1.0)


def test_constant_delta_is_observed_value_instead_of_missing_data():
    features = {name: 0.0 for name in FEATURE_NAMES}
    rows = [{"pair_id": f"constant-delta-{index}", "features": dict(features),
        "delta": .2} for index in range(12)]
    model = CUEv0().fit(rows)
    assert model.model is not None, "A regression can estimate a constant target; lack of heterogeneity is not missing data."
    assert model.predict_interval(features)["delta_hat"] == pytest.approx(.2)


def test_positive_prediction_interval_cannot_override_negative_mean_delta():
    features = {name: 0.0 for name in FEATURE_NAMES}
    # Conditional distribution at this feature vector: +.01 with p=.99,
    # -1 with p=.01. Mean=-.0001, but a 90% prediction interval can be
    # positive while covering 99% of individual outcomes. These are code
    # fixtures and do not enter the real research dataset.
    train = [{"pair_id": f"fit-{index}", "features": dict(features),
        "delta": -1.0 if index == 0 else .01} for index in range(100)]
    calibration = [{"pair_id": f"cal-{index}", "features": dict(features),
        "delta": -1.0 if index % 100 == 0 else .01} for index in range(200)]
    model = CUEv0().fit(train)
    assert model.predict_interval(features)["delta_hat"] < 0
    model.calibrate(calibration, alpha=.1, seed=0)
    result = model.decide(features)
    assert result["decision"] == "abstain", "A single-outcome prediction interval is not a conditional-mean confidence bound."


def test_global_residual_mean_is_not_a_query_conditional_mean_bound():
    common = {name: 0.0 for name in FEATURE_NAMES}
    rare = {**common, "experience_covers_all_actions": 1.0}
    # The true mean is linear: .2 - 1.2*x. Training has not observed x=1;
    # independent calibration contains a rare harmful x=1 group. A global
    # residual bias interval stays positive for both groups and must not be
    # promoted to a query-specific confidence bound.
    train = [{"pair_id": f"global-fit-{index}", "task_id": f"fit-task-{index}",
        "source_group": f"fit-source-{index}", "features": dict(common),
        "delta": .1 if index % 2 == 0 else .3} for index in range(100)]
    calibration = [{"pair_id": f"global-cal-{index}", "task_id": f"cal-task-{index}",
        "source_group": f"cal-source-{index}",
        "features": dict(rare if index % 100 == 0 else common),
        "delta": -1.0 if index % 100 == 0 else .2} for index in range(1000)]
    model = CUEv0().fit(train)
    model.calibrate(calibration, alpha=.1, seed=0)
    assert model.decide(rare)["decision"] != "apply", "A marginal bias interval cannot certify the rare group's negative conditional mean."
