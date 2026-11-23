"""Code checks, not research samples or proof of CUE effectiveness."""
import pytest
from cue_agent.metrics import Decision, PairedOutcome, evaluate_policy, interval_decision


def row(key, task, r0, r1, s0=None, s1=None):
    return PairedOutcome(key, task, "memory", "prefix", 1, r0, r1,
        bool(r0) if s0 is None and r0 is not None else s0,
        bool(r1) if s1 is None and r1 is not None else s1)


def test_success_after_intervention_is_not_incremental_benefit():
    assert row("neutral", "t", 1, 1).delta == 0
    assert row("rescue", "t", 0, 1).delta == 1
    assert row("harm", "t", 1, 0).delta == -1


def test_gate_requires_strictly_positive_lower_bound():
    assert interval_decision(.01, .4) == Decision.APPLY
    assert interval_decision(-.5, -.01) == Decision.REJECT
    assert interval_decision(-.5, 0) == Decision.ABSTAIN
    assert interval_decision(0, 0) == Decision.ABSTAIN
    assert interval_decision(0, .5) == Decision.ABSTAIN
    assert interval_decision(-.2, .2) == Decision.ABSTAIN
    with pytest.raises(ValueError):
        interval_decision(float("nan"), 1)


def test_abstain_and_reject_continue_base_instead_of_awarding_success():
    rows = [row("good_base", "a", 1, 0), row("missed_benefit", "b", 0, 1)]
    result = evaluate_policy(rows, {"good_base": Decision.ABSTAIN, "missed_benefit": Decision.REJECT})
    assert result["selected_success_count"] == 1
    assert result["selected_success_denominator"] == 2
    assert result["observed_reward_increase_intervention_recall"] == 0
    assert result["beneficial_intervention_recall"] is None
    assert result["task_macro_reward_gain_before_decision_cost"] == 0
    assert result["task_macro_reward_gain"] is None
    assert result["decision_cost_total_reward_units"] is None


def test_task_macro_does_not_count_extra_contexts_as_independent_tasks():
    rows = [row("a", "A", 0, 1)] + [row(f"b{i}", "B", 1, 0) for i in range(3)]
    result = evaluate_policy(rows, {r.pair_id: Decision.APPLY for r in rows})
    assert result["contexts"] == 4 and result["task_clusters"] == 2
    assert result["context_mean_reward_gain_before_decision_cost"] == -.5
    assert result["task_macro_reward_gain_before_decision_cost"] == 0
    assert result["context_mean_reward_gain"] is None
    assert result["observed_reward_decrease_after_apply_rate"] == .75
    assert result["negative_reward_transfer_rate"] is None


def test_unknown_counterfactual_is_not_filled_with_zero():
    unknown_base = row("apply", "a", None, 1)
    unknown_experience = row("continue", "b", 1, None)
    result = evaluate_policy([unknown_base, unknown_experience],
        {"apply": Decision.APPLY, "continue": Decision.ABSTAIN})
    assert unknown_base.delta is None and unknown_experience.delta is None
    assert result["known_paired_rewards"] == 0
    assert result["unknown_paired_rewards"] == 2
    assert result["task_macro_reward_gain_before_decision_cost"] == 0
    assert set(result["task_mean_reward_gains_before_decision_cost"]) == {"b"}
    assert result["reward_gain_task_denominator_before_decision_cost"] == 1
    assert result["task_macro_reward_gain"] is None
    assert result["known_reward_gain_contexts"] == 0
    assert result["selected_success_denominator"] == 2
    assert result["base_success_denominator"] == 1
    assert result["paired_context_success_denominator"] == 1
    assert result["paired_context_success_table"]["base1_policy1"] == 1


def test_reward_and_task_success_are_distinct():
    outcome = PairedOutcome("cost", "a", "memory", "prefix", 1, .8, .6, True, True)
    result = evaluate_policy([outcome], {"cost": Decision.APPLY})
    assert result["selected_success_rate"] == 1
    assert result["observed_reward_decrease_after_apply_count"] == 1
    assert result["observed_success_label_decreases_after_apply"] == 0
    assert result["negative_reward_transfer_count"] is None
    assert result["task_regressions_after_apply"] is None


def test_separate_user_utility_can_disagree_with_a_fixed_reference_score():
    score_drops_but_user_value_rises = PairedOutcome("case", "a", "memory", "prefix", 1,
        1, 0, True, False, user_utility_base=.4, user_utility_experience=.7)
    unknown_user_value = PairedOutcome("unknown", "b", "memory", "prefix", 1,
        1, 0, True, False, user_utility_base=.4, user_utility_experience=None)
    result = evaluate_policy([score_drops_but_user_value_rises, unknown_user_value],
        {"case": Decision.APPLY, "unknown": Decision.APPLY})
    assert result["observed_reward_decrease_after_apply_count"] == 2
    assert result["supplied_user_utility_paired_observations"] == 1
    assert result["unknown_user_utility_pairs"] == 1
    assert result["supplied_user_utility_increase_candidates"] == 1
    assert result["supplied_user_utility_increase_intervention_recall"] == 1
    assert result["supplied_user_utility_decrease_after_apply_count"] == 0
    assert result["supplied_user_utility_applied_denominator"] == 1
    assert not result["supplied_user_utility_provenance_verified_by_this_function"]
    assert not result["supplied_user_utility_metrics_include_selector_cost"]
    assert result["negative_reward_transfer_count"] is None


def test_all_user_utility_unknown_does_not_become_zero_harm():
    outcome = PairedOutcome("case", "a", "memory", "prefix", 1, 1, 0, True, False)
    result = evaluate_policy([outcome], {"case": Decision.APPLY})
    assert result["supplied_user_utility_paired_observations"] == 0
    assert result["unknown_user_utility_pairs"] == 1
    assert result["supplied_user_utility_increase_candidates"] is None
    assert result["supplied_user_utility_decrease_after_apply_count"] is None
    assert result["supplied_user_utility_decrease_after_apply_rate"] is None


def test_selector_cost_is_paid_even_when_rejecting_or_abstaining():
    rows = [row("reject", "a", 1, 0), row("abstain", "b", 1, 1), row("apply", "c", 0, 1)]
    decisions = {"reject": Decision.REJECT, "abstain": Decision.ABSTAIN, "apply": Decision.APPLY}
    result = evaluate_policy(rows, decisions,
        decision_costs={"reject": .2, "abstain": .4, "apply": .3})
    assert result["task_mean_reward_gains"] == pytest.approx({"a": -.2, "b": -.4, "c": .7})
    assert result["selected_success_rate"] == 1
    assert result["decision_cost_total_reward_units"] == pytest.approx(.9)
    with pytest.raises(ValueError):
        evaluate_policy(rows, decisions, decision_costs={"apply": .3})
    with pytest.raises(ValueError):
        evaluate_policy(rows, decisions, decision_costs={key: float("nan") for key in decisions})


def test_unknown_cost_is_not_silently_treated_as_zero():
    rows = [row("apply", "a", 0, 1), row("abstain", "b", 1, 1), row("reject", "c", 1, 0)]
    decisions = {"apply": Decision.APPLY, "abstain": Decision.ABSTAIN, "reject": Decision.REJECT}
    result = evaluate_policy(rows, decisions,
        decision_costs={"apply": .2, "abstain": None, "reject": 0.0})
    assert result["decision_costs_supplied"] is True
    assert result["decision_costs_complete"] is False
    assert result["known_decision_cost_contexts"] == 2
    assert result["unknown_decision_cost_contexts"] == 1
    assert result["decision_cost_total_reward_units"] is None
    assert result["known_decision_cost_total_reward_units"] == pytest.approx(.2)
    assert result["known_reward_gain_contexts_before_decision_cost"] == 3
    assert result["known_reward_gain_contexts"] == 2
    assert result["reward_gain_contexts_unknown_due_to_decision_cost"] == 1
    assert result["task_mean_reward_gains_before_decision_cost"] == {"a": 1, "b": 0, "c": 0}
    assert result["task_mean_reward_gains"] == pytest.approx({"a": .8, "c": 0})
    assert result["context_mean_reward_gain"] == pytest.approx(.4)


def test_explicit_zero_cost_is_known_and_invalid_costs_are_rejected():
    rows = [row("one", "t", 1, 1)]
    result = evaluate_policy(rows, {"one": Decision.ABSTAIN}, decision_costs={"one": 0.0})
    assert result["decision_costs_complete"] is True
    assert result["decision_cost_total_reward_units"] == 0
    assert result["context_mean_reward_gain"] == 0
    for value in (True, -1, "unknown", float("inf")):
        with pytest.raises(ValueError):
            evaluate_policy(rows, {"one": Decision.ABSTAIN}, decision_costs={"one": value})
