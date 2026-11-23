"""Evaluation arithmetic for CUE, separate from pre-intervention features.

These are outcome records, not model inputs or a full checkpoint format.
Rewards may include a predeclared cost penalty; this module never invents
weights or claims that a supplied uncertainty interval is calibrated.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class Decision(str, Enum):
    APPLY = "apply"
    REJECT = "reject"
    ABSTAIN = "abstain"


def interval_decision(lower: float, upper: float) -> Decision:
    """Mechanics only: confidence estimation must be validated separately."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("Finite ordered bounds are required.")
    if lower > 0:
        return Decision.APPLY
    if upper < 0:
        return Decision.REJECT
    return Decision.ABSTAIN


@dataclass(frozen=True)
class PairedOutcome:
    pair_id: str
    task_id: str
    experience_id: str
    prefix_id: str
    seed: int
    reward_base: float | None
    reward_experience: float | None
    success_base: bool | None
    success_experience: bool | None
    # Optional separately supplied user-utility observations. A caller must
    # still establish their provenance and comparability; the arithmetic here
    # cannot verify them or manufacture them from a benchmark score.
    user_utility_base: float | None = None
    user_utility_experience: float | None = None

    def __post_init__(self):
        for value in (self.reward_base, self.reward_experience,
                      self.user_utility_base, self.user_utility_experience):
            if value is not None and not math.isfinite(value):
                raise ValueError("Unknown rewards must be None, not NaN or infinity.")
        for value in (self.success_base, self.success_experience):
            if value is not None and not isinstance(value, bool):
                raise TypeError("Task success is bool or None; it is not inferred from reward.")

    @property
    def delta(self) -> float | None:
        if self.reward_base is None or self.reward_experience is None:
            return None
        return self.reward_experience - self.reward_base

    @property
    def supplied_user_utility_delta(self) -> float | None:
        if self.user_utility_base is None or self.user_utility_experience is None:
            return None
        return self.user_utility_experience - self.user_utility_base


def _mean(values):
    return sum(values) / len(values) if values else None


def evaluate_policy(
    rows: Sequence[PairedOutcome],
    decisions: Mapping[str, Decision],
    *,
    decision_costs: Mapping[str, float | None] | None = None,
) -> dict:
    ids = [row.pair_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("A pair ID must identify one task/prefix/experience/seed observation.")
    if set(ids) != set(decisions):
        raise ValueError("Exactly one decision per observed pair is required.")
    # Branch rewards already include declared branch costs. These are the
    # *additional* selector/verification costs, in the same reward units. A
    # reject/abstain policy still pays them even though it continues the base.
    if decision_costs is not None and set(ids) != set(decision_costs):
        raise ValueError("Provide a decision cost for every pair, including reject/abstain.")
    # Omitted or individually unknown costs are not zero. Callers can supply
    # an explicit 0.0 when a no-cost decision is actually established.
    costs = {key: None for key in ids} if decision_costs is None else dict(decision_costs)
    for value in costs.values():
        if value is None:
            continue
        try:
            valid = not isinstance(value, bool) and math.isfinite(value) and value >= 0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError("Decision costs must be finite and nonnegative, or None if unknown.")
    actions = {key: Decision(value) for key, value in decisions.items()}
    gross_effects, net_effects = defaultdict(list), defaultdict(list)
    selected_success, base_success = [], []
    success_pairs = {"base0_policy0": 0, "base0_policy1": 0, "base1_policy0": 0, "base1_policy1": 0}
    known_applied_deltas = []
    beneficial = beneficial_applied = task_regressions = 0
    known_pairs = unknown_selected_success = 0
    supplied_user_deltas, applied_user_deltas = [], []
    supplied_user_increases_applied = 0
    for row in rows:
        apply = actions[row.pair_id] == Decision.APPLY
        # Reject and abstain continue the base policy. They do not halt work or
        # receive automatic success. Selector costs are deducted separately.
        selected = row.success_experience if apply else row.success_base
        if selected is None:
            unknown_selected_success += 1
        else:
            selected_success.append(int(selected))
        if row.success_base is not None:
            base_success.append(int(row.success_base))
            if selected is not None:
                success_pairs[f"base{int(row.success_base)}_policy{int(selected)}"] += 1
        if apply and row.success_base is True and row.success_experience is False:
            task_regressions += 1
        delta = row.delta
        if delta is not None:
            known_pairs += 1
            if delta > 0:
                beneficial += 1
                beneficial_applied += int(apply)
            if apply:
                known_applied_deltas.append(delta)
        user_delta = row.supplied_user_utility_delta
        if user_delta is not None:
            supplied_user_deltas.append(user_delta)
            if apply:
                applied_user_deltas.append(user_delta)
                supplied_user_increases_applied += int(user_delta > 0)
        # A nonintervention has zero incremental reward when R0 is known even
        # if the unchosen experience branch has an unknown reward.
        effect = delta if apply else (0.0 if row.reward_base is not None else None)
        if effect is not None:
            gross_effects[row.task_id].append(effect)
            cost = costs[row.pair_id]
            if cost is not None:
                net_effects[row.task_id].append(effect - cost)
    gross_task_means = {task: _mean(values) for task, values in gross_effects.items()}
    net_task_means = {task: _mean(values) for task, values in net_effects.items()}
    known_costs = [value for value in costs.values() if value is not None]
    n_gross = sum(len(values) for values in gross_effects.values())
    n_net = sum(len(values) for values in net_effects.values())
    supplied_user_increases = sum(delta > 0 for delta in supplied_user_deltas)
    return {
        "contexts": len(rows), "task_clusters": len({row.task_id for row in rows}),
        "decisions": {action.value: sum(value == action for value in actions.values()) for action in Decision},
        "decision_costs_supplied": decision_costs is not None,
        "known_decision_cost_contexts": len(known_costs),
        "unknown_decision_cost_contexts": len(ids) - len(known_costs),
        "decision_costs_complete": decision_costs is not None and len(known_costs) == len(ids),
        "decision_cost_total_reward_units": (sum(known_costs)
            if decision_costs is not None and len(known_costs) == len(ids) else None),
        "known_decision_cost_total_reward_units": sum(known_costs) if known_costs else None,
        "apply_coverage": sum(value == Decision.APPLY for value in actions.values()) / len(rows) if rows else None,
        "known_paired_rewards": known_pairs,
        "unknown_paired_rewards": len(rows)-known_pairs,
        "selected_success_count": sum(selected_success),
        "selected_success_denominator": len(selected_success),
        "selected_success_rate": _mean(selected_success),
        "unknown_selected_success": unknown_selected_success,
        "base_success_rate": _mean(base_success),
        "base_success_count": sum(base_success),
        "base_success_denominator": len(base_success),
        "paired_context_success_table": success_pairs,
        "paired_context_success_denominator": sum(success_pairs.values()),
        "observed_reward_increase_candidates": beneficial,
        "observed_reward_increase_intervention_recall": beneficial_applied/beneficial if beneficial else None,
        "observed_reward_decrease_after_apply_count": sum(value < 0 for value in known_applied_deltas),
        "observed_reward_decrease_after_apply_rate": _mean([int(value < 0) for value in known_applied_deltas]),
        "known_applied_reward_denominator": len(known_applied_deltas),
        "observed_success_label_decreases_after_apply": task_regressions,
        "supplied_user_utility_paired_observations": len(supplied_user_deltas),
        "unknown_user_utility_pairs": len(rows) - len(supplied_user_deltas),
        "supplied_user_utility_increase_candidates": supplied_user_increases if supplied_user_deltas else None,
        "supplied_user_utility_increase_intervention_recall": (
            supplied_user_increases_applied / supplied_user_increases if supplied_user_increases else None),
        "supplied_user_utility_decrease_after_apply_count": (sum(value < 0 for value in applied_user_deltas)
                                                               if applied_user_deltas else None),
        "supplied_user_utility_decrease_after_apply_rate": _mean([int(value < 0) for value in applied_user_deltas]),
        "supplied_user_utility_applied_denominator": len(applied_user_deltas),
        "supplied_user_utility_provenance_verified_by_this_function": False,
        "supplied_user_utility_metrics_include_selector_cost": False,
        # Supplied labels may be fixed-reference scores; do not promote them
        # to verified user benefit, user harm, or a causal transfer estimate.
        "beneficial_intervention_count": None,
        "beneficial_intervention_recall": None,
        "negative_reward_transfer_count": None,
        "negative_reward_transfer_rate": None,
        "task_regressions_after_apply": None,
        "context_mean_reward_gain_before_decision_cost": _mean(
            [value for values in gross_effects.values() for value in values]),
        "task_macro_reward_gain_before_decision_cost": _mean(list(gross_task_means.values())),
        "known_reward_gain_contexts_before_decision_cost": n_gross,
        "reward_gain_task_denominator_before_decision_cost": len(gross_task_means),
        "task_mean_reward_gains_before_decision_cost": gross_task_means,
        "context_mean_reward_gain": _mean([value for values in net_effects.values() for value in values]),
        "task_macro_reward_gain": _mean(list(net_task_means.values())),
        "known_reward_gain_contexts": n_net,
        "reward_gain_contexts_unknown_due_to_decision_cost": n_gross - n_net,
        "reward_gain_task_denominator": len(net_task_means),
        "task_mean_reward_gains": net_task_means,
        "scope": "Net reward-gain summaries include only contexts with known supplied decision cost; "
            "before-decision-cost summaries are separate. Supplied scores are not verified user benefit/harm, "
            "calibrated confidence, causal identification, or deployed cost savings.",
    }
