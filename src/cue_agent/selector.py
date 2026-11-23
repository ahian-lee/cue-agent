"""CUE-v0 conservative selector prototype.

This is deliberately a small, inspectable model. It refuses to fit when the
development table is too small and does not issue a decision-eligible lower
bound without separately validated conditional evidence. A constant target is
still fit as an observed value. These are implementation properties, not
evidence that CUE works.

The fitted target is the *signed, numeric* incremental utility
``delta = R(experience) - R(base)``.  A positive-benefit probability is not
accepted as ``delta_hat``.  ``fit_label_ablation`` exposes the same features,
same split and same regressor family for the ``Y1`` and ``delta`` targets so a
later study can test whether the labels are empirically distinguishable.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge

from .metrics import Decision, interval_decision

FEATURE_NAMES = (
    "experience_covers_all_actions",
    "experience_kind_positive",
    "visible_order_count",
    "requested_action_count",
    "has_cheapest_request",
    "address_withheld",
    "same_user_as_source",
)
MODEL_FEATURE_NAMES = FEATURE_NAMES + tuple(f"{name}__missing" for name in FEATURE_NAMES)

REGRESSOR_ALPHA = 1.0
MEAN_INTERVAL_KIND = "student_t_confidence_interval_for_mean_delta_marginal_approximate"
PREDICTION_INTERVAL_KIND = "split_conformal_prediction_interval_for_a_new_paired_delta"
DIAGNOSTIC_INTERVAL_KIND = "diagnostic_marginal_residual_interval_not_a_conditional_mean_bound"


def _finite_number(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or None") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite or None")
    return number


def _row_id(row: dict[str, Any]) -> str | None:
    for key in ("pair_id", "row_id", "task_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _feature_vector(features: Any) -> list[float]:
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise ValueError("features must be a mapping")
    values = [_finite_number(features.get(name), field_name=f"feature {name}") for name in FEATURE_NAMES]
    # An unknown value must remain distinguishable from an observed zero.
    # The model can only learn a missingness effect if such cases occur in its
    # training data; this encoding alone does not validate an estimate.
    return [value if value is not None else 0.0 for value in values] + [
        1.0 if value is None else 0.0 for value in values]


def _target_value(row: dict[str, Any], target_key: str) -> float | None:
    return _finite_number(row.get(target_key), field_name=f"target {target_key}")


def _target_label(target_key: str) -> str:
    if target_key == "delta":
        return "signed incremental reward delta = experience reward - base reward"
    return str(target_key)


def _split_conformal_radius(residuals: Sequence[float], alpha: float) -> tuple[float | None, int]:
    """Finite-sample absolute-residual rank; None means an unbounded interval."""
    if not 0 < alpha < 1 or len(residuals) == 0:
        raise ValueError("conformal radius needs residuals and alpha in (0, 1)")
    scores = sorted(abs(float(value)) for value in residuals)
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("conformal residuals must be finite")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return (scores[rank - 1] if rank <= len(scores) else None), rank


@dataclass
class CUEv0:
    status: str = "unfitted"
    reason: str = "not fitted"
    target_key: str = "delta"
    n_rows: int = 0
    n_valid_rows: int = 0
    n_unknown_target: int = 0
    positive_rows: int = 0
    model: Any = None
    feature_names: tuple[str, ...] = MODEL_FEATURE_NAMES
    calibration_source: str | None = None
    calibration: dict[str, Any] | None = None
    fit_summary: dict[str, Any] = field(default_factory=dict)
    training_ids: tuple[str, ...] = ()
    last_error: str | None = None

    def _clear_fit_state(self) -> None:
        """A failed refit must not leave an old model or old estimate live."""
        self.model = None
        self.calibration_source = None
        self.calibration = None
        self.fit_summary = {}
        self.training_ids = ()
        self.last_error = None

    def _reset_all(self) -> None:
        self.status, self.reason = "unfitted", "not fitted"
        self.target_key = "delta"
        self.n_rows = self.n_valid_rows = self.n_unknown_target = self.positive_rows = 0
        self._clear_fit_state()

    def fit(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        target_key: str = "delta",
        min_rows: int = 6,
    ) -> "CUEv0":
        """Fit a signed-target ridge regression; clear any previous fit first."""
        self._reset_all()
        self.target_key = target_key
        self.n_rows = len(rows)
        try:
            valid: list[tuple[list[float], float, str]] = []
            for index, row in enumerate(rows):
                target = _target_value(row, target_key)
                if target is None:
                    self.n_unknown_target += 1
                    continue
                valid.append(
                    (
                        _feature_vector(row.get("features")),
                        target,
                        _row_id(row) or f"__fit_{index}",
                    )
                )
            self.n_valid_rows = len(valid)
            self.positive_rows = sum(1 for _, target, _ in valid if target > 0)
            if len(valid) < min_rows:
                self.status = "insufficient"
                self.reason = (
                    f"need at least {min_rows} paired rows with known {target_key}; "
                    f"got {len(valid)} known ({self.n_unknown_target} unknown)"
                )
                return self
            values = np.asarray([target for _, target, _ in valid], dtype=float)
            # A constant target is an observed value, not missing data.  A
            # regression can estimate it; lack of heterogeneity does not make
            # the fit invalid (it only limits any treatment-effect claim).
            X = np.asarray([features for features, _, _ in valid], dtype=float)
            model = Ridge(alpha=REGRESSOR_ALPHA).fit(X, values)
            fitted = model.predict(X)
            residuals = values - fitted
            self.model = model
            self.training_ids = tuple(row_id for _, _, row_id in valid)
            n_features = len(FEATURE_NAMES)
            missing_counts = {
                name: sum(int(features[n_features + i]) for features, _, _ in valid)
                for i, name in enumerate(FEATURE_NAMES)
            }
            self.fit_summary = {
                "target": target_key,
                "target_label": _target_label(target_key),
                "regressor": "ridge",
                "regressor_alpha": REGRESSOR_ALPHA,
                "input_feature_names": list(FEATURE_NAMES),
                "missingness_indicators": True,
                "feature_missing_counts": missing_counts,
                "feature_observed_counts": {name: len(valid) - count for name, count in missing_counts.items()},
                "n_rows": self.n_rows,
                "n_valid_rows": self.n_valid_rows,
                "n_unknown_target": self.n_unknown_target,
                "positive_rows": self.positive_rows,
                "coefficients": dict(zip(self.feature_names, model.coef_.tolist())),
                "intercept": float(model.intercept_),
                "train_mean_target": float(values.mean()),
                "train_residual_sd": float(np.sqrt(np.mean(residuals**2))),
                "fit_row_ids_verified": all(
                    not row_id.startswith("__fit_") for row_id in self.training_ids
                ),
            }
            self.status = "fit_uncalibrated"
            self.reason = (
                "development fit only; no independent calibration split with a "
                "coverage check, so no interval is issued"
            )
            return self
        except Exception as exc:  # malformed input must not leave a live old fit
            self._clear_fit_state()
            self.status = "fit_failed"
            self.reason = f"fit failed: {type(exc).__name__}: {exc}"
            self.last_error = str(exc)
            return self

    def predict_interval(self, features: dict[str, Any]) -> dict[str, Any]:
        """Return the signed point estimate plus clearly diagnostic intervals.

        The marginal residual interval and the individual-outcome interval are
        diagnostics.  They become decision inputs only if a separate
        query-conditional mean bound has been validated; this prototype has no
        such validation.
        """
        out: dict[str, Any] = {
            "target": self.target_key,
            "point_estimate": None,
            "delta_hat": None,
            "lower": None,
            "upper": None,
            "mean_lower": None,
            "mean_upper": None,
            "prediction_lower": None,
            "prediction_upper": None,
            "decision_eligible": False,
            "calibrated": False,
            "interval_kind": None,
            "prediction_interval_kind": None,
            "diagnostic_interval_kind": None,
            "diagnostic_marginal_mean_lower": None,
            "diagnostic_marginal_mean_upper": None,
            "diagnostic_prediction_lower": None,
            "diagnostic_prediction_upper": None,
            "calibration_source": None,
            "coverage_estimate": None,
            "coverage_target": None,
            "n_calibration": None,
            "feature_missingness_support": "unknown_without_fitted_model",
            "unseen_feature_missingness_states": None,
            "reason": self.reason,
        }
        if self.model is None:
            return out
        try:
            vector = _feature_vector(features)
            x = np.asarray([vector], dtype=float)
            prediction = float(self.model.predict(x)[0])
        except Exception as exc:
            out["reason"] = f"prediction failed: {type(exc).__name__}: {exc}"
            return out
        out["point_estimate"] = prediction
        missing_counts = self.fit_summary.get("feature_missing_counts") or {}
        observed_counts = self.fit_summary.get("feature_observed_counts") or {}
        if all(name in missing_counts and name in observed_counts for name in FEATURE_NAMES):
            n_features = len(FEATURE_NAMES)
            unseen = {name: ("missing_not_seen_in_training" if vector[n_features + i] else
                             "observed_not_seen_in_training")
                      for i, name in enumerate(FEATURE_NAMES)
                      if (missing_counts[name] == 0 if vector[n_features + i] else observed_counts[name] == 0)}
            out["unseen_feature_missingness_states"] = unseen
            out["feature_missingness_support"] = ("unseen_training_state" if unseen else "individual_states_seen")
        if self.target_key == "delta":
            out["delta_hat"] = prediction
        if self.calibration is None:
            return out

        diagnostic_mean_lower = prediction + float(self.calibration["mean_lower_residual"])
        diagnostic_mean_upper = prediction + float(self.calibration["mean_upper_residual"])
        diagnostic_prediction_lower = prediction + float(self.calibration["residual_lower"])
        diagnostic_prediction_upper = prediction + float(self.calibration["residual_upper"])
        out.update(
            {
                "interval_kind": self.calibration["interval_kind"],
                "prediction_interval_kind": self.calibration["prediction_interval_kind"],
                "diagnostic_interval_kind": self.calibration["interval_kind"],
                "diagnostic_marginal_mean_lower": diagnostic_mean_lower,
                "diagnostic_marginal_mean_upper": diagnostic_mean_upper,
                "diagnostic_prediction_lower": diagnostic_prediction_lower,
                "diagnostic_prediction_upper": diagnostic_prediction_upper,
                "calibration_source": self.calibration_source,
                "coverage_estimate": self.calibration["coverage_estimate"],
                "coverage_target": self.calibration["coverage_target"],
                "n_calibration": self.calibration["n_calibration"],
                "reason": self.reason,
            }
        )
        # Only a separately validated query-conditional mean bound may unlock
        # a decision.  The global residual-mean interval is kept as a diagnostic
        # and is never promoted to that role.
        if self.calibration.get("conditional_mean_bound_valid") is True:
            mean_lower = diagnostic_mean_lower
            mean_upper = diagnostic_mean_upper
            out.update(
                {
                    "lower": mean_lower,
                    "upper": mean_upper,
                    "mean_lower": mean_lower,
                    "mean_upper": mean_upper,
                    "prediction_lower": diagnostic_prediction_lower,
                    "prediction_upper": diagnostic_prediction_upper,
                    "decision_eligible": True,
                    "calibrated": True,
                    "reason": "validated query-conditional mean bound",
                }
            )
        return out

    def decide(self, features: dict[str, Any]) -> dict[str, Any]:
        """Apply only for a validated query-conditional mean lower bound > 0."""
        interval = self.predict_interval(features)
        if self.target_key != "delta":
            decision, reason = Decision.ABSTAIN.value, "CUE decisions require the signed delta target"
        elif not interval["decision_eligible"]:
            decision, reason = (
                Decision.ABSTAIN.value,
                "no validated query-conditional mean bound; diagnostic marginal "
                "intervals are not decision inputs; continue base policy",
            )
        elif interval["mean_lower"] is None or interval["mean_upper"] is None:
            decision, reason = Decision.ABSTAIN.value, "conditional mean-delta interval is unavailable"
        else:
            decision = interval_decision(interval["mean_lower"], interval["mean_upper"]).value
            if decision == Decision.APPLY.value:
                reason = "validated conditional mean lower bound is strictly above zero"
            elif decision == Decision.REJECT.value:
                reason = "validated conditional mean upper bound is strictly below zero"
            else:
                reason = "conditional mean interval includes zero"
        return {
            "decision": decision,
            "reason": reason,
            "interval": interval,
            "status": self.status,
        }

    def calibrate(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        alpha: float = 0.1,
        min_rows: int = 20,
        min_calibration: int = 10,
        min_coverage: int = 5,
        coverage_fraction: float = 0.5,
        coverage_tolerance: float = 0.05,
        seed: int = 0,
        source: str | None = None,
    ) -> "CUEv0":
        """Compute diagnostic marginal intervals; never a decision bound.

        The global residual-mean interval and the individual-outcome interval
        are retained as clearly labeled diagnostics.  They do not establish a
        query-conditional mean bound, so ``decide`` still abstains for every
        query.  A future conditional verification would have to set
        ``conditional_mean_bound_valid`` explicitly and provide its own
        evidence; this method does not do that.
        """
        if self.model is None:
            raise ValueError("fit a model before calibrating")
        if self.target_key != "delta":
            raise ValueError("calibration is defined for the signed delta target")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1")
        if not (0 < coverage_fraction < 1):
            raise ValueError("coverage_fraction must be strictly between 0 and 1")
        if not self.fit_summary.get("fit_row_ids_verified"):
            raise ValueError(
                "fit rows need pair_id/row_id values so calibration rows can be checked as disjoint"
            )
        parsed: list[tuple[list[float], float, str]] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            target = _target_value(row, "delta")
            if target is None:
                raise ValueError(f"calibration row {index} has no known delta")
            row_id = _row_id(row)
            if row_id is None:
                raise ValueError(f"calibration row {index} needs pair_id/row_id")
            if row_id in seen:
                raise ValueError(f"duplicate calibration row id: {row_id}")
            seen.add(row_id)
            parsed.append((_feature_vector(row.get("features")), target, row_id))
        if len(parsed) < min_rows:
            raise ValueError(f"need at least {min_rows} calibration rows; got {len(parsed)}")
        overlap = sorted(seen.intersection(self.training_ids))
        if overlap:
            raise ValueError(f"calibration rows overlap fitting rows: {overlap[:5]}")
        if len(parsed) - max(min_calibration, int(round(len(parsed) * coverage_fraction))) < min_coverage:
            raise ValueError("not enough rows to split calibration and coverage checks")

        rng = random.Random(seed)
        order = list(range(len(parsed)))
        rng.shuffle(order)
        n_calibration = max(min_calibration, int(round(len(parsed) * (1 - coverage_fraction))))
        n_calibration = min(n_calibration, len(parsed) - min_coverage)
        calibration_rows = [parsed[index] for index in order[:n_calibration]]
        coverage_rows = [parsed[index] for index in order[n_calibration:]]

        X_cal = np.asarray([features for features, _, _ in calibration_rows], dtype=float)
        y_cal = np.asarray([target for _, target, _ in calibration_rows], dtype=float)
        residuals = y_cal - self.model.predict(X_cal)
        # The finite-sample conformal rank applies to absolute residuals, not
        # ordinary empirical two-sided quantiles. If the rank exceeds n, the
        # formal interval is unbounded; do not serialize infinity as a bound.
        radius, conformal_rank = _split_conformal_radius(residuals, alpha)
        if radius is None:
            self.calibration = None
            self.calibration_source = None
            self.status = "fit_uncalibrated"
            self.reason = (f"no finite conformal prediction interval: rank {conformal_rank} "
                           f"exceeds {len(residuals)} calibration residuals")
            return self
        residual_lower, residual_upper = -radius, radius
        # A marginal Student-t interval for the global residual mean.  It is
        # deliberately kept diagnostic: it is not query-conditional and cannot
        # certify a specific history's mean delta.
        mean_residual = float(residuals.mean())
        mean_sd = float(residuals.std(ddof=1))
        mean_se = mean_sd / math.sqrt(len(residuals))
        t_critical = float(student_t.ppf(1 - alpha / 2, df=len(residuals) - 1))
        mean_lower_residual = mean_residual - t_critical * mean_se
        mean_upper_residual = mean_residual + t_critical * mean_se

        X_cov = np.asarray([features for features, _, _ in coverage_rows], dtype=float)
        y_cov = np.asarray([target for _, target, _ in coverage_rows], dtype=float)
        predicted_cov = self.model.predict(X_cov)
        covered = (y_cov >= predicted_cov + residual_lower) & (
            y_cov <= predicted_cov + residual_upper
        )
        coverage_estimate = float(covered.mean()) if len(covered) else None
        coverage_target = 1 - alpha
        coverage_ok = bool(
            coverage_estimate is not None
            and len(coverage_rows) >= min_coverage
            and coverage_estimate >= coverage_target - coverage_tolerance
        )
        source_text = source
        if source_text is None:
            labels = sorted({str(row.get("source")) for row in rows if row.get("source")})
            source_text = (
                "caller-supplied calibration rows, group independence unverified"
                if not labels
                else "caller-supplied calibration rows from " + ", ".join(labels)
            )
        self.calibration_source = source_text
        self.calibration = {
            "interval_kind": DIAGNOSTIC_INTERVAL_KIND,
            "diagnostic_mean_interval_kind": MEAN_INTERVAL_KIND,
            "prediction_interval_kind": PREDICTION_INTERVAL_KIND,
            "method": (
                "diagnostic Student-t global residual-mean interval plus a "
                "split-conformal individual-outcome coverage check"
            ),
            "target": "diagnostic marginal residual behaviour, not a query-conditional delta mean",
            "decision_eligible": False,
            "group_disjointness_verified": False,
            "exchangeability_verified": False,
            "conformal_rank": conformal_rank,
            "conformal_radius": radius,
            "conditional_mean_bound_valid": False,
            "conditional_mean_bound_status": "not_established",
            "conditional_mean_bound_note": (
                "a global residual mean/SE is not a query-conditional mean bound; "
                "no conditional coverage or stratum validation was performed"
            ),
            "alpha": alpha,
            "coverage_target": coverage_target,
            "coverage_tolerance": coverage_tolerance,
            "coverage_estimate": coverage_estimate,
            "coverage_ok": coverage_ok,
            "coverage_scope": "individual prediction interval, not a decision bound",
            "n_rows": len(parsed),
            "n_calibration": len(calibration_rows),
            "n_coverage": len(coverage_rows),
            "residual_lower": residual_lower,
            "residual_upper": residual_upper,
            "mean_residual": mean_residual,
            "mean_sd": mean_sd,
            "mean_se": mean_se,
            "t_critical": t_critical,
            "mean_lower_residual": mean_lower_residual,
            "mean_upper_residual": mean_upper_residual,
            "independent_ids_verified": True,
            "source": source_text,
            "assumptions": [
                "calibration and coverage rows are exchangeable for the diagnostic residual distribution",
                "the frozen feature map is used for both diagnostics",
                "features are pre-intervention and the delta target is not read after the decision",
            ],
            "limitations": [
                "the global residual-mean interval is marginal, not query-conditional, and is not used by decide",
                "the empirical coverage check is for the individual prediction interval, not for a conditional mean bound",
                "neither diagnostic authorizes Apply; a query-conditional mean bound is still future work",
                "rows are not clustered by task here; task-cluster uncertainty still needs a separate analysis",
                "no actual independent calibration data were used in the stage-0 pilot",
            ],
        }
        self.status = "fit_uncalibrated"
        self.reason = (
            "diagnostic marginal intervals computed; no validated query-conditional "
            "mean bound, so decisions must abstain "
            f"(coverage={coverage_estimate}, target={coverage_target})"
        )
        return self


def _ablation_split(
    rows: Sequence[dict[str, Any]], *, test_fraction: float, seed: int,
    min_train: int, min_test: int, group_by: Sequence[str] | None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Keep rows connected by any supplied group in one partition."""
    n = len(rows)
    if group_by is None:
        order = list(range(n))
        random.Random(seed).shuffle(order)
        n_test = min(max(min_test, int(round(n * test_fraction))), n - min_train)
        return order[n_test:], order[:n_test], {
            "strategy": "random_rows_development_only", "group_fields": [],
            "group_disjointness_verified": False,
        }
    if isinstance(group_by, str):
        raise ValueError("group_by must be a sequence of field names, not a string")
    fields = tuple(group_by)
    if not fields or any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("group_by must contain nonempty field names")
    if len(set(fields)) != len(fields):
        raise ValueError("group_by field names must be unique")
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for field_name in fields:
            value = row.get(field_name)
            if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
                raise ValueError(f"row {_row_id(row) or i} has unknown group {field_name}")
            key = (field_name, str(value))
            if key in first:
                parent[root(i)] = root(first[key])
            else:
                first[key] = i
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    # Dynamic programming finds a feasible group partition even when a greedy
    # choice of large components would leave too few rows on one side.
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(groups):
        for total, chosen in list(choices.items()):
            new_total = total + len(group)
            if new_total not in choices:
                choices[new_total] = chosen + (index,)
    feasible = [total for total in choices if total >= min_test and n - total >= min_train]
    if not feasible:
        raise ValueError("no group-disjoint split satisfies the requested row minimums")
    n_test = min(feasible, key=lambda total: (abs(total - n * test_fraction), total))
    selected = set(choices[n_test])
    test = [i for j, group in enumerate(groups) if j in selected for i in group]
    train = [i for j, group in enumerate(groups) if j not in selected for i in group]
    return train, test, {"strategy": "connected_group_components", "group_fields": list(fields),
                        "group_disjointness_verified": True, "n_components": len(groups)}


def fit_label_ablation(
    rows: Sequence[dict[str, Any]],
    *,
    test_fraction: float = 0.4,
    seed: int = 0,
    min_train: int = 4,
    min_test: int = 2,
    group_by: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare ``Y1`` and ``delta`` targets under one frozen split and model.

    ``Y1`` is the experience-branch reward (or explicit ``experience_reward``);
    ``delta`` is experience minus base.  This is an implementation ablation
    interface, not evidence that either label is empirically better. Specify
    group_by to keep all rows sharing any chosen task/source/template group in
    one partition; absent grouping, the row split is development-only.
    """
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")
    prepared: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        row_id = _row_id(row)
        if row_id is None:
            raise ValueError(f"ablation row {index} needs pair_id/row_id")
        if row_id in ids:
            raise ValueError(f"duplicate ablation row id: {row_id}")
        ids.add(row_id)
        delta = _target_value(row, "delta")
        if delta is None:
            raise ValueError(f"ablation row {row_id} has no known delta")
        if "y1" in row:
            y1 = _target_value(row, "y1")
        else:
            y1 = _target_value(row, "experience_reward")
        if y1 is None:
            raise ValueError(f"ablation row {row_id} has no known Y1/experience_reward")
        prepared.append({"pair_id": row_id, "features": row.get("features"), "delta": delta, "y1": y1})
    n = len(prepared)
    if n < min_train + min_test:
        raise ValueError(f"need at least {min_train + min_test} rows; got {n}")

    train_indices, test_indices, split = _ablation_split(
        rows, test_fraction=test_fraction, seed=seed, min_train=min_train,
        min_test=min_test, group_by=group_by,
    )
    train_rows = [prepared[index] for index in train_indices]
    test_rows = [prepared[index] for index in test_indices]

    def metrics(model: CUEv0, target_key: str) -> dict[str, Any]:
        pairs: list[tuple[float, float]] = []
        for row in test_rows:
            prediction = model.predict_interval(row["features"])["point_estimate"]
            if prediction is not None:
                pairs.append((float(prediction), float(row[target_key])))
        if not pairs:
            return {
                "n_evaluated": 0,
                "mae": None,
                "rmse": None,
                "sign_accuracy": None,
                "mean_prediction": None,
                "mean_actual": None,
            }
        errors = [prediction - actual for prediction, actual in pairs]
        return {
            "n_evaluated": len(pairs),
            "mae": float(sum(abs(error) for error in errors) / len(errors)),
            "rmse": float(math.sqrt(sum(error * error for error in errors) / len(errors))),
            "sign_accuracy": float(
                sum(
                    (prediction > 0) == (actual > 0)
                    for prediction, actual in pairs
                )
                / len(pairs)
            ),
            "mean_prediction": float(sum(prediction for prediction, _ in pairs) / len(pairs)),
            "mean_actual": float(sum(actual for _, actual in pairs) / len(pairs)),
        }

    targets: dict[str, Any] = {}
    for target_key in ("delta", "y1"):
        model = CUEv0().fit(train_rows, target_key=target_key, min_rows=min_train)
        targets[target_key] = {
            "fit_status": model.status,
            "fit_reason": model.reason,
            "fit_summary": model.fit_summary,
            "metrics": metrics(model, target_key),
        }
    return {
        "same_features": True,
        "same_split": True,
        "same_model_family": "ridge",
        "same_hyperparameters": {"alpha": REGRESSOR_ALPHA},
        "n_rows": n,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_fraction": test_fraction,
        "split_seed": seed,
        "split": {**split, "train_pair_ids": [row["pair_id"] for row in train_rows],
                  "test_pair_ids": [row["pair_id"] for row in test_rows]},
        "targets": targets,
        "scope": (
            "Implementation-only label ablation; tests whether the interface keeps the "
            "features/split/model fixed while changing Y1 to delta. It is not evidence "
            "that one label is superior on the research task."
        ),
    }


def project_features(
    *,
    requested_actions: Sequence[str] | None,
    experience_covered_actions: Sequence[str] | None,
    experience_kind: str | None,
    visible_order_count: int | None,
    has_cheapest_request: bool | None = None,
    address_withheld: bool | None = None,
    same_user_as_source: bool | None = None,
) -> dict[str, float | None]:
    """Project explicitly supplied observations; omitted optional facts stay unknown."""
    requested = {str(action) for action in requested_actions} if requested_actions is not None else None
    covered = {str(action) for action in experience_covered_actions} if experience_covered_actions is not None else None
    def optional_bool(value):
        return None if value is None else float(bool(value))
    return {
        "experience_covers_all_actions": (float(bool(requested) and requested <= covered)
            if requested is not None and covered is not None else None),
        "experience_kind_positive": (float(experience_kind == "positive_mechanism")
            if experience_kind is not None else None),
        "visible_order_count": float(visible_order_count) if visible_order_count is not None else None,
        "requested_action_count": float(len(requested)) if requested is not None else None,
        "has_cheapest_request": optional_bool(has_cheapest_request),
        "address_withheld": optional_bool(address_withheld),
        "same_user_as_source": optional_bool(same_user_as_source),
    }


def bootstrap_placeholder(rows: Sequence[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Descriptive mean only; explicitly not a calibrated CUE bound."""
    values = [float(row["delta"]) for row in rows if row.get("delta") is not None]
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "reason": "no known paired deltas"}
    return {
        "estimate": sum(values) / len(values),
        "lower": None,
        "upper": None,
        "reason": "descriptive mean only; no independent calibration / cluster interval",
        "n": len(values),
        "seed": seed,
    }
