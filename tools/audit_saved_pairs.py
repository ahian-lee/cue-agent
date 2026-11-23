"""Summarize existing development artifacts locally. No runs, fits, or writes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def signed(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-audit", required=True, type=Path)
    parser.add_argument("--early-pairs", required=True, type=Path)
    parser.add_argument("--overlap", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    args = parser.parse_args()

    audit = load(args.paired_audit)
    early_rows = load(args.early_pairs)["rows"]
    overlap = load(args.overlap)
    replay = load(args.replay)
    inventory = audit["original_pilot_inventory"]
    comparisons = [
        comparison
        for attempts in audit["development_attempts"].values()
        for attempt in attempts
        for comparison in attempt["comparisons_vs_same_attempt_base"].values()
    ]
    comparable_deltas = [
        row["whole_task_utility_delta"]
        for row in comparisons
        if row["same_annotated_requirements"] and row["whole_task_utility_delta"] is not None
    ]
    provisional = [
        row["raw_official_delta"] for row in early_rows if row["raw_official_delta"] is not None
    ]
    replay_counts = (
        "replayed_arms",
        "replayed_arms_matching_archive",
        "audited_injected_comparisons",
        "injected_comparisons_matching_archive_on_both_sides",
        "injected_comparisons_with_aligned_goals_and_matching_replays",
        "injected_comparisons_with_observed_whole_task_utility_delta",
    )
    print(json.dumps({
        "pilot": {
            "targets": len(inventory),
            "nominal_sources": len({row["source"] for row in inventory}),
            "all_three_scored": sum(set(row["arms"].values()) == {"evaluated"} for row in inventory),
            "preserved_evidence_files": audit["preserved_direct_evidence_files"],
            "comparisons": len(comparisons),
            "comparisons_with_aligned_annotated_requirements": sum(
                row["same_annotated_requirements"] for row in comparisons
            ),
            "comparable_whole_task_effects": signed(comparable_deltas),
            "independent_human_review": audit["independent_human_review"],
        },
        "early_collection": {
            "rows": len(early_rows),
            "labels": dict(Counter(row["label_status"] for row in early_rows)),
            "collector_status": dict(Counter(row["collector_status"] for row in early_rows)),
            "raw_official_delta_including_censored": signed(provisional),
            "declared_group_pairs": len({
                (row["target_group"], row["source_group"]) for row in early_rows
            }),
        },
        "declared_reuse_components": {
            name: overlap[name]["user_and_any_source_action"]["component_sizes"]
            for name in ("started_pairs", "natural_scored_pairs")
        },
        "replay": {key: replay[key] for key in replay_counts},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
