#!/usr/bin/env python3
"""Validate Golden Set JSONL files against schema, ID uniqueness, and Luhn rules."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from payguard.eval.schema import GoldenQuestion


def validate_file(file_path: Path) -> tuple[bool, list[GoldenQuestion], list[str]]:
    """Validate all JSON lines in file_path. Returns (success, items, errors)."""
    if not file_path.exists():
        return False, [], [f"File not found: {file_path}"]

    items: list[GoldenQuestion] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_num}: Invalid JSON - {exc}")
                continue

            try:
                item = GoldenQuestion.model_validate(data)
            except ValidationError as exc:
                errors.append(f"Line {line_num} (id={data.get('id')}): {exc}")
                continue

            if item.id in seen_ids:
                errors.append(f"Line {line_num}: Duplicate id {item.id!r}")
            else:
                seen_ids.add(item.id)

            items.append(item)

    success = len(errors) == 0
    return success, items, errors


def print_summary(file_path: Path, items: list[GoldenQuestion], errors: list[str]) -> None:
    """Print validation summary table and details."""
    print(f"=== VALIDATION REPORT: {file_path} ===")
    if errors:
        print(f"Status: FAILED ({len(errors)} error(s) found)\n")
        for err in errors:
            print(f"  [ERROR] {err}")
        print()
    else:
        print("Status: PASSED (100% valid)\n")

    print(f"Total entries: {len(items)}")
    verified_count = sum(1 for item in items if item.verified)
    unverified_count = len(items) - verified_count
    needs_gt_count = sum(1 for item in items if item.needs_ground_truth)

    print(f"  Verified      : {verified_count}")
    print(f"  Unverified    : {unverified_count}")
    print(f"  Needs GT      : {needs_gt_count}\n")

    type_counts = Counter(item.type for item in items)
    print("Counts by Question Type:")
    for q_type, count in sorted(type_counts.items()):
        print(f"  - {q_type:<15}: {count}")

    print("\nCounts by Source Document:")
    doc_counts = Counter(item.source_doc for item in items)
    for doc_id, count in sorted(doc_counts.items()):
        print(f"  - {doc_id:<32}: {count}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Golden Set dataset.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("eval/golden_set.jsonl"),
        help="Path to JSONL file to validate (default: eval/golden_set.jsonl)",
    )
    args = parser.parse_args()

    success, items, errors = validate_file(args.file)
    print_summary(args.file, items, errors)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

