#!/usr/bin/env python3
"""Phase 4 CI Evaluation Gate comparing run metrics against eval/baseline.json."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payguard.eval_gate")

DEFAULT_GATE_METRICS = ["faithfulness", "context_recall", "abstention_accuracy"]


def find_latest_run_dir(base_dir: Path = Path("eval/results/runs")) -> Path:
    """Find the most recently created run directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Runs directory {base_dir} does not exist.")
    runs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base_dir}.")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def extract_run_metrics(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Extract manifest and aggregated scores from a run directory."""
    manifest_file = run_dir / "manifest.json"
    scores_file = run_dir / "scores.jsonl"

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file {scores_file} not found.")

    manifest: dict[str, Any] = {}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    scores_by_cfg: dict[str, list[dict[str, Any]]] = {}
    with open(scores_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                scores_by_cfg.setdefault(item["config"], []).append(item)

    config_metrics: dict[str, dict[str, float]] = {}
    for cfg, items in scores_by_cfg.items():
        faith_vals: list[float] = []
        rel_vals: list[float] = []
        prec_vals: list[float] = []
        rec_vals: list[float] = []
        abst_vals: list[float] = []
        n_scored = 0

        for it in items:
            s = it.get("scores", {})
            ans = it.get("answerable", True)
            if not it.get("is_nan", False):
                n_scored += 1
            if ans:
                if s.get("faithfulness") is not None and not math.isnan(s["faithfulness"]):
                    faith_vals.append(float(s["faithfulness"]))
                if s.get("answer_relevancy") is not None and not math.isnan(s["answer_relevancy"]):
                    rel_vals.append(float(s["answer_relevancy"]))
                if s.get("context_precision") is not None and not math.isnan(s["context_precision"]):
                    prec_vals.append(float(s["context_precision"]))
                if s.get("context_recall") is not None and not math.isnan(s["context_recall"]):
                    rec_vals.append(float(s["context_recall"]))
            else:
                if s.get("correct_abstention") is not None and not math.isnan(s["correct_abstention"]):
                    abst_vals.append(float(s["correct_abstention"]))

        config_metrics[cfg] = {
            "faithfulness": sum(faith_vals) / len(faith_vals) if faith_vals else 0.0,
            "answer_relevancy": sum(rel_vals) / len(rel_vals) if rel_vals else 0.0,
            "context_precision": sum(prec_vals) / len(prec_vals) if prec_vals else 0.0,
            "context_recall": sum(rec_vals) / len(rec_vals) if rec_vals else 0.0,
            "abstention_accuracy": sum(abst_vals) / len(abst_vals) if abst_vals else 0.0,
            "n_scored": float(n_scored),
            "n_total": float(len(items)),
        }

    return manifest, config_metrics


def create_baseline(run_dir: Path, baseline_path: Path) -> None:
    """Snapshot a run directory's metrics into eval/baseline.json."""
    manifest, config_metrics = extract_run_metrics(run_dir)
    if manifest.get("mock", False):
        print(
            f"Error: Refusing to create baseline from mock run '{run_dir.name}' (mock=true)."
        )
        print("Baselines may only be created from verified real runs.")
        sys.exit(1)

    baseline_data = {
        "baseline_run_id": manifest.get("run_id", run_dir.name),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": manifest.get("git_commit", "uncommitted"),
        "golden_set_sha256": manifest.get("golden_set_sha256", ""),
        "judge_provider": manifest.get("judge_provider", ""),
        "judge_model": manifest.get("judge_model", ""),
        "mock": False,
        "configs": config_metrics,
    }

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    print(f"✓ Created evaluation baseline from run '{run_dir.name}' at {baseline_path}")


def evaluate_gate(
    run_dir: Path,
    baseline_path: Path,
    primary_config: str = "hybrid",
    tolerance: float = 0.0,
    check_metrics: list[str] | None = None,
) -> bool:
    """Compare all configs in current run against baseline. Return True if all pass."""
    if not baseline_path.exists():
        print(f"Error: Baseline file '{baseline_path}' does not exist.")
        print("Gate requires an explicit baseline to compare against.")
        print("Run 'make eval-baseline' or 'python scripts/eval_gate.py --create-baseline' first.")
        return False

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("mock", False):
        print("Error: Baseline was created from a mock run. Baselines must be from real runs.")
        return False

    base_configs = baseline.get("configs", {})
    manifest, current_configs = extract_run_metrics(run_dir)

    if manifest.get("mock", False):
        print(
            f"Error: Refusing to evaluate gate on mock run '{run_dir.name}' (mock=true)."
        )
        print("Evaluation gate requires real run results.")
        return False

    metrics_to_check = check_metrics or DEFAULT_GATE_METRICS

    # Evaluate every config present in run that also exists in baseline
    configs_to_evaluate = [c for c in current_configs if c in base_configs]
    if not configs_to_evaluate:
        print(
            f"Error: No matching configs found between run ({list(current_configs.keys())}) "
            f"and baseline ({list(base_configs.keys())})."
        )
        return False

    # Put primary_config first if present
    if primary_config in configs_to_evaluate:
        configs_to_evaluate.remove(primary_config)
        configs_to_evaluate.insert(0, primary_config)

    total_regressions: list[str] = []

    for cfg in configs_to_evaluate:
        base_metrics = base_configs[cfg]
        curr_metrics = current_configs[cfg]
        is_primary = (cfg == primary_config)
        tag = " [PRIMARY CONFIG]" if is_primary else ""

        print("=" * 70)
        print(f"=== EVALUATION GATE: {cfg}{tag} (vs {baseline.get('baseline_run_id')}) ===")
        print("=" * 70)
        print(f"{'Metric':<24} {'Baseline':<12} {'Current':<12} {'Delta':<12} {'Status'}")
        print("-" * 70)

        cfg_regressions: list[str] = []
        for m in metrics_to_check:
            base_val = float(base_metrics.get(m, 0.0))
            curr_val = float(curr_metrics.get(m, 0.0))
            delta = curr_val - base_val

            status = "PASS"
            if delta < -tolerance:
                status = "REGRESSION"
                cfg_regressions.append(
                    f"[{cfg}] {m} dropped by {abs(delta):.3f} (baseline={base_val:.3f}, current={curr_val:.3f})"
                )

            delta_str = f"{delta:+.3f}"
            print(f"{m:<24} {base_val:<12.3f} {curr_val:<12.3f} {delta_str:<12} {status}")

        print("=" * 70)
        if cfg_regressions:
            total_regressions.extend(cfg_regressions)

    if total_regressions:
        print("\n[FAIL] Evaluation gate rejected: regressions detected in 1 or more configurations:")
        for r in total_regressions:
            print(f"  - {r}")
        return False

    print("\n[PASS] Evaluation gate passed: all configurations meet or exceed baseline.")
    return True


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Evaluate CI quality gate against eval/baseline.json.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to run directory. If omitted, uses the most recent run in eval/results/runs.",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("eval/baseline.json"),
        help="Path to baseline file (default: eval/baseline.json).",
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Snapshot the specified run as the official baseline.",
    )
    parser.add_argument(
        "--primary-config",
        type=str,
        default=os.getenv("PRIMARY_CONFIG", "hybrid"),
        help="Primary configuration (default: env PRIMARY_CONFIG or 'hybrid').",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Allowed regression tolerance (default: 0.0).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()

    if args.create_baseline:
        create_baseline(run_dir, args.baseline_file)
        sys.exit(0)

    passed = evaluate_gate(
        run_dir=run_dir,
        baseline_path=args.baseline_file,
        primary_config=args.primary_config,
        tolerance=args.tolerance,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

