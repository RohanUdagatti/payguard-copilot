#!/usr/bin/env python3
"""Phase 4 Ablation Report Generator with Non-Parametric Bootstrap 95% Confidence Intervals."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payguard.make_report")


def percentile(data: list[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def compute_bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval (2.5%, 97.5%)."""
    if not values:
        return 0.0, 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) == 1:
        return mean_val, mean_val, mean_val

    rng = random.Random(seed)
    n = len(values)
    resample_means: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        resample_means.append(sum(sample) / n)

    resample_means.sort()
    low = percentile(resample_means, 0.025)
    high = percentile(resample_means, 0.975)
    return mean_val, low, high


def fmt_ci(mean: float, low: float, high: float) -> str:
    """Format mean and CI into string."""
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def is_retrieval_match(
    chunk_meta: dict[str, Any],
    expected_doc_id: str,
    section_hint: str | None,
) -> bool:
    """Check if chunk metadata matches expected document and section."""
    if not chunk_meta:
        return False
    doc_id = str(chunk_meta.get("doc_id", ""))
    if expected_doc_id and doc_id and expected_doc_id != doc_id:
        return False
    if not section_hint:
        return True
    sec = str(chunk_meta.get("section", "")).lower()
    hint = section_hint.lower()
    if hint in sec or sec in hint:
        return True
    hint_words = set(hint.split())
    sec_words = set(sec.split())
    return len(hint_words.intersection(sec_words)) >= 2


def find_latest_run_dir(base_dir: Path = Path("eval/results/runs")) -> Path:
    """Find the most recently created run directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Runs directory {base_dir} does not exist.")
    runs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base_dir}.")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def generate_report(
    run_dir: Path,
    output_path: Path,
    n_resamples: int = 1000,
    seed: int = 42,
) -> str:
    """Generate Markdown ablation report and write to output_path."""
    manifest_file = run_dir / "manifest.json"
    scores_file = run_dir / "scores.jsonl"
    answers_file = run_dir / "answers.jsonl"
    errors_file = run_dir / "judge_errors.jsonl"

    if not scores_file.exists():
        raise FileNotFoundError(f"{scores_file} not found.")

    manifest: dict[str, Any] = {}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Load scores
    scores_by_cfg: dict[str, list[dict[str, Any]]] = {}
    with open(scores_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                scores_by_cfg.setdefault(item["config"], []).append(item)

    # Load answers
    answers_by_cfg: dict[str, list[dict[str, Any]]] = {}
    if answers_file.exists():
        with open(answers_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    answers_by_cfg.setdefault(item["config"], []).append(item)

    # Load errors
    errors: list[dict[str, Any]] = []
    if errors_file.exists():
        with open(errors_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    errors.append(json.loads(line))

    # Calculate metrics per config
    configs = list(scores_by_cfg.keys())
    # Ensure ordered: dense, hybrid, hybrid+parent-child
    ordered_cfgs = [c for c in ["dense", "hybrid", "hybrid+parent-child"] if c in configs]
    for c in configs:
        if c not in ordered_cfgs:
            ordered_cfgs.append(c)

    table_rows = []
    error_summary = []

    for cfg in ordered_cfgs:
        items = scores_by_cfg[cfg]
        ans_items = answers_by_cfg.get(cfg, [])
        n_total = len(items)
        n_scored = sum(1 for it in items if not it.get("is_nan", False))
        nan_count = sum(1 for it in items if it.get("is_nan", False))

        # Metrics lists
        faith_vals: list[float] = []
        rel_vals: list[float] = []
        prec_vals: list[float] = []
        rec_vals: list[float] = []
        abst_vals: list[float] = []

        for it in items:
            s = it.get("scores", {})
            ans = it.get("answerable", True)
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

        m_faith, l_faith, h_faith = compute_bootstrap_ci(faith_vals, n_resamples, seed)
        m_rel, l_rel, h_rel = compute_bootstrap_ci(rel_vals, n_resamples, seed)
        m_prec, l_prec, h_prec = compute_bootstrap_ci(prec_vals, n_resamples, seed)
        m_rec, l_rec, h_rec = compute_bootstrap_ci(rec_vals, n_resamples, seed)
        m_abst, l_abst, h_abst = compute_bootstrap_ci(abst_vals, n_resamples, seed)

        # Retrieval metrics from answers
        ans_answerable = [a for a in ans_items if a.get("answerable", True)]
        h1_count = 0
        h5_count = 0
        for a in ans_answerable:
            ret_meta = a.get("retrieved_meta", [])
            s_doc = a.get("source_doc", "")
            s_hint = a.get("section_hint")
            if ret_meta and is_retrieval_match(ret_meta[0], s_doc, s_hint):
                h1_count += 1
            if any(is_retrieval_match(m, s_doc, s_hint) for m in ret_meta[:5]):
                h5_count += 1

        n_ans = max(1, len(ans_answerable))
        h1_pct = (h1_count / n_ans) * 100.0 if ans_answerable else 0.0
        h5_pct = (h5_count / n_ans) * 100.0 if ans_answerable else 0.0

        # Latencies
        lats = [float(a.get("total_ms", 0.0)) for a in ans_items if a.get("total_ms")]
        lats.sort()
        p50_lat = percentile(lats, 0.50) if lats else 0.0
        p95_lat = percentile(lats, 0.95) if lats else 0.0

        table_rows.append({
            "config": cfg,
            "n": f"{n_scored}/{n_total}",
            "faithfulness": fmt_ci(m_faith, l_faith, h_faith),
            "relevancy": fmt_ci(m_rel, l_rel, h_rel),
            "precision": fmt_ci(m_prec, l_prec, h_prec),
            "recall": fmt_ci(m_rec, l_rec, h_rec),
            "abstention": fmt_ci(m_abst, l_abst, h_abst) if abst_vals else "N/A",
            "hit1": f"{h1_pct:.1f}%",
            "hit5": f"{h5_pct:.1f}%",
            "p50_lat": f"{p50_lat:.1f} ms",
            "p95_lat": f"{p95_lat:.1f} ms",
            "nans": str(nan_count),
        })

        if nan_count > 0:
            error_summary.append(f"- **{cfg}**: {nan_count} items yielded NaN scores.")

    # Negative result row for hybrid+rerank (Phase 3.1)
    table_rows.append({
        "config": "hybrid+rerank *(Phase 3.1 negative result)*",
        "n": "10/10",
        "faithfulness": "—",
        "relevancy": "—",
        "precision": "—",
        "recall": "—",
        "abstention": "—",
        "hit1": "60.0%",
        "hit5": "80.0%",
        "p50_lat": "8049.3 ms",
        "p95_lat": "8461.2 ms",
        "nans": "0",
    })

    # Build Markdown document
    run_id = manifest.get("run_id", run_dir.name)
    timestamp = manifest.get("timestamp_utc", "Unknown")
    git_hash = manifest.get("git_commit", "uncommitted")
    gen_model = manifest.get("generator_model", "Unknown")
    judge_model = manifest.get("judge_model") or "mock-judge"
    judge_provider = manifest.get("judge_provider", "local")
    ragas_ver = manifest.get("ragas_version", "0.4.3")
    golden_sha = manifest.get("golden_set_sha256", "Unknown")
    is_mock = bool(manifest.get("mock", False))

    # Compute verified questions per type and document
    golden_file = Path(manifest.get("golden_set_path", "eval/golden_set.jsonl"))
    type_counts: dict[str, int] = {}
    doc_counts: dict[str, int] = {}
    total_verified = 0
    if golden_file.exists():
        with open(golden_file, encoding="utf-8") as gf:
            for line in gf:
                line = line.strip()
                if line and not line.startswith("#"):
                    data = json.loads(line)
                    if data.get("verified", False):
                        total_verified += 1
                        t = data.get("type", "unknown")
                        d = data.get("source_doc", "unknown")
                        type_counts[t] = type_counts.get(t, 0) + 1
                        doc_counts[d] = doc_counts.get(d, 0) + 1

    md = []
    if is_mock:
        md.append("# ⚠️ MOCK RUN - NOT REAL RESULTS ⚠️")
        md.append("")
        md.append(
            "> [!CAUTION] MOCK RUN\n"
            "> This evaluation report was generated from a mock run. Scores and latencies are synthetic."
        )
        md.append("")

    md.append(f"# Phase 4 RAGAS Ablation Report: `{run_id}`")
    md.append("")
    md.append(f"- **Timestamp**: `{timestamp}`")
    md.append(f"- **Git Commit**: `{git_hash}`")
    md.append(f"- **Golden Set SHA-256**: `{golden_sha}`")
    md.append(f"- **Generator Model**: `{gen_model}` (temp=0.0)")
    md.append(f"- **Judge Model**: `{judge_provider}` / `{judge_model}`")
    md.append(f"- **RAGAS Version**: `{ragas_ver}` | **Bootstrap Resamples**: `{n_resamples}` (seed={seed})")
    md.append("")
    md.append("> [!WARNING] Self-Preference Limitation")
    md.append(
        "> When the generator and judge share the same model family, "
        "RAGAS evaluation scores may exhibit self-preference bias. "
        "Evaluator models should ideally be drawn from an independent model family "
        "when production budgets permit."
    )
    md.append("")
    md.append("## Verified Golden Questions Distribution")
    md.append(f"- **Total Human-Verified Questions**: {total_verified}")
    md.append("")
    md.append("| Question Type | Verified Count |")
    md.append("| :--- | :---: |")
    if type_counts:
        for t, cnt in sorted(type_counts.items()):
            md.append(f"| {t} | {cnt} |")
    else:
        md.append("| *(none verified)* | 0 |")
    md.append("")
    md.append("| Source Document | Verified Count |")
    md.append("| :--- | :---: |")
    if doc_counts:
        for d, cnt in sorted(doc_counts.items()):
            md.append(f"| {d} | {cnt} |")
    else:
        md.append("| *(none verified)* | 0 |")
    md.append("")
    md.append("## Evaluation Results & Ablation Matrix")
    md.append("")
    md.append(
        "| Configuration | n (scored/tot) | Faithfulness (95% CI) | Relevancy (95% CI) | Context Precision (95% CI) | Context Recall (95% CI) | Abstention Accuracy | Hit@1 | Hit@5 | p50 Total Latency | p95 Total Latency | NaNs/Errors |"
    )
    md.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for row in table_rows:
        md.append(
            f"| {row['config']} | {row['n']} | {row['faithfulness']} | {row['relevancy']} | {row['precision']} | {row['recall']} | {row['abstention']} | {row['hit1']} | {row['hit5']} | {row['p50_lat']} | {row['p95_lat']} | {row['nans']} |"
        )

    md.append("")
    md.append("## Metric Definitions & Methodology")
    md.append("1. **Confidence Intervals**: 95% non-parametric bootstrap intervals (2.5th and 97.5th percentiles) derived from 1,000 resamples.")
    md.append("2. **Abstention Accuracy**: Percentage of out-of-scope compliance questions where the copilot successfully refrained from hallucinating and returned the exact refusal message.")
    md.append("3. **Hybrid + Reranker Row**: Included as a Phase 3.1 benchmark negative result. Cross-encoder reranking increased p50 retrieval latency from 53 ms to 8,049 ms while dropping Hit@1 from 70% to 60%, and was excluded from LLM generation stages.")
    md.append("")
    md.append("## NaN & Error Log Summary")
    if error_summary or errors:
        for err in error_summary:
            md.append(err)
        for err in errors:
            md.append(f"- **Error** [{err.get('config')} - {err.get('question_id')}]: {err.get('error')}")
    else:
        md.append("✓ **0 NaNs and 0 Judge Errors detected.** All verified items evaluated successfully.")
    md.append("")

    report_text = "\n".join(md)
    # Output path routing: if mock, write to ablation_MOCK.md
    if is_mock and (
        output_path == Path("eval/results/ablation.md")
        or output_path.name == "ablation.md"
    ):
        final_output_path = output_path.with_name("ablation_MOCK.md")
    else:
        final_output_path = output_path

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Wrote ablation report to {final_output_path}")

    if is_mock:
        print("\n" + "=" * 65)
        print("          *** MOCK RUN - NOT REAL RESULTS ***          ")
        print("=" * 65)

    print(f"\n--- Run Summary: {run_id} ---")
    print(f"Judge Model: {judge_provider} / {judge_model}")
    print("Scored / Total Counts:")
    for r in table_rows:
        print(f"  - {r['config']:<42}: {r['n']}")
    print(f"Verified Questions in Golden Set: {total_verified}")
    print(f"  - By Type    : {type_counts if type_counts else 'None'}")
    print(f"  - By Document: {doc_counts if doc_counts else 'None'}")
    if is_mock:
        print("=" * 65 + "\n")

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 4 RAGAS ablation report.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to run directory. If omitted, uses the most recent run in eval/results/runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/ablation.md"),
        help="Path to write the markdown report (default: eval/results/ablation.md).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Generating report from run directory: {run_dir}")
    report = generate_report(
        run_dir=run_dir,
        output_path=args.output,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print("\n" + report)


if __name__ == "__main__":
    main()

