"""Unit and integration tests for the Phase 4 evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from scripts.eval_gate import evaluate_gate
from scripts.make_report import compute_bootstrap_ci, generate_report
from scripts.run_ragas import RateLimiter, judge_item_mock

from payguard.eval.schema import GoldenQuestion
from payguard.rag.generator import NOT_FOUND_MESSAGE


def test_golden_question_schema_valid() -> None:
    """Valid golden question validates successfully."""
    q = GoldenQuestion(
        id="q_valid_1",
        question="What is the timeframe for dispute condition 13.1?",
        ground_truth="The timeframe is 120 calendar days.",
        source_doc="visa_chargeback_guide",
        section_hint="Condition 13.1",
        region="US",
        type="lookup",
        answerable=True,
        verified=True,
        needs_ground_truth=False,
    )
    assert q.id == "q_valid_1"
    assert q.answerable is True
    assert q.verified is True


def test_golden_question_luhn_rejection() -> None:
    """Questions containing Luhn-valid PANs are rejected."""
    # 4111111111111111 is a known valid Luhn Visa card number
    with pytest.raises(ValidationError, match="PAN detected"):
        GoldenQuestion(
            id="q_pan_fail",
            question="Can chargeback apply to card 4111111111111111?",
            ground_truth="Yes.",
            source_doc="visa_chargeback_guide",
            type="lookup",
            answerable=True,
            verified=True,
        )


def test_golden_question_out_of_scope_validation() -> None:
    """Out-of-scope questions must have answerable=False and exact ground_truth."""
    # Valid out of scope
    q = GoldenQuestion(
        id="q_oos_valid",
        question="What are PayPal seller protection rules for digital goods?",
        ground_truth=NOT_FOUND_MESSAGE,
        source_doc="none",
        type="out_of_scope",
        answerable=False,
        verified=True,
    )
    assert q.answerable is False

    # Out of scope with answerable=True must fail
    with pytest.raises(ValidationError, match="must have answerable=False"):
        GoldenQuestion(
            id="q_oos_fail_1",
            question="What are PayPal seller protection rules for digital goods?",
            ground_truth=NOT_FOUND_MESSAGE,
            source_doc="none",
            type="out_of_scope",
            answerable=True,
            verified=True,
        )

    # Out of scope with wrong ground_truth must fail
    with pytest.raises(ValidationError, match=NOT_FOUND_MESSAGE):
        GoldenQuestion(
            id="q_oos_fail_2",
            question="What are PayPal seller protection rules for digital goods?",
            ground_truth="They are protected.",
            source_doc="none",
            type="out_of_scope",
            answerable=False,
            verified=True,
        )


def test_rate_limiter_rpd_cap() -> None:
    """RateLimiter raises RuntimeError when daily limit is reached."""
    limiter = RateLimiter(rpm=6000, rpd=3)
    # 3 calls allowed
    limiter.wait_before_call()
    limiter.wait_before_call()
    limiter.wait_before_call()

    # 4th call exceeds RPD cap
    with pytest.raises(RuntimeError, match="Daily rate limit of 3 calls"):
        limiter.wait_before_call()


def test_abstention_scoring_logic() -> None:
    """Verify abstention and false-abstention scoring behavior."""
    # Correct abstention on out-of-scope question
    oos_correct = judge_item_mock(
        {
            "answerable": False,
            "answer": NOT_FOUND_MESSAGE,
        }
    )
    assert oos_correct["correct_abstention"] == 1.0
    assert oos_correct["false_abstention"] == 0.0

    # Hallucination on out-of-scope question
    oos_hallucinated = judge_item_mock(
        {
            "answerable": False,
            "answer": "PayPal gives 180 days for disputes.",
        }
    )
    assert oos_hallucinated["correct_abstention"] == 0.0
    assert oos_hallucinated["false_abstention"] == 0.0

    # False abstention on answerable question
    ans_wrongly_refused = judge_item_mock(
        {
            "answerable": True,
            "answer": NOT_FOUND_MESSAGE,
        }
    )
    assert ans_wrongly_refused["correct_abstention"] == 0.0
    assert ans_wrongly_refused["false_abstention"] == 1.0

    # Correct response on answerable question
    ans_answered = judge_item_mock(
        {
            "answerable": True,
            "answer": "The dispute timeframe is 120 calendar days.",
        }
    )
    assert ans_answered["correct_abstention"] == 0.0
    assert ans_answered["false_abstention"] == 0.0


def test_bootstrap_confidence_interval() -> None:
    """Bootstrap CI captures mean and bounds with seed reproducibility."""
    data = [0.80, 0.85, 0.90, 0.95, 1.00]
    mean_val, low, high = compute_bootstrap_ci(data, n_resamples=500, seed=42)

    assert round(mean_val, 2) == 0.90
    assert 0.80 <= low <= mean_val
    assert mean_val <= high <= 1.00

    # Seed reproducibility
    m2, l2, h2 = compute_bootstrap_ci(data, n_resamples=500, seed=42)
    assert mean_val == m2
    assert low == l2
    assert high == h2


def test_candidate_isolation_unverified_not_scored(tmp_path: Path) -> None:
    """Ensure candidate questions with verified=False are excluded from scoring."""
    golden_path = tmp_path / "test_golden.jsonl"
    q_verified = GoldenQuestion(
        id="q_ver",
        question="What is the timeframe?",
        ground_truth="120 days.",
        source_doc="visa_chargeback_guide",
        type="lookup",
        answerable=True,
        verified=True,
    )
    q_unverified = GoldenQuestion(
        id="q_unver",
        question="Unverified question draft?",
        ground_truth="",
        source_doc="visa_chargeback_guide",
        type="lookup",
        answerable=True,
        verified=False,
        needs_ground_truth=True,
    )

    with open(golden_path, "w", encoding="utf-8") as f:
        f.write(q_verified.model_dump_json() + "\n")
        f.write(q_unverified.model_dump_json() + "\n")

    # Read and filter verified
    verified_questions = []
    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            item = GoldenQuestion.model_validate_json(line)
            if item.verified:
                verified_questions.append(item)

    assert len(verified_questions) == 1
    assert verified_questions[0].id == "q_ver"


def test_report_generation(tmp_path: Path) -> None:
    """Generate Markdown report from run directory fixtures."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True)

    manifest = {
        "run_id": "test_run_001",
        "timestamp_utc": "2026-11-05T12:00:00Z",
        "git_commit": "abc1234",
        "golden_set_sha256": "fakehash",
        "generator_model": "mock-gen",
        "judge_provider": "gemini",
        "judge_model": "mock-judge",
        "ragas_version": "0.4.3",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    answers = [
        {
            "run_id": "test_run_001",
            "config": "hybrid",
            "question_id": "q1",
            "total_ms": 65.0,
            "answerable": True,
            "source_doc": "doc1",
            "section_hint": "sec1",
            "retrieved_meta": [{"doc_id": "doc1", "section": "sec1"}],
        }
    ]
    with open(run_dir / "answers.jsonl", "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")

    scores = [
        {
            "run_id": "test_run_001",
            "config": "hybrid",
            "question_id": "q1",
            "type": "lookup",
            "answerable": True,
            "scores": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.90,
                "context_precision": 0.88,
                "context_recall": 0.92,
                "correct_abstention": 0.0,
                "false_abstention": 0.0,
            },
            "is_nan": False,
        }
    ]
    with open(run_dir / "scores.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    out_report = tmp_path / "ablation.md"
    report_text = generate_report(run_dir, out_report, n_resamples=100)

    assert "Phase 4 RAGAS Ablation Report: `test_run_001`" in report_text
    assert "Self-Preference Limitation" in report_text
    assert "hybrid+rerank *(Phase 3.1 negative result)*" in report_text
    assert "0.950" in report_text
    assert out_report.exists()


def test_eval_gate_pass_and_fail(tmp_path: Path) -> None:
    """Evaluate gate logic correctly identifies passes and regressions."""
    baseline_file = tmp_path / "baseline.json"
    baseline_data: dict[str, Any] = {
        "baseline_run_id": "base_run",
        "configs": {
            "hybrid": {
                "faithfulness": 0.90,
                "context_recall": 0.85,
                "abstention_accuracy": 1.0,
            }
        },
    }
    baseline_file.write_text(json.dumps(baseline_data))

    # Run dir 1: Passing run
    pass_dir = tmp_path / "pass_run"
    pass_dir.mkdir()
    scores_pass = [
        {
            "config": "hybrid",
            "question_id": "q1",
            "answerable": True,
            "scores": {"faithfulness": 0.92, "context_recall": 0.88},
            "is_nan": False,
        },
        {
            "config": "hybrid",
            "question_id": "q2",
            "answerable": False,
            "scores": {"correct_abstention": 1.0},
            "is_nan": False,
        },
    ]
    with open(pass_dir / "scores.jsonl", "w") as f:
        for s in scores_pass:
            f.write(json.dumps(s) + "\n")

    assert evaluate_gate(pass_dir, baseline_file, primary_config="hybrid") is True

    # Run dir 2: Regressing run (faithfulness dropped from 0.90 to 0.70)
    fail_dir = tmp_path / "fail_run"
    fail_dir.mkdir()
    scores_fail = [
        {
            "config": "hybrid",
            "question_id": "q1",
            "answerable": True,
            "scores": {"faithfulness": 0.70, "context_recall": 0.88},
            "is_nan": False,
        },
        {
            "config": "hybrid",
            "question_id": "q2",
            "answerable": False,
            "scores": {"correct_abstention": 1.0},
            "is_nan": False,
        },
    ]
    with open(fail_dir / "scores.jsonl", "w") as f:
        for s in scores_fail:
            f.write(json.dumps(s) + "\n")

    assert evaluate_gate(fail_dir, baseline_file, primary_config="hybrid") is False


def test_eval_gate_two_config_partial_regression(tmp_path: Path) -> None:
    """Gate evaluates all configs and exits non-zero if ANY config regresses."""
    baseline_file = tmp_path / "baseline_two_cfg.json"
    baseline_data: dict[str, Any] = {
        "baseline_run_id": "base_run_multi",
        "mock": False,
        "configs": {
            "dense": {
                "faithfulness": 0.85,
                "context_recall": 0.80,
                "abstention_accuracy": 1.0,
            },
            "hybrid": {
                "faithfulness": 0.90,
                "context_recall": 0.88,
                "abstention_accuracy": 1.0,
            },
        },
    }
    baseline_file.write_text(json.dumps(baseline_data))

    run_dir = tmp_path / "two_cfg_run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "r2", "mock": False}))

    scores = [
        # dense PASSES (faithfulness 0.88 >= 0.85, context_recall 0.82 >= 0.80)
        {
            "config": "dense",
            "question_id": "q1",
            "answerable": True,
            "scores": {"faithfulness": 0.88, "context_recall": 0.82},
            "is_nan": False,
        },
        {
            "config": "dense",
            "question_id": "q2",
            "answerable": False,
            "scores": {"correct_abstention": 1.0},
            "is_nan": False,
        },
        # hybrid REGRESSES (faithfulness 0.70 < 0.90)
        {
            "config": "hybrid",
            "question_id": "q1",
            "answerable": True,
            "scores": {"faithfulness": 0.70, "context_recall": 0.90},
            "is_nan": False,
        },
        {
            "config": "hybrid",
            "question_id": "q2",
            "answerable": False,
            "scores": {"correct_abstention": 1.0},
            "is_nan": False,
        },
    ]
    with open(run_dir / "scores.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    # Even though dense passes, hybrid regresses so gate must fail (return False)
    assert (
        evaluate_gate(
            run_dir,
            baseline_file,
            primary_config="hybrid",
        )
        is False
    )


def test_gate_refuses_mock_run(tmp_path: Path) -> None:
    """Gate refuses to evaluate or create baseline from a mock run."""
    mock_dir = tmp_path / "mock_run"
    mock_dir.mkdir()
    (mock_dir / "manifest.json").write_text(json.dumps({"run_id": "m1", "mock": True}))
    (mock_dir / "scores.jsonl").write_text(
        json.dumps(
            {
                "config": "hybrid",
                "question_id": "q1",
                "answerable": True,
                "scores": {"faithfulness": 0.9},
                "is_nan": False,
            }
        )
        + "\n"
    )

    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "baseline_run_id": "base",
                "mock": False,
                "configs": {"hybrid": {"faithfulness": 0.9}},
            }
        )
    )

    # Refuse evaluate_gate on mock run
    assert evaluate_gate(mock_dir, baseline_file) is False

    # Refuse create_baseline from mock run
    with pytest.raises(SystemExit):
        from scripts.eval_gate import create_baseline

        create_baseline(mock_dir, tmp_path / "new_base.json")


def test_generator_model_refuses_latest_alias() -> None:
    """validate_generator_model refuses models ending in '-latest'."""
    from scripts.run_ragas import validate_generator_model

    with pytest.raises(ValueError, match="not an unpinned '-latest' alias"):
        validate_generator_model("gemini-flash-latest")

    with pytest.raises(ValueError, match="not an unpinned '-latest' alias"):
        validate_generator_model("claude-3-5-sonnet-latest")

    # Pinned model passes
    validate_generator_model("gemini-2.5-flash")


def test_mock_report_writes_ablation_mock(tmp_path: Path) -> None:
    """Mock runs route output to ablation_MOCK.md and include mock warnings."""
    mock_dir = tmp_path / "mock_test_run"
    mock_dir.mkdir()
    (mock_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "mock_run_99",
                "mock": True,
                "golden_set_path": "eval/golden_set.jsonl",
            }
        )
    )
    (mock_dir / "scores.jsonl").write_text(
        json.dumps(
            {
                "config": "dense",
                "question_id": "q1",
                "type": "lookup",
                "answerable": True,
                "scores": {"faithfulness": 0.9},
                "is_nan": False,
            }
        )
        + "\n"
    )

    out_file = tmp_path / "eval/results/ablation.md"
    report_text = generate_report(mock_dir, out_file)

    assert "MOCK RUN - NOT REAL RESULTS" in report_text
    # Output was routed to ablation_MOCK.md
    expected_mock_path = tmp_path / "eval/results/ablation_MOCK.md"
    assert expected_mock_path.exists()
