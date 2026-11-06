#!/usr/bin/env python3
"""Generate candidate evaluation questions stratified across docs and import manual questions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import QdrantClient

from payguard.config import get_settings
from payguard.eval.schema import GoldenQuestion
from payguard.rag.generator import NOT_FOUND_MESSAGE
from payguard.rag.ingest.chunker import is_junk_boilerplate_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payguard.generate_testset")

DOC_IDS = [
    "fed_payment_risk",
    "nist_digital_identity_sp800_63b",
    "visa_chargeback_guide",
]

_PROMPT_TEMPLATE = """You are an expert evaluator in payment systems, regulatory compliance, and digital identity.
Based on the following reference policy chunk, generate a realistic question and a precise factual ground-truth answer.

Document ID: {doc_id}
Section: {section}
Context:
{text}

Select ONE question type from:
- lookup: Direct factual query answered by specific sentence.
- paraphrase: Complex question rephrasing regulatory requirements.
- near_miss: Question asking about closely related rules or adjacent conditions.
- multi_section: Question synthesizing multiple rules or conditions in this chunk.

Return ONLY a JSON object with this schema:
{{
  "question": "<question string>",
  "ground_truth": "<factual ground truth answer string>",
  "type": "<lookup|paraphrase|near_miss|multi_section>",
  "section_hint": "<section title>"
}}
"""


def load_manual_questions(yaml_path: Path) -> list[GoldenQuestion]:
    """Import ALL entries from manual_questions.yaml with verified=false and needs_ground_truth=true."""
    if not yaml_path.exists():
        logger.warning(f"Manual questions file {yaml_path} not found.")
        return []

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_list: list[dict[str, Any]] = data.get("questions", [])
    candidates: list[GoldenQuestion] = []

    for idx, item in enumerate(raw_list, start=1):
        q_text = item.get("question", "").strip()
        doc_id = item.get("expected_doc_id", "default").strip()
        hint = item.get("expected_section_hint")
        candidates.append(
            GoldenQuestion(
                id=f"manual_{idx:03d}",
                question=q_text,
                ground_truth="",
                source_doc=doc_id,
                section_hint=hint,
                region="US",
                type="lookup",
                answerable=True,
                verified=False,
                needs_ground_truth=True,
            )
        )
    return candidates


def fetch_stratified_chunks(
    client: QdrantClient,
    collection_name: str,
    target_count: int,
) -> list[dict[str, Any]]:
    """Fetch non-junk parent chunks stratified across all target document IDs."""
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {d: [] for d in DOC_IDS}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            payload = pt.payload or {}
            doc_id = str(payload.get("doc_id", ""))
            text = str(payload.get("text", ""))
            is_child = payload.get("is_child", False)

            if is_child:
                continue
            if is_junk_boilerplate_chunk(text):
                continue
            if doc_id in chunks_by_doc:
                chunks_by_doc[doc_id].append(
                    {
                        "chunk_id": str(pt.id),
                        "doc_id": doc_id,
                        "section": str(payload.get("section", "")),
                        "page_start": payload.get("page_start", 1),
                        "text": text,
                    }
                )

        if next_offset is None or all(len(v) >= target_count for v in chunks_by_doc.values()):
            break
        offset = next_offset

    stratified: list[dict[str, Any]] = []
    per_doc = max(1, target_count // len(DOC_IDS))
    for doc_id, c_list in chunks_by_doc.items():
        stratified.extend(c_list[:per_doc])
    return stratified[:target_count]


def generate_candidate_with_llm(
    chunk: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Call LLM to propose question and ground truth from chunk text."""
    prompt = _PROMPT_TEMPLATE.format(
        doc_id=chunk["doc_id"],
        section=chunk["section"],
        text=chunk["text"][:1500],
    )
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp)).strip()
    # Extract JSON block
    if content.startswith("```"):
        content = content.strip("`").lstrip("json\n")
    return json.loads(content)


def generate_mock_candidate(chunk: dict[str, Any], idx: int) -> dict[str, Any]:
    """Deterministic mock generation for offline tests and dry runs."""
    types = ["lookup", "paraphrase", "near_miss", "multi_section"]
    q_type = types[idx % len(types)]
    sec = chunk.get("section") or "General Policy"
    return {
        "question": f"What are the compliance requirements outlined in {sec}?",
        "ground_truth": f"The policy specifies standard operating requirements under {sec}.",
        "type": q_type,
        "section_hint": sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden set candidate questions.")
    parser.add_argument("--limit", type=int, default=15, help="Number of new questions to generate.")
    parser.add_argument("--dry-run", action="store_true", help="Print call estimates without calling LLM.")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic questions without real LLM.")
    parser.add_argument(
        "--manual-file",
        type=Path,
        default=Path("eval/manual_questions.yaml"),
        help="Path to manual_questions.yaml to import.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/golden_candidates.jsonl"),
        help="Path to output candidates JSONL file.",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Step 1: Import all manual questions
    manual_candidates = load_manual_questions(args.manual_file)
    print(f"Imported {len(manual_candidates)} manual candidate entries from {args.manual_file}.")

    # Step 2: Dry run check
    if args.dry_run:
        print("\n=== DRY-RUN GENERATION ESTIMATES ===")
        print(f"Manual Questions to Import : {len(manual_candidates)} (0 LLM calls)")
        print(f"Target Synthetic Candidates: {args.limit}")
        print(f"Estimated LLM Calls        : {args.limit}")
        print(f"Target Output File         : {args.output}")
        print("Dry run completed. No external API calls were made.")
        return

    # Step 3: Fetch chunks from Qdrant
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    chunks = fetch_stratified_chunks(client, settings.qdrant_collection, args.limit)
    print(f"Sampled {len(chunks)} stratified chunks across {DOC_IDS}.")

    llm = None
    if not args.mock:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from pydantic import SecretStr

        api_key = settings.resolve_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY required for real LLM generation; use --mock or --dry-run.")
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=SecretStr(api_key),
            temperature=0,
        )

    # Step 4: Propose questions
    generated_candidates: list[GoldenQuestion] = []
    for idx, c in enumerate(chunks, start=1):
        if args.mock or llm is None:
            data = generate_mock_candidate(c, idx)
        else:
            try:
                data = generate_candidate_with_llm(c, llm)
            except Exception as exc:
                logger.warning(f"Candidate generation failed for chunk {c['chunk_id']}: {exc}")
                data = generate_mock_candidate(c, idx)

        cand = GoldenQuestion(
            id=f"cand_{idx:03d}",
            question=data["question"],
            ground_truth=data.get("ground_truth", ""),
            source_doc=c["doc_id"],
            section_hint=data.get("section_hint") or c["section"],
            region="US",
            type=data.get("type", "lookup"),
            answerable=True,
            verified=False,
            needs_ground_truth=False,
        )
        generated_candidates.append(cand)

    # Step 5: Add an out-of-scope candidate to candidate pool
    out_of_scope_cand = GoldenQuestion(
        id=f"cand_{(len(generated_candidates)+1):03d}",
        question="What are the maximum penalties for crypto trading violations under MiCA in Germany?",
        ground_truth=NOT_FOUND_MESSAGE,
        source_doc="none",
        section_hint=None,
        region="EU",
        type="out_of_scope",
        answerable=False,
        verified=False,
        needs_ground_truth=False,
    )
    generated_candidates.append(out_of_scope_cand)

    # Step 6: Write to candidates JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_candidates = manual_candidates + generated_candidates

    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_candidates:
            f.write(item.model_dump_json() + "\n")

    print(f"\nSuccessfully wrote {len(all_candidates)} candidates ({len(manual_candidates)} manual, {len(generated_candidates)} generated) to {args.output}.")
    print("Next step: review candidates by hand and move accepted ones to eval/golden_set.jsonl with verified=true.")


if __name__ == "__main__":
    main()

