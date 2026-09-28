#!/usr/bin/env python
"""Strict, method-symmetric OOD evaluation for reasoning-like SAE features.

This module separates four operations that must not share labels:

1. ``build-candidates`` retrieves the same number of candidate coordinates for
   each SAE from frozen, blindly generated descriptions of their natural top
   activations.
2. ``judge`` applies a method-blind semantic audit to those candidate examples
   and independently labels a fresh natural-text pool.
3. ``extract`` evaluates only the frozen candidate coordinates on the fresh
   pool, using each SAE's native complete-chunk activation convention.
4. ``analyze`` selects the first semantically accepted candidate without using
   fresh-pool labels and measures its document/source OOD selectivity.

The design is intentionally conservative.  A feature can count as robust only
when it is first discoverable without task labels and then separates genuine
multi-step relation instances from same-domain hard negatives in unseen
documents.  Token and Temporal SAEs receive max-over-token scoring, which is
the same convention used to construct their original feature evidence and is
generous to those baselines.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from chunk_saes.artifacts import resolve_sae_artifact_set
from chunk_saes.modeling import TargetLayerExtractor
from evals.reasoning.evaluate_reason_counterparts import FrozenEncoder


METHODS = ("token", "temporal", "mean", "cross")
DEFAULT_DEFINITIONS = {
    "causal_mechanism": (
        "Experimental or observational scientific prose that traces a concrete "
        "intervention, perturbation, or initiating factor through an "
        "intermediate biological or physical mechanism to a downstream "
        "measured consequence. The defining property is an explicit "
        "multi-step mechanism, not merely association, entity lists, methods, "
        "or isolated results."
    ),
    "mathematical_derivation": (
        "Formal mathematical or mathematical-physics prose that actively "
        "carries out a multi-step proof or derivation, linking assumptions "
        "through symbolic transformations, case analysis, inequalities, "
        "substitutions, or intermediate lemmas to a justified result. "
        "Definitions, theorem statements, notation lists, and final answers "
        "alone do not count."
    ),
    "technical_diagnosis": (
        "Technical troubleshooting prose that states an observed software, "
        "build, configuration, or system failure, identifies or tests its "
        "cause, and gives or validates a concrete corrective action. Error "
        "logs alone, generic instructions, and descriptions without a "
        "diagnosis-to-remedy chain do not count."
    ),
    "rule_guided_decision": (
        "Legal, regulatory, tax, HR, or institutional guidance that applies "
        "explicit rules, eligibility conditions, constraints, deadlines, or "
        "exceptions to a concrete case and reaches an actionable compliance "
        "decision. Mere rule lists, citations, or factual narratives without "
        "application do not count."
    ),
    "methodological_inference": (
        "Quantitative research prose that reasons from assumptions, bias, "
        "uncertainty, estimator behavior, experimental design, or data "
        "limitations to a qualified conclusion about what can or cannot be "
        "inferred. Mere descriptions of methods, measurements, or numerical "
        "results do not count."
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _parse_judge_description(raw: str) -> str:
    try:
        payload = json.loads(raw.strip())
        description = str(payload.get("description", "")).strip()
        if description:
            return description
    except (json.JSONDecodeError, AttributeError):
        pass
    return raw.strip()


def _embed_texts(
    texts: list[str],
    *,
    model_path: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            ).to(device)
            hidden = model(
                **encoded,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(F.normalize(pooled, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def _load_top_chunks(
    *,
    evidence_dir: Path,
    needed: dict[str, set[int]],
    top_n: int,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    chunks: dict[int, dict[str, Any]] = {}
    for path in sorted((evidence_dir / "partials").glob("chunks-rank*.jsonl")):
        for row in _read_jsonl(path):
            chunks[int(row["global_chunk_id"])] = row

    candidates: dict[tuple[str, int], list[tuple[float, int]]] = defaultdict(
        list
    )
    for path in sorted(
        (evidence_dir / "partials").glob("candidates-rank*.safetensors")
    ):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for method in METHODS:
                feature_ids = handle.get_tensor(
                    f"{method}_feature_ids"
                ).numpy()
                indices = handle.get_tensor(f"{method}_indices").numpy()
                scores = handle.get_tensor(f"{method}_scores").numpy()
                for feature_id, row_indices, row_scores in zip(
                    feature_ids,
                    indices,
                    scores,
                    strict=True,
                ):
                    feature_id = int(feature_id)
                    if feature_id not in needed[method]:
                        continue
                    candidates[(method, feature_id)].extend(
                        (float(score), int(index))
                        for score, index in zip(
                            row_scores,
                            row_indices,
                            strict=True,
                        )
                        if int(index) >= 0
                    )

    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key, rows in candidates.items():
        best_by_chunk: dict[int, float] = {}
        for score, chunk_id in rows:
            best_by_chunk[chunk_id] = max(
                score,
                best_by_chunk.get(chunk_id, float("-inf")),
            )
        ordered = sorted(
            ((score, chunk_id) for chunk_id, score in best_by_chunk.items()),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        used_documents: set[str] = set()
        for score, chunk_id in ordered:
            chunk = chunks[chunk_id]
            document = str(chunk["doc_id"])
            if document in used_documents:
                continue
            used_documents.add(document)
            selected.append(
                {
                    "activation": score,
                    "global_chunk_id": chunk_id,
                    "doc_id": document,
                    "length": int(chunk["length"]),
                    "text": str(chunk["text"]),
                }
            )
            if len(selected) == top_n:
                break
        result[key] = selected
    return result


def build_candidates(args: argparse.Namespace) -> None:
    evidence_dir = Path(args.evidence_dir)
    blind_map = json.loads(
        (evidence_dir / "blind_map.json").read_text(encoding="utf-8")
    )
    judge_rows = {
        row["blind_id"]: row
        for row in _read_jsonl(Path(args.level_judgments))
    }
    by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    for blind_id, metadata in blind_map.items():
        judged = judge_rows[blind_id]
        level = str(judged["level"])
        if level not in {"3", "4"}:
            continue
        by_method[str(metadata["method"])].append(
            {
                "feature_id": int(metadata["feature_id"]),
                "level": level,
                "description": _parse_judge_description(
                    str(judged["raw_response"])
                ),
            }
        )
    for method in METHODS:
        by_method[method].sort(key=lambda row: row["feature_id"])

    definitions = DEFAULT_DEFINITIONS
    flattened = list(definitions.values())
    offsets: dict[str, tuple[int, int]] = {}
    for method in METHODS:
        start = len(flattened)
        flattened.extend(row["description"] for row in by_method[method])
        offsets[method] = (start, len(flattened))
    embeddings = _embed_texts(
        flattened,
        model_path=args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    target_embeddings = embeddings[: len(definitions)]

    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scenario_index, (scenario, definition) in enumerate(
        definitions.items()
    ):
        pools[scenario] = {}
        for method in METHODS:
            rows = by_method[method]
            start, stop = offsets[method]
            semantic = embeddings[start:stop] @ target_embeddings[
                scenario_index
            ]
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
            ).fit_transform(
                [definition] + [row["description"] for row in rows]
            )
            lexical = cosine_similarity(tfidf[0], tfidf[1:]).ravel()
            semantic_order = np.argsort(semantic)[::-1]
            lexical_order = np.argsort(lexical)[::-1]
            semantic_rank = {
                int(index): rank
                for rank, index in enumerate(semantic_order)
            }
            lexical_rank = {
                int(index): rank
                for rank, index in enumerate(lexical_order)
            }
            union = set(
                semantic_order[: args.candidate_count].astype(int).tolist()
            )
            union.update(
                lexical_order[: args.candidate_count].astype(int).tolist()
            )
            ordered = sorted(
                union,
                key=lambda index: (
                    1.0 / (20 + semantic_rank[index])
                    + 1.0 / (20 + lexical_rank[index]),
                    float(semantic[index]),
                    float(lexical[index]),
                    -int(rows[index]["feature_id"]),
                ),
                reverse=True,
            )[: args.candidate_count]
            pools[scenario][method] = [
                {
                    **rows[index],
                    "rank": rank,
                    "semantic_similarity": float(semantic[index]),
                    "tfidf_similarity": float(lexical[index]),
                }
                for rank, index in enumerate(ordered, 1)
            ]

    needed = {
        method: {
            int(row["feature_id"])
            for scenario in pools.values()
            for row in scenario[method]
        }
        for method in METHODS
    }
    top_chunks = _load_top_chunks(
        evidence_dir=evidence_dir,
        needed=needed,
        top_n=args.evidence_chunks,
    )
    for scenario in pools.values():
        for method, rows in scenario.items():
            for row in rows:
                row["top_activating_chunks"] = top_chunks[
                    (method, int(row["feature_id"]))
                ]

    output = {
        "format": "eval7-apple-to-apple-candidate-pool-v1",
        "complete": True,
        "definitions": definitions,
        "protocol": {
            "feature_sample_per_method": 1000,
            "eligible_blind_levels": ["3", "4"],
            "candidate_count_per_method_scenario": args.candidate_count,
            "candidate_retrieval": (
                "reciprocal-rank fusion of semantic embedding and TF-IDF "
                "over identically generated blind descriptions"
            ),
            "candidate_evidence": (
                f"{args.evidence_chunks} strongest distinct-document complete "
                "chunks per feature"
            ),
            "method_identity_used": False,
            "task_labels_used": False,
        },
        "pools": pools,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class LocalJudge:
    def __init__(
        self,
        *,
        model_path: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def run(
        self,
        prompts: list[str],
        *,
        max_input_tokens: int,
        progress_label: str,
    ) -> list[str]:
        output: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            rendered = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in batch
            ]
            encoded = self.tokenizer(
                rendered,
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
            output.extend(
                self.tokenizer.batch_decode(
                    generated[:, encoded.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            )
            if start % (10 * self.batch_size) == 0:
                print(
                    f"[{progress_label}] "
                    f"{min(start + len(batch), len(prompts))}/{len(prompts)}",
                    flush=True,
                )
        return output


def _candidate_prompt(
    row: dict[str, Any],
    definition: str,
    *,
    variant: int,
) -> str:
    evidence = "\n\n".join(
        f"EXAMPLE {index + 1}:\n{chunk['text']}"
        for index, chunk in enumerate(row["top_activating_chunks"])
    )
    if variant == 0:
        rule = (
            "A match requires the examples to repeatedly execute the target "
            "operation itself, including its linked relation structure. Do "
            "not reward generic topic, genre, scientific style, equations, "
            "citations, error words, rule words, or conclusion words."
        )
    else:
        rule = (
            "Be conservative and judge the common relation across examples, "
            "not one cherry-picked passage. A narrower domain-specific "
            "realization may match only when the required multi-step relation "
            "is repeatedly visible; topical association alone is false."
        )
    return (
        "You are a blinded semantic auditor. SAE method, feature ID, retrieval "
        "rank, and benchmark labels are hidden.\n\n"
        f"TARGET FUNCTION:\n{definition}\n\n"
        f"DECISION RULE:\n{rule}\n\n"
        f"TOP ACTIVATING EXAMPLES:\n{evidence}\n\n"
        'Return JSON only: {"match":true or false}'
    )


def _fresh_prompt(
    row: dict[str, Any],
    definition: str,
    *,
    variant: int,
) -> str:
    if variant == 0:
        rule = (
            "Positive only if the visible passage itself executes the linked "
            "target operation with at least two connected steps. Reject "
            "topic-only, methods-only, equations-only, error-only, rule-only, "
            "entity-list, or bare-conclusion passages."
        )
    else:
        rule = (
            "Apply a strict relation test, not a keyword or domain test. "
            "Accept only when the passage visibly links premises, evidence, or "
            "conditions through an intermediate step to the stated inference, "
            "diagnosis, decision, derivation, or consequence. Do not infer "
            "missing steps."
        )
    return (
        "You are a blinded annotator of unseen natural text. SAE method, "
        "feature IDs, activation values, and retrieval metadata are hidden.\n\n"
        f"TARGET FUNCTION:\n{definition}\n\n"
        f"DECISION RULE:\n{rule}\n\n"
        f"PASSAGE:\n{row['text']}\n\n"
        'Return JSON only: {"label":"positive" or "negative",'
        '"confidence":"high" or "medium" or "low"}'
    )


def _parse_match(raw: str) -> bool | None:
    try:
        payload = json.loads(raw.strip())
        if isinstance(payload.get("match"), bool):
            return bool(payload["match"])
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"?match"?\s*:\s*(true|false)', raw, re.I)
    return None if match is None else match.group(1).lower() == "true"


def _parse_label(raw: str) -> tuple[int | None, str]:
    try:
        payload = json.loads(raw.strip())
        label = str(payload.get("label", "")).lower()
        confidence = str(payload.get("confidence", "low")).lower()
        if label in {"positive", "negative"}:
            return int(label == "positive"), confidence
    except (json.JSONDecodeError, AttributeError):
        pass
    lowered = raw.lower()
    if "positive" in lowered and "negative" not in lowered:
        return 1, "low"
    if "negative" in lowered:
        return 0, "low"
    return None, "parse_error"


def _resumable_judge_pass(
    *,
    judge: LocalJudge,
    prompts: list[str],
    output_path: Path,
    max_input_tokens: int,
    progress_label: str,
) -> list[str]:
    completed: dict[int, str] = {}
    if output_path.exists():
        for row in _read_jsonl(output_path):
            completed[int(row["index"])] = str(row["raw"])
    pending = [index for index in range(len(prompts)) if index not in completed]
    print(
        f"[{progress_label}] resumed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        with output_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), judge.batch_size):
                indices = pending[start : start + judge.batch_size]
                generated = judge.run(
                    [prompts[index] for index in indices],
                    max_input_tokens=max_input_tokens,
                    progress_label=progress_label,
                )
                for index, raw in zip(indices, generated, strict=True):
                    completed[index] = raw
                    handle.write(
                        json.dumps(
                            {"index": index, "raw": raw},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                if start % (10 * judge.batch_size) == 0:
                    print(
                        f"[{progress_label}] durable "
                        f"{min(start + len(indices), len(pending))}/"
                        f"{len(pending)}",
                        flush=True,
                    )
    return [completed[index] for index in range(len(prompts))]


def _judge_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pool = json.loads(Path(args.candidate_pool).read_text(encoding="utf-8"))
    definitions = pool["definitions"]
    candidates = [
        {"scenario": scenario, "method": method, **row}
        for scenario, method_rows in pool["pools"].items()
        for method, rows in method_rows.items()
        for row in rows
    ]
    fresh = _read_jsonl(Path(args.fresh_pool))
    return pool, definitions, candidates, fresh


def _finalize_judgments(
    args: argparse.Namespace,
    *,
    candidates: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
) -> None:
    work_dir = Path(args.work_dir)
    candidate_passes = [
        [
            str(row["raw"])
            for row in sorted(
                _read_jsonl(work_dir / f"candidate-pass-{variant}.jsonl"),
                key=lambda row: int(row["index"]),
            )
        ]
        for variant in (0, 1)
    ]
    fresh_passes = [
        [
            str(row["raw"])
            for row in sorted(
                _read_jsonl(work_dir / f"fresh-pass-{variant}.jsonl"),
                key=lambda row: int(row["index"]),
            )
        ]
        for variant in (0, 1)
    ]
    if any(len(rows) != len(candidates) for rows in candidate_passes):
        raise ValueError("candidate judge passes are incomplete")
    if any(len(rows) != len(fresh) for rows in fresh_passes):
        raise ValueError("fresh-text judge passes are incomplete")

    candidate_results = []
    for index, row in enumerate(candidates):
        votes = [
            _parse_match(candidate_passes[variant][index])
            for variant in (0, 1)
        ]
        candidate_results.append(
            {
                **row,
                "match": all(vote is True for vote in votes),
                "votes": votes,
                "raw_judgments": [
                    candidate_passes[variant][index]
                    for variant in (0, 1)
                ],
                "judge_model": args.judge_model,
                "method_hidden_in_prompt": True,
                "feature_id_hidden_in_prompt": True,
                "benchmark_labels_hidden_in_prompt": True,
            }
        )
    Path(args.candidate_judgments).write_text(
        json.dumps(candidate_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fresh_results = []
    for index, row in enumerate(fresh):
        parsed = [
            _parse_label(fresh_passes[variant][index])
            for variant in (0, 1)
        ]
        label = (
            parsed[0][0]
            if parsed[0][0] is not None and parsed[0][0] == parsed[1][0]
            else None
        )
        fresh_results.append(
            {
                **row,
                "label": label,
                "agreement": label is not None,
                "votes": [item[0] for item in parsed],
                "confidences": [item[1] for item in parsed],
                "raw_judgments": [
                    fresh_passes[variant][index]
                    for variant in (0, 1)
                ],
                "judge_model": args.judge_model,
                "method_hidden_in_prompt": True,
                "feature_id_hidden_in_prompt": True,
                "activations_hidden_in_prompt": True,
            }
        )
    _write_jsonl(Path(args.fresh_judgments), fresh_results)
    print(
        "candidate matches:",
        Counter((row["method"], row["match"]) for row in candidate_results),
    )
    print(
        "fresh labels:",
        Counter((row["scenario"], row["label"]) for row in fresh_results),
    )


def judge(args: argparse.Namespace) -> None:
    _, definitions, candidates, fresh = _judge_inputs(args)
    if args.finalize_only:
        _finalize_judgments(args, candidates=candidates, fresh=fresh)
        return

    local = LocalJudge(
        model_path=args.judge_model,
        batch_size=args.judge_batch_size,
        max_new_tokens=24,
    )

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    variants = (0, 1) if args.variant < 0 else (args.variant,)
    if args.scope in {"all", "candidate"}:
        for variant in variants:
            _resumable_judge_pass(
                judge=local,
                prompts=[
                    _candidate_prompt(
                        row,
                        definitions[row["scenario"]],
                        variant=variant,
                    )
                    for row in candidates
                ],
                output_path=work_dir / f"candidate-pass-{variant}.jsonl",
                max_input_tokens=args.candidate_max_input_tokens,
                progress_label=f"candidate-pass-{variant}",
            )
    if args.scope in {"all", "fresh"}:
        for variant in variants:
            _resumable_judge_pass(
                judge=local,
                prompts=[
                    _fresh_prompt(
                        row,
                        definitions[row["scenario"]],
                        variant=variant,
                    )
                    for row in fresh
                ],
                output_path=work_dir / f"fresh-pass-{variant}.jsonl",
                max_input_tokens=args.fresh_max_input_tokens,
                progress_label=f"fresh-pass-{variant}",
            )
    expected = {
        work_dir / f"{scope}-pass-{variant}.jsonl"
        for scope in ("candidate", "fresh")
        for variant in (0, 1)
    }
    if all(path.exists() for path in expected):
        try:
            _finalize_judgments(args, candidates=candidates, fresh=fresh)
        except ValueError:
            print("All pass files exist but at least one remains incomplete.")
    else:
        print("Requested judge pass complete; run --finalize-only after all four passes.")


def _candidate_feature_ids(
    candidate_pool: dict[str, Any],
) -> dict[str, list[int]]:
    return {
        method: sorted(
            {
                int(row["feature_id"])
                for scenario in candidate_pool["pools"].values()
                for row in scenario[method]
            }
        )
        for method in METHODS
    }


def extract(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.fresh_pool))
    candidate_pool = json.loads(
        Path(args.candidate_pool).read_text(encoding="utf-8")
    )
    feature_ids = _candidate_feature_ids(candidate_pool)
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection=args.checkpoint_selection,
        modes=METHODS,
    )
    device = torch.device(args.device)
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
        offload_embeddings_to_cpu=True,
    )
    encoders = {
        method: FrozenEncoder(
            Path(sae_set["modes"][method]["checkpoint_path"]),
            device,
        )
        for method in METHODS
    }
    sequences = [
        extractor.tokenizer(
            str(row["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        ).input_ids
        for row in rows
    ]
    order = sorted(range(len(rows)), key=lambda index: len(sequences[index]))
    output = {
        f"{method}_feature_ids": np.asarray(ids, dtype=np.int64)
        for method, ids in feature_ids.items()
    }
    for method, ids in feature_ids.items():
        output[f"{method}_mean"] = np.zeros(
            (len(rows), len(ids)),
            dtype=np.float16,
        )
        if method in {"token", "temporal"}:
            output[f"{method}_max"] = np.zeros(
                (len(rows), len(ids)),
                dtype=np.float16,
            )

    cursor = 0
    completed = 0
    try:
        while cursor < len(order):
            batch: list[int] = []
            maximum = 0
            while cursor < len(order) and len(batch) < args.batch_size:
                index = order[cursor]
                proposed = max(maximum, len(sequences[index])) * (
                    len(batch) + 1
                )
                if batch and proposed > args.token_budget:
                    break
                batch.append(index)
                maximum = max(maximum, len(sequences[index]))
                cursor += 1
            layer_batch = extractor.forward_ids(
                [sequences[index] for index in batch]
            )
            means = layer_batch.means()
            for method in ("mean", "cross"):
                values = encoders[method].selected_activations(
                    means,
                    feature_ids[method],
                )
                output[f"{method}_mean"][batch] = (
                    values.float().cpu().numpy().astype(np.float16)
                )
            flat_hidden = layer_batch.hidden.reshape(
                -1,
                layer_batch.hidden.shape[-1],
            )
            mask = layer_batch.mask.bool()
            for method in ("token", "temporal"):
                values = encoders[method].selected_activations(
                    flat_hidden,
                    feature_ids[method],
                ).reshape(
                    layer_batch.hidden.shape[0],
                    layer_batch.hidden.shape[1],
                    -1,
                )
                expanded_mask = mask.unsqueeze(-1)
                mean_values = (values * expanded_mask).sum(1) / (
                    expanded_mask.sum(1).clamp_min(1)
                )
                max_values = values.masked_fill(
                    ~expanded_mask,
                    float("-inf"),
                ).max(1).values
                max_values = torch.where(
                    torch.isfinite(max_values),
                    max_values,
                    torch.zeros_like(max_values),
                )
                output[f"{method}_mean"][batch] = (
                    mean_values.float().cpu().numpy().astype(np.float16)
                )
                output[f"{method}_max"][batch] = (
                    max_values.float().cpu().numpy().astype(np.float16)
                )
            completed += len(batch)
            if completed % 160 < len(batch):
                print(f"[extract] {completed}/{len(rows)}", flush=True)
    finally:
        extractor.close()
        del encoders
        torch.cuda.empty_cache()

    output["row_ids"] = np.arange(len(rows), dtype=np.int64)
    np.savez_compressed(args.output, **output)
    manifest = {
        "format": "eval7-fresh-natural-selected-activations-v1",
        "complete": True,
        "rows": len(rows),
        "feature_ids": feature_ids,
        "pooling": {
            "token": ["max_after_threshold", "mean_after_threshold"],
            "temporal": ["max_after_threshold", "mean_after_threshold"],
            "mean": ["thresholded_chunk_mean"],
            "cross": ["thresholded_chunk_mean"],
        },
        "model": args.model,
        "layer": args.layer,
        "checkpoint_selection": args.checkpoint_selection,
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    if np.unique(labels).size < 2 or np.unique(values).size < 2:
        return 0.5
    return float(roc_auc_score(labels, values))


def _document_bootstrap(
    labels: np.ndarray,
    values: np.ndarray,
    documents: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> list[float]:
    unique = np.unique(documents)
    positions = {
        document: np.flatnonzero(documents == document)
        for document in unique
    }
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[document] for document in sampled])
        scores.append(_safe_auc(labels[indices], values[indices]))
    return [
        float(np.quantile(scores, 0.025)),
        float(np.quantile(scores, 0.975)),
    ]


def analyze(args: argparse.Namespace) -> None:
    pool = json.loads(Path(args.candidate_pool).read_text(encoding="utf-8"))
    candidate_judgments = json.loads(
        Path(args.candidate_judgments).read_text(encoding="utf-8")
    )
    fresh_all = _read_jsonl(Path(args.fresh_judgments))
    fresh = [
        {**row, "_activation_row": index}
        for index, row in enumerate(fresh_all)
        if row.get("label") in {0, 1}
    ]
    activations = np.load(args.activations)
    selected: dict[tuple[str, str], dict[str, Any] | None] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_judgments:
        grouped[(row["scenario"], row["method"])].append(row)
    for key, rows in grouped.items():
        accepted = sorted(
            (row for row in rows if row["match"]),
            key=lambda row: (int(row["rank"]), int(row["feature_id"])),
        )
        selected[key] = accepted[0] if accepted else None

    feature_columns = {
        method: {
            int(feature_id): index
            for index, feature_id in enumerate(
                activations[f"{method}_feature_ids"]
            )
        }
        for method in METHODS
    }
    results: dict[str, Any] = {
        "format": "eval7-fresh-natural-feature-ood-v1",
        "complete": True,
        "protocol": {
            **pool["protocol"],
            "fresh_pool": (
                "method-independent relation-cue retrieval from previously "
                "unused held-out Pile documents"
            ),
            "fresh_labeling": (
                "two method/feature/activation-blind judgments; agreement only"
            ),
            "candidate_selection": (
                "first semantically accepted retrieval candidate; no fresh "
                "labels used"
            ),
            "primary_pooling": {
                "token": "max_after_threshold",
                "temporal": "max_after_threshold",
                "mean": "thresholded_chunk_mean",
                "cross": "thresholded_chunk_mean",
            },
            "bootstrap": (
                f"{args.bootstrap_samples} document-level resamples"
            ),
            "robust_gate": (
                "overall AUC lower 95% CI > 0.5 and every evaluable source "
                "AUC > 0.5"
            ),
        },
        "scenarios": {},
        "aggregate": {},
    }
    scenarios = list(pool["definitions"])
    for scenario_index, scenario in enumerate(scenarios):
        labeled_positions = np.asarray(
            [
                index
                for index, row in enumerate(fresh)
                if row["scenario"] == scenario
            ],
            dtype=np.int64,
        )
        activation_positions = np.asarray(
            [
                int(fresh[index]["_activation_row"])
                for index in labeled_positions
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [int(fresh[index]["label"]) for index in labeled_positions],
            dtype=np.int8,
        )
        documents = np.asarray(
            [str(fresh[index]["doc_id"]) for index in labeled_positions]
        )
        sources = np.asarray(
            [str(fresh[index]["source"]) for index in labeled_positions]
        )
        results["scenarios"][scenario] = {
            "rows": int(labeled_positions.size),
            "documents": int(np.unique(documents).size),
            "positives": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "sources": dict(Counter(sources.tolist())),
            "methods": {},
        }
        for method_index, method in enumerate(METHODS):
            matches = [
                row
                for row in grouped[(scenario, method)]
                if row["match"]
            ]
            candidate = selected[(scenario, method)]
            if candidate is None:
                results["scenarios"][scenario]["methods"][method] = {
                    "semantic_discovery": False,
                    "matching_candidates": 0,
                    "feature_id": None,
                    "auc": 0.5,
                    "auc_95ci": [0.5, 0.5],
                    "average_precision": float(labels.mean()),
                    "active_recall": 0.0,
                    "false_positive_rate": 0.0,
                    "source_auc": {},
                    "robust_discovery": False,
                }
                continue
            feature_id = int(candidate["feature_id"])
            column = feature_columns[method][feature_id]
            key = (
                f"{method}_max"
                if method in {"token", "temporal"}
                else f"{method}_mean"
            )
            values = activations[key][
                activation_positions,
                column,
            ].astype(np.float32)
            auc = _safe_auc(labels, values)
            interval = _document_bootstrap(
                labels,
                values,
                documents,
                seed=args.seed
                + scenario_index * 100
                + method_index,
                samples=args.bootstrap_samples,
            )
            source_auc: dict[str, float | None] = {}
            evaluable = []
            for source in sorted(np.unique(sources)):
                mask = sources == source
                if (
                    mask.sum() < args.minimum_source_rows
                    or np.unique(labels[mask]).size < 2
                ):
                    source_auc[source] = None
                    continue
                score = _safe_auc(labels[mask], values[mask])
                source_auc[source] = score
                evaluable.append(score)
            active = values > 0
            positive_documents = np.unique(documents[labels == 1])
            active_positive_documents = np.unique(
                documents[(labels == 1) & active]
            )
            robust = bool(
                interval[0] > 0.5
                and evaluable
                and all(score > 0.5 for score in evaluable)
                and active_positive_documents.size
                >= args.minimum_active_positive_documents
            )
            results["scenarios"][scenario]["methods"][method] = {
                "semantic_discovery": True,
                "matching_candidates": len(matches),
                "feature_id": feature_id,
                "retrieval_rank": int(candidate["rank"]),
                "blind_description": candidate["description"],
                "auc": auc,
                "auc_95ci": interval,
                "average_precision": float(
                    average_precision_score(labels, values)
                ),
                "active_recall": float(active[labels == 1].mean()),
                "false_positive_rate": float(active[labels == 0].mean()),
                "positive_documents": int(positive_documents.size),
                "active_positive_documents": int(
                    active_positive_documents.size
                ),
                "source_auc": source_auc,
                "robust_discovery": robust,
            }

    for method in METHODS:
        rows = [
            results["scenarios"][scenario]["methods"][method]
            for scenario in scenarios
        ]
        results["aggregate"][method] = {
            "semantic_discoveries": sum(
                bool(row["semantic_discovery"]) for row in rows
            ),
            "robust_discoveries": sum(
                bool(row["robust_discovery"]) for row in rows
            ),
            "macro_auc": float(np.mean([row["auc"] for row in rows])),
            "macro_average_precision": float(
                np.mean([row["average_precision"] for row in rows])
            ),
        }
    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["aggregate"], indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-candidates")
    build.add_argument("--evidence-dir", required=True)
    build.add_argument("--level-judgments", required=True)
    build.add_argument("--embedding-model", required=True)
    build.add_argument("--embedding-device", default="cuda:0")
    build.add_argument("--embedding-batch-size", type=int, default=32)
    build.add_argument("--candidate-count", type=int, default=32)
    build.add_argument("--evidence-chunks", type=int, default=10)
    build.add_argument("--output", required=True)
    build.set_defaults(func=build_candidates)

    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument("--candidate-pool", required=True)
    judge_parser.add_argument("--fresh-pool", required=True)
    judge_parser.add_argument("--judge-model", required=True)
    judge_parser.add_argument("--judge-batch-size", type=int, default=16)
    judge_parser.add_argument(
        "--variant",
        type=int,
        choices=(-1, 0, 1),
        default=-1,
        help="-1 runs both prompt variants; 0 or 1 runs one resumable pass.",
    )
    judge_parser.add_argument(
        "--scope",
        choices=("all", "candidate", "fresh"),
        default="all",
    )
    judge_parser.add_argument("--finalize-only", action="store_true")
    judge_parser.add_argument(
        "--candidate-max-input-tokens",
        type=int,
        default=3000,
    )
    judge_parser.add_argument(
        "--fresh-max-input-tokens",
        type=int,
        default=1150,
    )
    judge_parser.add_argument("--work-dir", required=True)
    judge_parser.add_argument("--candidate-judgments", required=True)
    judge_parser.add_argument("--fresh-judgments", required=True)
    judge_parser.set_defaults(func=judge)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--fresh-pool", required=True)
    extract_parser.add_argument("--candidate-pool", required=True)
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--sae-root", required=True)
    extract_parser.add_argument("--layer", type=int, default=21)
    extract_parser.add_argument(
        "--checkpoint-selection",
        choices=("best", "final"),
        default="best",
    )
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--model-dtype", default="bfloat16")
    extract_parser.add_argument("--attn-implementation", default="sdpa")
    extract_parser.add_argument("--batch-size", type=int, default=32)
    extract_parser.add_argument("--token-budget", type=int, default=8192)
    extract_parser.add_argument("--max-length", type=int, default=512)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--manifest", required=True)
    extract_parser.set_defaults(func=extract)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--candidate-pool", required=True)
    analyze_parser.add_argument("--candidate-judgments", required=True)
    analyze_parser.add_argument("--fresh-judgments", required=True)
    analyze_parser.add_argument("--activations", required=True)
    analyze_parser.add_argument("--bootstrap-samples", type=int, default=5000)
    analyze_parser.add_argument("--minimum-source-rows", type=int, default=20)
    analyze_parser.add_argument(
        "--minimum-active-positive-documents",
        type=int,
        default=5,
    )
    analyze_parser.add_argument("--seed", type=int, default=20260825)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.set_defaults(func=analyze)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
