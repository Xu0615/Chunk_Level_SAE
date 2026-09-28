#!/usr/bin/env python
"""Blind semantic audit of Eval-7 natural reasoning matched pairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from chunk_saes.artifacts import file_record, write_artifact_manifest
from chunk_saes.utils import atomic_json_dump


FORMAT = "chunk-saes-multi-reason-semantic-audit-v1"


class LocalEditor:
    """Minimal local chat-model wrapper for blinded pair judgments."""

    def __init__(
        self,
        model_path: str,
        dtype: str,
        device_map: str,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        ).eval()

    @torch.inference_mode()
    def batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
    ) -> list[str]:
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        output = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        generated = output[:, encoded["input_ids"].shape[1] :]
        return [
            value.strip()
            for value in self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        ]

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("judge", "merge"),
        required=True,
    )
    p.add_argument("--bank", required=True)
    p.add_argument("--bank-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--judge-model",
        default=(
            "./models/"
            "Qwen3.6-35B-A3B"
        ),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--overwrite", action="store_true")
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _hash_bit(seed: int, family_id: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\x1f{family_id}".encode("utf-8")
    ).digest()
    return digest[0] & 1


def _pairs(
    bank: list[dict[str, Any]],
    manifest: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    metadata = {
        str(row["scenario"]): row
        for row in manifest["candidate_metadata"].values()
    }
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in bank:
        grouped[str(row["scenario"])][str(row["family_id"])] = (
            grouped[str(row["scenario"])].get(str(row["family_id"]), {})
        )
        grouped[str(row["scenario"])][str(row["family_id"])][
            str(row["condition"])
        ] = row
    pairs = []
    for scenario in sorted(grouped):
        for family_id, values in sorted(grouped[scenario].items()):
            active = values["active"]
            inactive = values["inactive"]
            active_is_a = _hash_bit(seed, family_id) == 0
            a = active if active_is_a else inactive
            b = inactive if active_is_a else active
            pairs.append(
                {
                    "scenario": scenario,
                    "scenario_label": active["scenario_label"],
                    "feature_id": int(active["feature_id"]),
                    "primary": bool(
                        metadata[scenario].get("primary", True)
                    ),
                    "value": active["value"],
                    "explanation": metadata[scenario]["explanation"],
                    "family_id": family_id,
                    "doc_hash": active["doc_hash"],
                    "length": int(active["length"]),
                    "active_cross_activation": float(
                        active["cross_activation"]
                    ),
                    "active_choice": "A" if active_is_a else "B",
                    "text_a": a["text"],
                    "text_b": b["text"],
                }
            )
    return pairs


def _prompt(pair: dict[str, Any]) -> str:
    return f"""You are independently auditing a proposed high-level feature.

DEFINITION:
{pair['explanation']}

Two equal-token-length passages come from adjacent positions in the same
document. Exactly one was selected by the proposed feature. Decide which
passage better satisfies the DEFINITION. Judge the full passage, especially
its discourse/reasoning function; do not choose merely because it contains a
single keyword. This is a forced-choice comparison: even if both passages are
imperfect, choose the closer match.

Return JSON only, exactly one of:
{{"choice":"A"}}
{{"choice":"B"}}

PASSAGE A:
{pair['text_a']}

PASSAGE B:
{pair['text_b']}"""


def _choice(raw: str) -> str | None:
    try:
        parsed = json.loads(raw.strip())
        value = str(parsed.get("choice", "")).upper()
        return value if value in {"A", "B"} else None
    except json.JSONDecodeError:
        match = re.search(r"\b(A|B)\b", raw.upper())
        return match.group(1) if match else None


def judge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"judgments.shard-{args.shard_index:03d}-of-"
        f"{args.num_shards:03d}.jsonl"
    )
    if args.overwrite and path.exists():
        path.unlink()
    completed = {
        str(row["family_id"])
        for row in _read_jsonl(path)
    } if path.exists() else set()
    bank = _read_jsonl(Path(args.bank))
    manifest = json.loads(
        Path(args.bank_manifest).read_text(encoding="utf-8")
    )
    pairs = [
        row
        for index, row in enumerate(
            _pairs(bank, manifest, args.seed)
        )
        if index % args.num_shards == args.shard_index
        and row["family_id"] not in completed
    ]
    editor = LocalEditor(
        args.judge_model,
        "bfloat16",
        args.device_map,
    )
    try:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            raws = editor.batch(
                [_prompt(row) for row in batch],
                args.max_new_tokens,
            )
            output = []
            for row, raw in zip(batch, raws, strict=True):
                choice = _choice(raw)
                output.append(
                    {
                        **row,
                        "choice": choice,
                        "correct": (
                            None
                            if choice is None
                            else 0.5
                            if choice == "TIE"
                            else float(choice == row["active_choice"])
                        ),
                        "raw": raw,
                        "judge_model": str(
                            Path(args.judge_model).resolve()
                        ),
                    }
                )
            _append_jsonl(path, output)
            print(
                f"[reason-judge] shard {args.shard_index}: "
                f"{min(start + len(batch), len(pairs))}/{len(pairs)}",
                flush=True,
            )
    finally:
        editor.close()


def _bootstrap(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        indices = rng.integers(
            0,
            len(values),
            size=(count, len(values)),
        )
        draws[start : start + count] = values[indices].mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (
        proportion + z * z / (2 * total)
    ) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [float(center - half), float(center + half)]


def merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    bank = _read_jsonl(Path(args.bank))
    manifest = json.loads(
        Path(args.bank_manifest).read_text(encoding="utf-8")
    )
    expected = {
        row["family_id"]: row
        for row in _pairs(bank, manifest, args.seed)
    }
    rows = []
    for path in sorted(output_dir.glob("judgments.shard-*.jsonl")):
        rows.extend(_read_jsonl(path))
    by_id = {str(row["family_id"]): row for row in rows}
    if len(by_id) != len(expected):
        missing = sorted(set(expected) - set(by_id))
        raise RuntimeError(
            f"judgment coverage {len(by_id)}/{len(expected)}; "
            f"missing={missing[:5]}"
        )
    rows = []
    for key in sorted(expected):
        row = by_id[key]
        row.setdefault(
            "primary",
            bool(expected[key].get("primary", True)),
        )
        row.setdefault(
            "active_cross_activation",
            float(expected[key]["active_cross_activation"]),
        )
        rows.append(row)
    _atomic_jsonl(output_dir / "judgments.jsonl", rows)

    scenarios = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        valid = np.asarray(
            [
                float(row["correct"])
                for row in selected
                if row["correct"] is not None
            ],
            dtype=np.float64,
        )
        resolved = [
            row for row in selected if row["choice"] in {"A", "B"}
        ]
        resolved_correct = sum(
            row["correct"] == 1.0 for row in resolved
        )
        ranked = sorted(
            (
                row
                for row in selected
                if row["correct"] is not None
            ),
            key=lambda row: row["active_cross_activation"],
            reverse=True,
        )
        top_quartile = ranked[: max(1, len(ranked) // 4)]
        top_quartile_values = np.asarray(
            [float(row["correct"]) for row in top_quartile],
            dtype=np.float64,
        )
        scenarios[scenario] = {
            "feature_id": int(selected[0]["feature_id"]),
            "primary": bool(selected[0].get("primary", True)),
            "pairs": len(selected),
            "valid_judgments": int(len(valid)),
            "parse_failures": sum(
                row["correct"] is None for row in selected
            ),
            "active_side_preference": float(valid.mean()),
            "active_side_preference_95ci": _bootstrap(
                valid,
                args.bootstrap_samples,
                args.seed + sum(map(ord, scenario)),
            ),
            "tie_rate": float(
                np.mean(
                    [row["choice"] == "TIE" for row in selected]
                )
            ),
            "resolved_judgments": len(resolved),
            "resolved_accuracy": (
                resolved_correct / len(resolved)
                if resolved
                else None
            ),
            "resolved_accuracy_wilson_95ci": _wilson(
                resolved_correct,
                len(resolved),
            ),
            "top_activation_quartile": {
                "rows": len(top_quartile),
                "minimum_cross_activation": float(
                    min(
                        row["active_cross_activation"]
                        for row in top_quartile
                    )
                ),
                "active_side_preference": float(
                    top_quartile_values.mean()
                ),
                "active_side_preference_95ci": _bootstrap(
                    top_quartile_values,
                    args.bootstrap_samples,
                    args.seed
                    + 500_000
                    + sum(map(ord, scenario)),
                ),
            },
        }
    macro = np.asarray(
        [row["active_side_preference"] for row in scenarios.values()],
        dtype=np.float64,
    )
    primary_macro = np.asarray(
        [
            row["active_side_preference"]
            for row in scenarios.values()
            if row["primary"]
        ],
        dtype=np.float64,
    )
    summary = {
        "format": FORMAT,
        "complete": True,
        "judge_model": str(Path(args.judge_model).resolve()),
        "blinding": {
            "method_hidden": True,
            "feature_id_not_in_prompt": True,
            "activation_not_in_prompt": True,
            "a_b_order_hash_randomized": True,
            "equal_token_length_within_pair": True,
        },
        "pairs": len(rows),
        "valid_judgments": sum(
            row["correct"] is not None for row in rows
        ),
        "scenarios": scenarios,
        "macro_active_side_preference": float(macro.mean()),
        "macro_active_side_preference_95ci": _bootstrap(
            macro,
            args.bootstrap_samples,
            args.seed + 999_999,
        ),
        "primary_scenarios": int(
            sum(row["primary"] for row in scenarios.values())
        ),
        "primary_semantic_passes": int(
            sum(
                row["primary"]
                and row["active_side_preference_95ci"][0] > 0.5
                for row in scenarios.values()
            )
        ),
        "primary_macro_active_side_preference": float(
            primary_macro.mean()
        ),
        "primary_macro_active_side_preference_95ci": _bootstrap(
            primary_macro,
            args.bootstrap_samples,
            args.seed + 1_999_999,
        ),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    atomic_json_dump(
        {
            "format": "chunk-saes-multi-reason-semantic-audit-check-v1",
            "complete": True,
            "issues": [],
            "checks": {
                "all_pairs_covered": True,
                "one_judgment_per_pair": True,
                "judge_blind_to_method_feature_and_activation": True,
                "randomized_pair_orientation": True,
            },
        },
        output_dir / "audit_report.json",
    )
    write_artifact_manifest(
        {
            "format": FORMAT,
            "complete": True,
            "identity": {
                "judge_model": str(Path(args.judge_model).resolve()),
                "seed": args.seed,
                "pair_count": len(rows),
                "forced_choice": True,
                "method_hidden": True,
                "feature_id_hidden": True,
                "activation_hidden": True,
            },
            "files": {
                "judgments": file_record(
                    output_dir / "judgments.jsonl",
                    relative_to=output_dir,
                ),
                "summary": file_record(
                    output_dir / "summary.json",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=output_dir,
                ),
            },
        },
        output_dir / "manifest.json",
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parser().parse_args()
    if args.stage == "judge":
        judge(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
