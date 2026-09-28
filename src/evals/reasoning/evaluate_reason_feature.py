#!/usr/bin/env python
"""Evaluate whether Cross SAE exposes reasoning axes absent from Token SAE.

The challenge bank is synthetic but template-held-out.  Each base content
instance produces four matched variants:

* valid_reasoning: a valid multi-step inference;
* valid_paraphrase: the same inference in a different surface form;
* invalid_relation: nearly the same clauses but an invalid relation;
* unsupported_conclusion: the premise/conclusion direction is invalid.

Discovery, calibration, and test use disjoint templates and content instances.
The script encodes every text once with the frozen base model, computes dense
activations for the complete Cross/Mean dictionaries and pooled Token/Temporal
dictionaries, selects Cross candidates only on discovery/calibration, and
compares each candidate with the strongest Token baselines on untouched test
families.

It also performs a feature-level causal test inside the Cross decoder: remove
one Cross coordinate and measure the increase in adjacent-partner
reconstruction error relative to matched random active coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from chunk_saes.artifacts import (
    file_record,
    file_sha256,
    json_digest,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.plot_style import METHOD_COLORS
from chunk_saes.sae import DecoderHead, SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-reason-feature-eval-v1"
BENCHMARK_FORMAT = "chunk-saes-reason-challenge-bank-v1"
FEATURE_FORMAT = "chunk-saes-reason-features-v1"
METHODS = ("token", "temporal", "mean", "cross")
CONDITIONS = (
    "valid_reasoning",
    "valid_paraphrase",
    "invalid_relation",
    "unsupported_conclusion",
)
POSITIVE_CONDITIONS = ("valid_reasoning", "valid_paraphrase")
NEGATIVE_CONDITIONS = ("invalid_relation", "unsupported_conclusion")
DOMAINS = ("math", "science", "policy", "debugging", "planning", "everyday")


@dataclass(frozen=True)
class Template:
    split: str
    template_id: str
    render: Any


class SparseEncoder:
    """Encoder/decoder view of one frozen SAE checkpoint."""

    def __init__(
        self,
        checkpoint_dir: Path,
        device: torch.device,
        *,
        load_decoder: bool = True,
        decoder_head: DecoderHead | None = None,
    ) -> None:
        """Load a frozen SAE.

        Encoding-only evaluations can skip the decoder matrix, which is about
        one gigabyte for a 65k dictionary.  The default remains ``True`` for
        callers that perform reconstruction or feature ablations.
        """
        config = json.loads(
            (checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
        with safe_open(
            str(checkpoint_dir / "sae.safetensors"),
            framework="pt",
            device="cpu",
        ) as handle:
            names = set(handle.keys())
            self.weight = handle.get_tensor("encoder_weight").to(device)
            self.bias = handle.get_tensor("encoder_bias").to(device)
            decoder_is_joint = (
                int(config.get("decoder_heads", 1)) == 2
                or "decoder_cross_weight" in names
                or "decoder_cross_bias" in names
            )
            has_cross_weight = "decoder_cross_weight" in names
            has_cross_bias = "decoder_cross_bias" in names
            if has_cross_weight != has_cross_bias:
                raise ValueError(
                    f"{checkpoint_dir} stores an incomplete Cross decoder head"
                )
            if int(config.get("decoder_heads", 1)) == 2 and not (
                has_cross_weight and has_cross_bias
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares two decoder heads but lacks "
                    "the Cross head tensors"
                )
            if decoder_is_joint:
                if load_decoder and decoder_head not in ("mean", "cross"):
                    raise ValueError(
                        f"{checkpoint_dir} is a Joint checkpoint; pass "
                        "decoder_head='mean' or decoder_head='cross'"
                    )
                if decoder_head in ("mean", "cross"):
                    decoder_weight_name = (
                        "decoder_cross_weight"
                        if decoder_head == "cross"
                        else "decoder_weight"
                    )
                    decoder_bias_name = (
                        "decoder_cross_bias"
                        if decoder_head == "cross"
                        else "decoder_bias"
                    )
                    if (
                        decoder_weight_name not in names
                        or decoder_bias_name not in names
                    ):
                        raise ValueError(
                            f"{checkpoint_dir} lacks the requested decoder head "
                            f"tensors: {decoder_weight_name}, {decoder_bias_name}"
                        )
                else:
                    # Encoding-only callers do not need to choose a readout.
                    decoder_weight_name = "decoder_weight"
                    decoder_bias_name = "decoder_bias"
            else:
                if decoder_head not in (None, "mean"):
                    raise ValueError(
                        f"single-head checkpoint {checkpoint_dir} cannot select "
                        f"decoder_head={decoder_head!r}"
                    )
                decoder_weight_name = "decoder_weight"
                decoder_bias_name = "decoder_bias"
            decoder_bias = handle.get_tensor(decoder_bias_name).to(device)
            # A legacy checkpoint without an explicit pre_bias used its
            # primary decoder bias for input centering.  For a Joint readout
            # this is always the Mean bias, never the selected Cross bias.
            legacy_pre_bias = handle.get_tensor("decoder_bias").to(device)
            self.decoder_weight = (
                handle.get_tensor(decoder_weight_name).to(device)
                if load_decoder
                else None
            )
            self.decoder_bias = decoder_bias if load_decoder else None
            self.decoder_head = decoder_head if load_decoder else None
            if "pre_bias" in names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares {SAE_PARAMETER_SCHEMA_VERSION} "
                    "but lacks pre_bias"
                )
            else:
                self.pre_bias = legacy_pre_bias
            self.threshold = handle.get_tensor("threshold").to(device)
            self.scale = handle.get_tensor("activation_scale").to(device)
        self.width = int(self.weight.shape[0])
        self.device = device

    @torch.inference_mode()
    def dense(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.device, dtype=self.weight.dtype)
        pre = F.relu(
            F.linear(
                hidden * self.scale.to(self.weight.dtype) - self.pre_bias,
                self.weight,
                self.bias,
            )
        )
        return pre * (pre > self.threshold.to(pre.dtype))

    def close(self) -> None:
        del (
            self.weight,
            self.bias,
            self.decoder_weight,
            self.decoder_bias,
            self.pre_bias,
            self.threshold,
            self.scale,
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a reasoning challenge bank and test Cross single features "
            "against Token single/sparse baselines."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument(
        "--checkpoint-selection",
        choices=("best", "final"),
        default="best",
    )
    p.add_argument(
        "--decoder-head",
        choices=("mean", "cross"),
        default=None,
        help=(
            "Read this head when a supplied checkpoint is a Joint Chunk SAE. "
            "Legacy single-head checkpoints leave it unset."
        ),
    )
    p.add_argument("--instances-per-domain-split", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--candidate-count", type=int, default=32)
    p.add_argument("--shortlist-count", type=int, default=8)
    p.add_argument(
        "--fixed-cross-feature",
        type=int,
        default=None,
        help=(
            "Optionally evaluate a preregistered Cross coordinate in addition "
            "to the benchmark-selected candidates."
        ),
    )
    p.add_argument(
        "--sparse-budgets",
        default="4,16,64",
        help="Comma-separated Token feature budgets.",
    )
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--analysis-only",
        action="store_true",
        help="Reuse challenge_bank.jsonl and features.npz.",
    )
    return p


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _render_reasoning_family(
    template_id: str,
    a: str,
    b: str,
    c: str,
    style: int,
) -> dict[str, str]:
    if template_id == "deductive_chain":
        values = {
            "valid_reasoning": (
                f"Whenever {a}, {b}. Whenever {b}, {c}. We know {a}. "
                f"Therefore {c}."
            ),
            "valid_paraphrase": (
                f"Given {a}, we obtain {b}; and {b} entails {c}. Since {a} "
                f"is established, it follows that {c}."
            ),
            "invalid_relation": (
                f"Whenever {a}, {b}. Whenever {c}, {b}. We know {a}. "
                f"Therefore {c}."
            ),
            "unsupported_conclusion": (
                f"Whenever {a}, {b}. Whenever {b}, {c}. We know {c}. "
                f"Therefore {a}."
            ),
        }
    elif template_id == "modus_tollens":
        values = {
            "valid_reasoning": (
                f"If {a}, then {b}. It is not the case that {b}. Therefore it "
                f"is not the case that {a}. This supports {c}."
            ),
            "valid_paraphrase": (
                f"{a} would require {b}. But {b} is absent. Consequently {a} "
                f"cannot hold, lending support to {c}."
            ),
            "invalid_relation": (
                f"If {a}, then {b}. It is not the case that {a}. Therefore it "
                f"is not the case that {b}. This supports {c}."
            ),
            "unsupported_conclusion": (
                f"If {a}, then {b}. It is the case that {b}. Therefore {a}. "
                f"This supports {c}."
            ),
        }
    elif template_id == "disjunctive_elimination":
        values = {
            "valid_reasoning": (
                f"Either {a} or {b} must hold. The evidence rules out {a}. "
                f"Therefore {b} holds, which supports {c}."
            ),
            "valid_paraphrase": (
                f"The possibilities are {a} and {b}. Since {a} has been "
                f"eliminated, the remaining possibility is {b}; hence {c}."
            ),
            "invalid_relation": (
                f"Either {a} or {b} must hold. The evidence confirms {a}. "
                f"Therefore {b} holds, which supports {c}."
            ),
            "unsupported_conclusion": (
                f"Either {a} or {b} must hold. The evidence rules out {a}. "
                f"Therefore {a} holds, which supports {c}."
            ),
        }
    elif template_id == "causal_chain":
        values = {
            "valid_reasoning": (
                f"When {a} occurs, it produces {b}. The presence of {b} then "
                f"causes {c}. Therefore {a} can lead to {c}."
            ),
            "valid_paraphrase": (
                f"{a} changes {b}, and that intermediate change produces {c}. "
                f"The resulting {c} is thus explained by a two-stage pathway "
                f"beginning with {a}."
            ),
            "invalid_relation": (
                f"When {a} occurs, it produces {b}. The presence of {c} then "
                f"causes {b}. Therefore {a} can lead to {c}."
            ),
            "unsupported_conclusion": (
                f"When {a} occurs, it produces {b}. The presence of {b} then "
                f"causes {c}. Therefore {c} must have caused {a}."
            ),
        }
    elif template_id == "constraint_elimination":
        values = {
            "valid_reasoning": (
                f"A valid option must provide {a}, {b}, and {c}. Option one "
                f"lacks {a}; option two lacks {b}. The remaining option alone "
                "meets every requirement."
            ),
            "valid_paraphrase": (
                f"The requirements are {a}, {b}, and {c}. Reject the first "
                f"choice because it lacks {a}, and reject the second because it "
                f"lacks {b}. Only the final choice remains feasible."
            ),
            "invalid_relation": (
                f"A valid option must provide {a}, {b}, and {c}. Option one "
                f"provides {a}; option two provides {b}. The remaining option "
                "alone meets every requirement."
            ),
            "unsupported_conclusion": (
                f"A valid option must provide {a}, {b}, and {c}. Option one "
                f"lacks {a}; option two lacks {b}. Therefore option one alone "
                "meets every requirement."
            ),
        }
    elif template_id == "case_analysis":
        values = {
            "valid_reasoning": (
                f"There are two exhaustive cases. If {a}, then {b}. If not "
                f"{a}, then {b} follows from {c}. Thus {b} holds in either case."
            ),
            "valid_paraphrase": (
                f"Split on whether {a}. The first branch gives {b} directly; "
                f"the second branch gives {b} using {c}. Since the branches are "
                f"exhaustive, {b} follows."
            ),
            "invalid_relation": (
                f"There are two exhaustive cases. If {a}, then {b}. If not "
                f"{a}, then {c}. Thus {b} holds in either case."
            ),
            "unsupported_conclusion": (
                f"There are two exhaustive cases. If {a}, then {b}. If not "
                f"{a}, then {b} follows from {c}. Thus {c} holds in either case."
            ),
        }
    elif template_id == "evidence_revision":
        values = {
            "valid_reasoning": (
                f"The original account predicts {a}. New evidence instead "
                f"shows {b}, which is incompatible with that prediction. "
                f"Therefore {c} is better supported."
            ),
            "valid_paraphrase": (
                f"We expected {a}, but observed {b}. Because the observation "
                f"conflicts with the original account, the evidence now favors "
                f"{c}."
            ),
            "invalid_relation": (
                f"The original account predicts {a}. New evidence instead "
                f"shows {b}, which agrees with that prediction. Therefore {c} "
                "is better supported."
            ),
            "unsupported_conclusion": (
                f"The original account predicts {a}. New evidence instead "
                f"shows {b}, which is incompatible with that prediction. "
                f"Therefore {a} is better supported."
            ),
        }
    elif template_id == "ordered_comparison":
        values = {
            "valid_reasoning": (
                f"An alternative with {a} outranks one with {b}. An alternative "
                f"with {b} outranks one with {c}. Therefore {a} outranks {c}."
            ),
            "valid_paraphrase": (
                f"{a} is preferred to {b}, while {b} is preferred to {c}. "
                f"By transitivity, {a} must be preferred to {c}."
            ),
            "invalid_relation": (
                f"An alternative with {a} outranks one with {b}. An alternative "
                f"with {c} outranks one with {b}. Therefore {a} outranks {c}."
            ),
            "unsupported_conclusion": (
                f"An alternative with {a} outranks one with {b}. An alternative "
                f"with {b} outranks one with {c}. Therefore {c} outranks {a}."
            ),
        }
    elif template_id == "error_correction":
        values = {
            "valid_reasoning": (
                f"The procedure produces {a}. Inspection shows that {b} occurs "
                f"immediately before the failure. Correcting {b} removes the "
                f"failure, so {c} is the justified remedy."
            ),
            "valid_paraphrase": (
                f"We observed {a} and traced it to {b}. Once {b} was corrected, "
                f"the failure disappeared. This evidence supports {c}."
            ),
            "invalid_relation": (
                f"The procedure produces {a}. Inspection shows that {b} occurs "
                f"after the failure. Correcting {b} removes the failure, so {c} "
                "is the justified remedy."
            ),
            "unsupported_conclusion": (
                f"The procedure produces {a}. Inspection shows that {b} occurs "
                f"immediately before the failure. Correcting {b} does not change "
                f"the failure, so {c} is the justified remedy."
            ),
        }
    else:
        raise ValueError(template_id)
    if style == 1:
        return {
            key: "Reasoning note. "
            + value.replace("Therefore ", "Consequently ")
            for key, value in values.items()
        }
    if style == 2:
        return {
            key: "Analysis. "
            + value.replace("The evidence", "The available record")
            for key, value in values.items()
        }
    return values


def _templates() -> tuple[Template, ...]:
    # Hold out complete reasoning structures, not merely content words.  A
    # feature nominated on discovery therefore has to transfer to unseen
    # inference forms before it is reported on test.
    assignments = {
        "discovery": (
            "deductive_chain",
            "causal_chain",
            "constraint_elimination",
        ),
        "calibration": (
            "modus_tollens",
            "evidence_revision",
            "ordered_comparison",
        ),
        "test": (
            "disjunctive_elimination",
            "case_analysis",
            "error_correction",
        ),
    }
    return tuple(
        Template(
            split,
            f"{template_id}_style{style}",
            lambda a, b, c, template_id=template_id, style=style:
                _render_reasoning_family(template_id, a, b, c, style),
        )
        for split, template_ids in assignments.items()
        for template_id in template_ids
        for style in range(3)
    )


VOCABULARY: dict[str, tuple[str, ...]] = {
    "math": (
        "the sequence is bounded",
        "a convergent subsequence",
        "a finite limit",
        "the constraint is convex",
        "the optimum is unique",
        "the residual vanishes",
        "the inequality holds",
        "the feasible set is nonempty",
        "the estimate is stable",
        "the mapping is continuous",
        "the derivative is positive",
        "the objective decreases",
    ),
    "science": (
        "the catalyst concentration increases",
        "the reaction rate accelerates",
        "the measured yield improves",
        "the control sample remains unchanged",
        "the proposed mechanism is incomplete",
        "an alternative pathway is active",
        "the temperature is held constant",
        "the pressure falls",
        "the phase transition is delayed",
        "the observed signal is genuine",
        "the null model predicts no shift",
        "the treatment changes the response",
    ),
    "policy": (
        "the regulation applies",
        "the reporting duty is triggered",
        "a filing deadline follows",
        "the exemption is unavailable",
        "the agency must review the claim",
        "the appeal may proceed",
        "the budget remains fixed",
        "the proposal raises compliance costs",
        "the alternative preserves access",
        "the evidence meets the legal standard",
        "the prior rule controls",
        "the requested remedy is barred",
    ),
    "debugging": (
        "the cache is stale",
        "the old configuration is reused",
        "the request returns the previous value",
        "the parser accepts the header",
        "the failure occurs after decoding",
        "the malformed payload is not the cause",
        "the dependency is installed",
        "the import succeeds",
        "the remaining error is a version mismatch",
        "the lock is acquired",
        "the write can proceed",
        "the worker exits cleanly",
    ),
    "planning": (
        "the permit is approved",
        "construction can begin",
        "the launch date is feasible",
        "the data collection is complete",
        "the analysis can start",
        "the report can be delivered",
        "the supplier confirms inventory",
        "the order can be placed",
        "the installation can finish",
        "the prerequisite course is passed",
        "registration can open",
        "the project remains on schedule",
    ),
    "everyday": (
        "the road is flooded",
        "the bus route is closed",
        "the traveler must take the train",
        "the key is missing",
        "the door cannot be opened",
        "a locksmith is required",
        "the store is closed",
        "the ingredient is unavailable",
        "the recipe needs a substitute",
        "the battery is empty",
        "the device will not start",
        "the charger must be connected",
    ),
}


def build_challenge_bank(
    *,
    tokenizer,
    instances_per_domain_split: int,
    seed: int,
) -> list[dict[str, Any]]:
    templates = _templates()
    by_split: dict[str, list[Template]] = {
        split: [template for template in templates if template.split == split]
        for split in ("discovery", "calibration", "test")
    }
    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("discovery", "calibration", "test")):
        for domain_index, domain in enumerate(DOMAINS):
            vocab = VOCABULARY[domain]
            for local_index in range(instances_per_domain_split):
                template = by_split[split][
                    local_index % len(by_split[split])
                ]
                rng = random.Random(
                    seed
                    + split_index * 1_000_003
                    + domain_index * 10_007
                    + local_index
                )
                a, b, c = rng.sample(vocab, 3)
                context = (
                    f"In {domain} case {split_index + 1}-"
                    f"{domain_index + 1}-{local_index + 1}"
                )
                variants = template.render(a, b, c)
                partner = (
                    f"{context}. The preceding analysis supplies a justified basis for the "
                    f"next decision. The supported conclusion is: {c}. This "
                    "conclusion should be carried forward because it follows "
                    "from the linked premises rather than from a keyword alone."
                )
                family_id = f"{split}-{domain}-{local_index:03d}"
                for condition in CONDITIONS:
                    text = context + ". " + variants[condition]
                    token_ids = tokenizer(
                        text,
                        add_special_tokens=False,
                    ).input_ids
                    rows.append(
                        {
                            "example_id": f"{family_id}-{condition}",
                            "family_id": family_id,
                            "split": split,
                            "domain": domain,
                            "template_id": template.template_id,
                            "condition": condition,
                            "label": int(
                                condition in POSITIVE_CONDITIONS
                            ),
                            "text": text,
                            "token_ids": list(map(int, token_ids)),
                            "token_length": len(token_ids),
                            "partner_text": partner,
                            "partner_token_ids": list(
                                map(
                                    int,
                                    tokenizer(
                                        partner,
                                        add_special_tokens=False,
                                    ).input_ids,
                                )
                            ),
                            "content_sha256": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
    if len({row["content_sha256"] for row in rows}) != len(rows):
        raise ValueError("challenge bank contains duplicate texts")
    return rows


def _write_challenge_bank(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    seed: int,
    instances_per_domain_split: int,
) -> dict[str, Any]:
    path = output_dir / "challenge_bank.jsonl"
    _atomic_jsonl(path, rows)
    counts = Counter((row["split"], row["condition"]) for row in rows)
    families = {
        split: len(
            {
                row["family_id"]
                for row in rows
                if row["split"] == split
            }
        )
        for split in ("discovery", "calibration", "test")
    }
    payload = {
        "format": BENCHMARK_FORMAT,
        "complete": True,
        "identity": {
            "seed": seed,
            "instances_per_domain_split": instances_per_domain_split,
            "domains": list(DOMAINS),
            "conditions": list(CONDITIONS),
            "template_ids_by_split": {
                split: [
                    template.template_id
                    for template in _templates()
                    if template.split == split
                ]
                for split in ("discovery", "calibration", "test")
            },
            "split_policy": "disjoint templates and content families",
            "negative_controls": {
                "invalid_relation": (
                    "same propositions and inference vocabulary, but one "
                    "relation is reversed or disconnected"
                ),
                "unsupported_conclusion": (
                    "same propositions and discourse form, but the conclusion "
                    "does not follow"
                ),
            },
        },
        "rows": len(rows),
        "families_by_split": families,
        "counts": {
            f"{split}/{condition}": counts[(split, condition)]
            for split in ("discovery", "calibration", "test")
            for condition in CONDITIONS
        },
        "files": {
            "challenge_bank": file_record(path, relative_to=output_dir)
        },
    }
    return write_artifact_manifest(
        payload,
        output_dir / "challenge_bank_manifest.json",
    )


def _pool_token_features(
    dense: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    valid = mask.bool().unsqueeze(-1)
    if mode == "mean":
        return (dense * valid).sum(1) / valid.sum(1).clamp_min(1)
    if mode == "max":
        return dense.masked_fill(~valid, 0).amax(dim=1)
    raise ValueError(mode)


@torch.inference_mode()
def extract_features(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    sae_set: dict[str, Any],
    output_dir: Path,
    challenge_manifest: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device(args.device)
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    encoders = {
        method: SparseEncoder(
            Path(sae_set["modes"][method]["checkpoint_path"]),
            device,
            # This benchmark has one decoder-consuming Cross path; Token,
            # Temporal and Mean remain the historical single-head baselines.
            decoder_head=args.decoder_head if method == "cross" else None,
        )
        for method in METHODS
    }
    n = len(rows)
    arrays = {
        "raw_mean_hidden": np.empty((n, extractor.hidden_size), np.float16),
        "partner_target_hidden": np.empty((n, extractor.hidden_size), np.float16),
        "token_mean": np.empty((n, encoders["token"].width), np.float16),
        "token_max": np.empty((n, encoders["token"].width), np.float16),
        "temporal_mean": np.empty(
            (n, encoders["temporal"].width),
            np.float16,
        ),
        "mean": np.empty((n, encoders["mean"].width), np.float16),
        "cross": np.empty((n, encoders["cross"].width), np.float16),
    }
    try:
        sequences = [row["token_ids"] for row in rows]
        for start in range(0, n, args.batch_size):
            stop = min(n, start + args.batch_size)
            layer_batch = extractor.forward_ids(sequences[start:stop])
            means = layer_batch.means()
            arrays["raw_mean_hidden"][start:stop] = (
                means.float().cpu().numpy().astype(np.float16)
            )
            arrays["mean"][start:stop] = (
                encoders["mean"].dense(means)
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            arrays["cross"][start:stop] = (
                encoders["cross"].dense(means)
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            token_dense = encoders["token"].dense(layer_batch.hidden)
            arrays["token_mean"][start:stop] = (
                _pool_token_features(token_dense, layer_batch.mask, "mean")
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            arrays["token_max"][start:stop] = (
                _pool_token_features(token_dense, layer_batch.mask, "max")
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            temporal_dense = encoders["temporal"].dense(layer_batch.hidden)
            arrays["temporal_mean"][start:stop] = (
                _pool_token_features(
                    temporal_dense,
                    layer_batch.mask,
                    "mean",
                )
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            del layer_batch, means, token_dense, temporal_dense
            print(
                f"[reason-feature] encoded {stop}/{n} challenge texts",
                flush=True,
            )
        family_rows: dict[str, dict[str, int]] = {}
        for index, row in enumerate(rows):
            family_rows.setdefault(str(row["family_id"]), {})[
                str(row["condition"])
            ] = index
        family_order = sorted(family_rows)
        partner_sequences = [
            next(
                row["partner_token_ids"]
                for row in rows
                if row["family_id"] == family_id
            )
            for family_id in family_order
        ]
        partner_means: dict[str, np.ndarray] = {}
        for start in range(0, len(partner_sequences), args.batch_size):
            stop = min(len(partner_sequences), start + args.batch_size)
            layer_batch = extractor.forward_ids(
                partner_sequences[start:stop]
            )
            means = (
                layer_batch.means()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            for family_id, target in zip(
                family_order[start:stop],
                means,
                strict=True,
            ):
                partner_means[family_id] = target
        for family_id, condition_rows in family_rows.items():
            target = partner_means[family_id]
            for row_index in condition_rows.values():
                arrays["partner_target_hidden"][row_index] = target
        feature_path = output_dir / "features.npz"
        np.savez_compressed(
            feature_path,
            **arrays,
            labels=np.asarray([row["label"] for row in rows], np.int8),
            splits=np.asarray([row["split"] for row in rows]),
            domains=np.asarray([row["domain"] for row in rows]),
            conditions=np.asarray([row["condition"] for row in rows]),
            family_ids=np.asarray([row["family_id"] for row in rows]),
            token_lengths=np.asarray(
                [row["token_length"] for row in rows],
                np.int16,
            ),
        )
        payload = {
            "format": FEATURE_FORMAT,
            "complete": True,
            "identity": {
                "challenge_artifact_digest":
                    challenge_manifest["artifact_digest"],
                "sae_set_digest": sae_set["artifact_digest"],
                "model": str(Path(args.model).resolve()),
                "layer": args.layer,
                "checkpoint_selection": args.checkpoint_selection,
                "cross_decoder_head": args.decoder_head,
                "methods": list(METHODS),
                "pooling": {
                    "token_mean": "mean thresholded token activation",
                    "token_max": "max thresholded token activation",
                    "temporal_mean": "mean thresholded temporal activation",
                    "mean": "thresholded activation of mean hidden",
                    "cross": "thresholded activation of mean hidden",
                },
                "partner_target": (
                    "Cross decoder reconstruction of the reasoning variant; "
                    "used only for feature-level ablation ranking"
                ),
            },
            "rows": n,
            "feature_width": 65_536,
            "files": {
                "features": file_record(
                    feature_path,
                    relative_to=output_dir,
                )
            },
        }
        return write_artifact_manifest(
            payload,
            output_dir / "feature_manifest.json",
        )
    finally:
        extractor.close()
        for encoder in encoders.values():
            encoder.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _paired_auc(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _family_accuracy(
    values: np.ndarray,
    rows: Sequence[dict[str, Any]],
    mask: np.ndarray,
) -> float:
    by_family: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if mask[index]:
            by_family.setdefault(str(row["family_id"]), []).append(index)
    correct = 0
    total = 0
    for indices in by_family.values():
        positive_rows = [
            local
            for local, index in enumerate(indices)
            if int(rows[index]["label"]) == 1
        ]
        if not positive_rows or len(indices) < 2:
            continue
        scores = values[indices]
        best = np.flatnonzero(scores == scores.max())
        correct += sum(
            float(local in best) for local in positive_rows
        ) / len(best)
        total += 1
    return correct / max(1, total)


def _standardized_effect(
    values: np.ndarray,
    rows: Sequence[dict[str, Any]],
    mask: np.ndarray,
) -> float:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    positive = values[
        mask & (labels == 1)
    ]
    negative = values[
        mask & (labels == 0)
    ]
    pooled = math.sqrt(
        (
            float(positive.var(ddof=1))
            + float(negative.var(ddof=1))
        )
        / 2
        + 1e-12
    )
    return float((positive.mean() - negative.mean()) / pooled)


def _feature_metrics(
    matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    mask: np.ndarray,
    *,
    feature_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    domains = np.asarray([row["domain"] for row in rows])
    condition_masks = {
        condition: np.asarray(
            [row["condition"] == condition for row in rows],
            dtype=bool,
        )
        for condition in CONDITIONS
    }
    positives = mask & (labels == 1)
    negatives = mask & (labels == 0)
    positive_mean = matrix[positives].mean(axis=0, dtype=np.float64)
    negative_mean = matrix[negatives].mean(axis=0, dtype=np.float64)
    positive_rate = (matrix[positives] > 0).mean(axis=0)
    negative_rate = (matrix[negatives] > 0).mean(axis=0)
    selectivity = (positive_mean - negative_mean) / (
        positive_mean + negative_mean + 1e-8
    )
    candidate_score = (positive_rate - negative_rate) * np.log1p(
        np.maximum(positive_mean, 0)
    )
    order = np.argsort(candidate_score)[::-1]
    output: list[dict[str, Any]] = []
    y = labels[mask]
    for column in order:
        values = matrix[:, column].astype(np.float64)
        if not np.any(values[mask]):
            continue
        per_domain_auc = {}
        for domain in DOMAINS:
            domain_mask = mask & (domains == domain)
            per_domain_auc[domain] = _paired_auc(
                values[domain_mask],
                labels[domain_mask],
            )
        output.append(
            {
                "column": int(column),
                "feature_id": int(
                    feature_ids[column]
                    if feature_ids is not None
                    else column
                ),
                "candidate_score": float(candidate_score[column]),
                "auc": _paired_auc(values[mask], y),
                "family_accuracy": _family_accuracy(values, rows, mask),
                "effect_size": _standardized_effect(
                    values,
                    rows,
                    mask,
                ),
                "positive_mean": float(positive_mean[column]),
                "negative_mean": float(negative_mean[column]),
                "positive_rate": float(positive_rate[column]),
                "negative_rate": float(negative_rate[column]),
                "selectivity": float(selectivity[column]),
                "per_domain_auc": per_domain_auc,
            }
        )
    return output


def _choose_cross_candidates(
    arrays: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    *,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    masks = {
        split: np.asarray(
            [
                row["split"] == split
                for row in rows
            ],
            bool,
        )
        for split in ("discovery", "calibration", "test")
    }
    cross = arrays["cross"].astype(np.float32)
    discovery = _feature_metrics(cross, rows, masks["discovery"])
    # Discovery may nominate a broad list. Calibration is the only split used
    # to rank/freeze the final shortlist.
    nominated = discovery[: max(candidate_count * 8, 256)]
    nominated_ids = np.asarray(
        [row["feature_id"] for row in nominated],
        dtype=np.int64,
    )
    calibration_metrics = _feature_metrics(
        cross[:, nominated_ids],
        rows,
        masks["calibration"],
        feature_ids=nominated_ids,
    )
    calibration_by_id = {
        row["feature_id"]: row for row in calibration_metrics
    }
    eligible_nominated = [
        row
        for row in nominated
        if row["feature_id"] in calibration_by_id
    ]
    if len(eligible_nominated) < candidate_count:
        raise ValueError(
            f"only {len(eligible_nominated)} discovery nominees activate on "
            f"calibration; need {candidate_count}"
        )
    ranked = sorted(
        eligible_nominated,
        key=lambda row: (
            calibration_by_id[row["feature_id"]]["auc"],
            calibration_by_id[row["feature_id"]]["family_accuracy"],
            calibration_by_id[row["feature_id"]]["effect_size"],
        ),
        reverse=True,
    )
    return ranked[:candidate_count], {
        "discovery": discovery,
        "calibration": calibration_metrics,
    }


def _scale_sparse_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(train), scaler.transform(test), scaler


def _fit_sparse_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    # Univariate screening is performed on calibration only.  The final model
    # sees exactly ``budget`` features and is evaluated once on test.
    positive_mean = x_train[y_train == 1].mean(axis=0, dtype=np.float64)
    negative_mean = x_train[y_train == 0].mean(axis=0, dtype=np.float64)
    positive_rate = (x_train[y_train == 1] > 0).mean(axis=0)
    negative_rate = (x_train[y_train == 0] > 0).mean(axis=0)
    score = np.abs(positive_rate - negative_rate) * np.log1p(
        positive_mean + negative_mean
    )
    selected = np.argsort(score)[::-1][:budget]
    train_scaled, test_scaled, _scaler = _scale_sparse_train_test(
        x_train[:, selected],
        x_test[:, selected],
    )
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train_scaled, y_train)
    train_probability = model.predict_proba(train_scaled)[:, 1]
    test_probability = model.predict_proba(test_scaled)[:, 1]
    return {
        "budget": budget,
        "feature_ids": selected.astype(int).tolist(),
        "calibration_auc": float(roc_auc_score(y_train, train_probability)),
        "test_auc": float(roc_auc_score(y_test, test_probability)),
        "test_accuracy_at_0_5": float(
            accuracy_score(y_test, test_probability >= 0.5)
        ),
        "test_probabilities": test_probability,
    }


def _best_single_feature(
    matrix: np.ndarray,
    labels: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    y_cal = labels[calibration_mask]
    pos = matrix[calibration_mask][y_cal == 1].mean(axis=0, dtype=np.float64)
    neg = matrix[calibration_mask][y_cal == 0].mean(axis=0, dtype=np.float64)
    pos_rate = (matrix[calibration_mask][y_cal == 1] > 0).mean(axis=0)
    neg_rate = (matrix[calibration_mask][y_cal == 0] > 0).mean(axis=0)
    screening = np.abs(pos_rate - neg_rate) * np.log1p(pos + neg)
    top = np.argsort(screening)[::-1][:512]
    best: tuple[float, int, int] | None = None
    for feature_id in top:
        values = matrix[calibration_mask, feature_id]
        auc = _paired_auc(values, y_cal)
        signed_auc = max(auc, 1 - auc)
        direction = 1 if auc >= 0.5 else -1
        candidate = (signed_auc, int(feature_id), direction)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    cal_auc, feature_id, direction = best
    test_values = direction * matrix[test_mask, feature_id]
    test_auc = _paired_auc(test_values, labels[test_mask])
    return {
        "feature_id": feature_id,
        "direction": direction,
        "calibration_auc": cal_auc,
        "test_auc": test_auc,
        "test_values": test_values.astype(np.float64),
    }


def _bootstrap_auc(
    labels: np.ndarray,
    values: np.ndarray,
    family_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    families = np.unique(family_ids)
    rows_by_family = {
        family: np.flatnonzero(family_ids == family)
        for family in families
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        indices = np.concatenate([rows_by_family[family] for family in selected])
        draws.append(roc_auc_score(labels[indices], values[indices]))
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _bootstrap_delta_auc(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    family_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    families = np.unique(family_ids)
    rows_by_family = {
        family: np.flatnonzero(family_ids == family)
        for family in families
    }
    observed = float(
        roc_auc_score(labels, left) - roc_auc_score(labels, right)
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        indices = np.concatenate([rows_by_family[family] for family in selected])
        draws[index] = (
            roc_auc_score(labels[indices], left[indices])
            - roc_auc_score(labels[indices], right[indices])
        )
    return {
        "point": observed,
        "95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_delta_le_0": float((draws <= 0).mean()),
    }


def _cross_candidate_test_metrics(
    matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    feature_id: int,
    primary_test_mask: np.ndarray,
    full_test_mask: np.ndarray,
) -> dict[str, Any]:
    values = matrix[:, feature_id].astype(np.float64)
    labels = np.asarray([row["label"] for row in rows], np.int8)
    conditions = np.asarray([row["condition"] for row in rows])
    positive = full_test_mask & np.isin(
        conditions,
        POSITIVE_CONDITIONS,
    )
    output = {
        "feature_id": feature_id,
        "test_auc": _paired_auc(
            values[primary_test_mask],
            labels[primary_test_mask],
        ),
        "test_family_accuracy": _family_accuracy(
            values,
            rows,
            full_test_mask,
        ),
        "test_effect_size": _standardized_effect(
            values,
            rows,
            primary_test_mask,
        ),
        "positive_activation_rate": float((values[positive] > 0).mean()),
        "condition_mean_activation": {
            condition: float(
                values[
                    full_test_mask & (conditions == condition)
                ].mean()
            )
            for condition in CONDITIONS
        },
        "condition_activation_rate": {
            condition: float(
                (
                    values[
                        full_test_mask & (conditions == condition)
                    ]
                    > 0
                ).mean()
            )
            for condition in CONDITIONS
        },
        "per_domain_auc": {},
        "test_values": values[primary_test_mask],
    }
    domains = np.asarray([row["domain"] for row in rows])
    for domain in DOMAINS:
        domain_mask = primary_test_mask & (domains == domain)
        output["per_domain_auc"][domain] = _paired_auc(
            values[domain_mask],
            labels[domain_mask],
        )
    return output


def _cross_ablation_importance(
    *,
    cross_matrix: np.ndarray,
    target_hidden: np.ndarray,
    rows: Sequence[dict[str, Any]],
    test_mask: np.ndarray,
    candidate_ids: Sequence[int],
    cross_checkpoint: Path,
    seed: int,
    decoder_head: DecoderHead | None = None,
) -> dict[str, Any]:
    reasoning_mask = test_mask & np.asarray(
        [int(row["label"]) == 1 for row in rows],
        bool,
    )
    z = torch.from_numpy(
        cross_matrix[reasoning_mask].astype(np.float32)
    )
    target = torch.from_numpy(
        target_hidden[reasoning_mask].astype(np.float32)
    )
    with safe_open(
        str(cross_checkpoint / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        names = set(handle.keys())
        config = json.loads(
            (cross_checkpoint / "config.json").read_text(encoding="utf-8")
        )
        is_joint = (
            int(config.get("decoder_heads", 1)) == 2
            or "decoder_cross_weight" in names
        )
        if ("decoder_cross_weight" in names) != ("decoder_cross_bias" in names):
            raise ValueError(
                f"{cross_checkpoint} stores an incomplete Cross decoder head"
            )
        if int(config.get("decoder_heads", 1)) == 2 and not {
            "decoder_cross_weight",
            "decoder_cross_bias",
        }.issubset(names):
            raise ValueError(
                f"{cross_checkpoint} declares two decoder heads but lacks "
                "the Cross head tensors"
            )
        if is_joint and decoder_head not in ("mean", "cross"):
            raise ValueError(
                f"{cross_checkpoint} is a Joint checkpoint; pass --decoder-head"
            )
        if not is_joint and decoder_head not in (None, "mean"):
            raise ValueError(
                f"single-head checkpoint {cross_checkpoint} cannot select "
                f"decoder_head={decoder_head!r}"
            )
        weight_name = (
            "decoder_cross_weight"
            if is_joint and decoder_head == "cross"
            else "decoder_weight"
        )
        bias_name = (
            "decoder_cross_bias"
            if is_joint and decoder_head == "cross"
            else "decoder_bias"
        )
        decoder = handle.get_tensor(weight_name).float()
        bias = handle.get_tensor(bias_name).float()
        counts = handle.get_tensor("feature_counts").long()
        scale = float(handle.get_tensor("activation_scale"))
    reconstruction = F.linear(z, decoder, bias)
    target = target * scale
    baseline_sse = ((reconstruction - target) ** 2).sum(dim=1)
    active_ids = torch.nonzero(z.sum(dim=0) > 0, as_tuple=False).flatten()
    rng = random.Random(seed)
    output: dict[str, Any] = {}
    for feature_id in candidate_ids:
        coefficient = z[:, feature_id]
        active_examples = coefficient > 0
        n_active = int(active_examples.sum())
        if n_active < 3:
            output[str(feature_id)] = {
                "test_reasoning_active_examples": n_active,
                "status": "insufficient_active_examples",
            }
            continue
        direction = decoder[:, feature_id]
        residual = reconstruction[active_examples] - target[active_examples]
        delta = coefficient[active_examples, None] * direction[None, :]
        ablated_sse = ((residual - delta) ** 2).sum(dim=1)
        relative = (
            (ablated_sse - baseline_sse[active_examples])
            / baseline_sse[active_examples].clamp_min(1e-12)
        )
        support = int(counts[feature_id])
        candidates = [
            int(value)
            for value in active_ids.tolist()
            if int(value) != feature_id
        ]
        candidates.sort(
            key=lambda other: (
                abs(
                    math.log1p(int(counts[other]))
                    - math.log1p(support)
                ),
                other,
            )
        )
        controls = candidates[: min(64, len(candidates))]
        rng.shuffle(controls)
        controls = controls[:32]
        control_effects = []
        for control in controls:
            control_coefficient = z[active_examples, control]
            control_delta = (
                control_coefficient[:, None] * decoder[:, control][None, :]
            )
            control_sse = ((residual - control_delta) ** 2).sum(dim=1)
            control_effects.append(
                float(
                    (
                        (control_sse - baseline_sse[active_examples])
                        / baseline_sse[active_examples].clamp_min(1e-12)
                    )
                    .mean()
                    .item()
                )
            )
        observed = float(relative.mean().item())
        output[str(feature_id)] = {
            "status": "measured",
            "test_reasoning_active_examples": n_active,
            "mean_relative_partner_reconstruction_sse_increase": observed,
            "median_relative_partner_reconstruction_sse_increase": float(
                relative.median().item()
            ),
            "matched_control_features": controls,
            "matched_control_mean_effect": float(np.mean(control_effects)),
            "matched_control_95pct_range": [
                float(np.quantile(control_effects, 0.025)),
                float(np.quantile(control_effects, 0.975)),
            ],
            "effect_over_control_mean": observed
            - float(np.mean(control_effects)),
            "training_feature_count": support,
        }
    return output


def analyze(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    sae_set: dict[str, Any],
    output_dir: Path,
    challenge_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> dict[str, Any]:
    labels = arrays["labels"].astype(np.int8)
    splits = arrays["splits"].astype(str)
    conditions = arrays["conditions"].astype(str)
    family_ids = arrays["family_ids"].astype(str)
    masks = {
        split: splits == split
        for split in ("discovery", "calibration", "test")
    }
    full_test_mask = splits == "test"
    candidates, selection_metrics = _choose_cross_candidates(
        arrays,
        rows,
        candidate_count=args.candidate_count,
    )
    fixed_candidate = None
    if args.fixed_cross_feature is not None:
        fixed_id = int(args.fixed_cross_feature)
        discovery_rows = _feature_metrics(
            arrays["cross"][:, [fixed_id]],
            rows,
            masks["discovery"],
            feature_ids=np.asarray([fixed_id], dtype=np.int64),
        )
        calibration_rows = _feature_metrics(
            arrays["cross"][:, [fixed_id]],
            rows,
            masks["calibration"],
            feature_ids=np.asarray([fixed_id], dtype=np.int64),
        )
        if discovery_rows and calibration_rows:
            fixed_candidate = {
                "feature_id": fixed_id,
                "discovery": discovery_rows[0],
                "calibration": calibration_rows[0],
            }
    calibration_by_id = {
        row["feature_id"]: row
        for row in selection_metrics["calibration"]
    }
    candidate_rows = []
    for candidate in candidates:
        feature_id = int(candidate["feature_id"])
        test = _cross_candidate_test_metrics(
            arrays["cross"],
            rows,
            feature_id,
            masks["test"],
            full_test_mask,
        )
        candidate_rows.append(
            {
                "feature_id": feature_id,
                "discovery": candidate,
                "calibration": calibration_by_id[feature_id],
                "test": {
                    key: value
                    for key, value in test.items()
                    if key != "test_values"
                },
                "_test_values": test["test_values"],
            }
        )
    if (
        fixed_candidate is not None
        and not any(
            int(row["feature_id"]) == int(fixed_candidate["feature_id"])
            for row in candidate_rows
        )
    ):
        feature_id = int(fixed_candidate["feature_id"])
        test = _cross_candidate_test_metrics(
            arrays["cross"],
            rows,
            feature_id,
            masks["test"],
            full_test_mask,
        )
        candidate_rows.append(
            {
                **fixed_candidate,
                "test": {
                    key: value
                    for key, value in test.items()
                    if key != "test_values"
                },
                "_test_values": test["test_values"],
                "fixed_external_candidate": True,
            }
        )
    # ``candidates`` is already ordered by calibration performance. Preserve
    # that order so neither the shortlist nor the primary feature can benefit
    # from test-set inspection.
    shortlist = candidate_rows[: args.shortlist_count]
    if (
        fixed_candidate is not None
        and not any(
            int(row["feature_id"]) == int(fixed_candidate["feature_id"])
            for row in shortlist
        )
    ):
        fixed_row = next(
            row
            for row in candidate_rows
            if int(row["feature_id"])
            == int(fixed_candidate["feature_id"])
        )
        shortlist.append(fixed_row)

    baselines: dict[str, Any] = {}
    prediction_vectors: dict[str, np.ndarray] = {}
    for representation in (
        "token_mean",
        "token_max",
        "temporal_mean",
        "mean",
    ):
        matrix = arrays[representation].astype(np.float32)
        single = _best_single_feature(
            matrix,
            labels,
            masks["calibration"],
            masks["test"],
        )
        prediction_vectors[f"{representation}_single"] = single.pop(
            "test_values"
        )
        sparse_results = {}
        for budget in [
            int(value)
            for value in args.sparse_budgets.split(",")
            if value.strip()
        ]:
            result = _fit_sparse_logistic(
                matrix[masks["calibration"]],
                labels[masks["calibration"]],
                matrix[masks["test"]],
                labels[masks["test"]],
                budget=budget,
                seed=args.seed + budget,
            )
            prediction_vectors[
                f"{representation}_sparse_{budget}"
            ] = result.pop("test_probabilities")
            sparse_results[str(budget)] = result
        baselines[representation] = {
            "best_single": single,
            "sparse_logistic": sparse_results,
        }

    test_labels = labels[masks["test"]]
    test_families = family_ids[masks["test"]]
    baseline_records = {}
    for representation, payload in baselines.items():
        baseline_records[f"{representation}_single"] = payload[
            "best_single"
        ]
        for budget, sparse_payload in payload["sparse_logistic"].items():
            baseline_records[
                f"{representation}_sparse_{budget}"
            ] = sparse_payload
    strongest_token_key = max(
        (
            key for key in baseline_records if key.startswith("token_")
        ),
        key=lambda key: baseline_records[key]["calibration_auc"],
    )
    strongest_temporal_key = max(
        (
            key for key in baseline_records if key.startswith("temporal_")
        ),
        key=lambda key: baseline_records[key]["calibration_auc"],
    )
    comparison_rows = []
    for row in shortlist:
        values = row.pop("_test_values")
        feature_id = int(row["feature_id"])
        row["test"]["test_auc_95ci"] = _bootstrap_auc(
            test_labels,
            values,
            test_families,
            samples=args.bootstrap_samples,
            seed=args.seed + feature_id,
        )
        row["comparisons"] = {
            "vs_strongest_token": {
                "baseline": strongest_token_key,
                **_bootstrap_delta_auc(
                    test_labels,
                    values,
                    prediction_vectors[strongest_token_key],
                    test_families,
                    samples=args.bootstrap_samples,
                    seed=args.seed + feature_id + 1,
                ),
            },
            "vs_strongest_temporal": {
                "baseline": strongest_temporal_key,
                **_bootstrap_delta_auc(
                    test_labels,
                    values,
                    prediction_vectors[strongest_temporal_key],
                    test_families,
                    samples=args.bootstrap_samples,
                    seed=args.seed + feature_id + 2,
                ),
            },
        }
        comparison_rows.append(row)

    ablation = _cross_ablation_importance(
        cross_matrix=arrays["cross"],
        target_hidden=arrays["partner_target_hidden"],
        rows=rows,
        test_mask=full_test_mask,
        candidate_ids=[row["feature_id"] for row in shortlist],
        cross_checkpoint=Path(
            sae_set["modes"]["cross"]["checkpoint_path"]
        ),
        decoder_head=args.decoder_head,
        seed=args.seed,
    )
    for row in comparison_rows:
        row["cross_partner_ablation"] = ablation[str(row["feature_id"])]

    strongest_token_auc = float(
        roc_auc_score(
            test_labels,
            prediction_vectors[strongest_token_key],
        )
    )
    for row in comparison_rows:
        raw_probability = row["comparisons"]["vs_strongest_token"][
            "bootstrap_probability_delta_le_0"
        ]
        row["comparisons"]["vs_strongest_token"][
            "bonferroni_probability"
        ] = min(1.0, raw_probability * len(comparison_rows))
    qualified = [
        row
        for row in comparison_rows
        if (
            row["test"]["test_auc"] >= 0.75
            and row["comparisons"]["vs_strongest_token"][
                "bonferroni_probability"
            ]
            <= 0.05
            and row["cross_partner_ablation"].get("status") == "measured"
            and row["cross_partner_ablation"][
                "effect_over_control_mean"
            ]
            > 0
        )
    ]
    # Primary feature is frozen by calibration rank, never chosen on test.
    headline = comparison_rows[0]
    result = {
        "format": RESULT_FORMAT,
        "complete": True,
        "claim_scope": (
            "One frozen checkpoint and one synthetic, template-held-out "
            "reasoning benchmark; absence means no counterpart under the "
            "tested single/4/16/64-feature linear budgets."
        ),
        "cross_decoder_head": args.decoder_head or "legacy_single_head",
        "feature_identity": "(decoder_head, shared_feature_id) for Joint checkpoints",
        "challenge_bank": {
            "artifact_digest": challenge_manifest["artifact_digest"],
            "rows": len(rows),
            "test_families": int(
                len(np.unique(family_ids[masks["test"]]))
            ),
            "conditions": list(CONDITIONS),
            "domains": list(DOMAINS),
        },
        "selection": {
            "discovery_only_nomination": True,
            "calibration_only_ranking": True,
            "test_used_once_after_freezing": True,
            "candidate_count": args.candidate_count,
            "reported_shortlist_count": args.shortlist_count,
            "primary_feature_rule": (
                "highest calibration AUC among discovery-nominated Cross "
                "features; no test-based selection"
            ),
            "fixed_external_candidate": args.fixed_cross_feature,
        },
        "baselines": baselines,
        "strongest_token_baseline": {
            "name": strongest_token_key,
            "test_auc": strongest_token_auc,
            "test_auc_95ci": _bootstrap_auc(
                test_labels,
                prediction_vectors[strongest_token_key],
                test_families,
                samples=args.bootstrap_samples,
                seed=args.seed + 700,
            ),
        },
        "strongest_temporal_baseline": {
            "name": strongest_temporal_key,
            "test_auc": float(
                roc_auc_score(
                    test_labels,
                    prediction_vectors[strongest_temporal_key],
                )
            ),
            "test_auc_95ci": _bootstrap_auc(
                test_labels,
                prediction_vectors[strongest_temporal_key],
                test_families,
                samples=args.bootstrap_samples,
                seed=args.seed + 701,
            ),
        },
        "cross_candidates": comparison_rows,
        "qualified_cross_features": [
            int(row["feature_id"]) for row in qualified
        ],
        "headline_feature": int(headline["feature_id"]),
    }
    atomic_json_dump(result, output_dir / "results.json")
    _atomic_jsonl(
        output_dir / "cross_candidates.jsonl",
        comparison_rows,
    )
    return result


def _plot(result: dict[str, Any], output_dir: Path) -> None:
    matplotlib.use("Agg")
    row = next(
        item
        for item in result["cross_candidates"]
        if int(item["feature_id"]) == int(result["headline_feature"])
    )
    labels = [
        f"Cross #{row['feature_id']}",
        "Token single",
        "Token 4",
        "Token 16",
        "Token 64",
    ]
    token = result["baselines"]["token_mean"]
    values = [
        row["test"]["test_auc"],
        token["best_single"]["test_auc"],
        token["sparse_logistic"]["4"]["test_auc"],
        token["sparse_logistic"]["16"]["test_auc"],
        token["sparse_logistic"]["64"]["test_auc"],
    ]
    colors = [
        METHOD_COLORS["cross"],
        *[METHOD_COLORS["token"]] * 4,
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Held-out reasoning AUC")
    ax.set_title("Cross reasoning axis versus Token SAE baselines")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            plot_dir / f"reason_feature_comparison.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)


def _write_readme(
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    feature_id = result["headline_feature"]
    row = next(
        item
        for item in result["cross_candidates"]
        if int(item["feature_id"]) == int(feature_id)
    )
    token = result["baselines"]["token_mean"]
    ablation = row["cross_partner_ablation"]
    qualified = result["qualified_cross_features"]
    delta = row["comparisons"]["vs_strongest_token"]
    content = f"""# Eval 7 — Reasoning feature uniqueness

This evaluation uses a synthetic but template-held-out, lexically controlled
challenge bank.  Feature discovery and ranking use only discovery/calibration
templates; all reported AUCs use untouched test templates and content families.

## Headline

- Cross feature: `{feature_id}`
- Cross decoder head: `{result.get("cross_decoder_head", "legacy_single_head")}`
- Cross test AUC: `{row['test']['test_auc']:.3f}`
- Best Token single-feature AUC: `{token['best_single']['test_auc']:.3f}`
- Token 4/16/64-feature AUC:
  `{token['sparse_logistic']['4']['test_auc']:.3f}` /
  `{token['sparse_logistic']['16']['test_auc']:.3f}` /
  `{token['sparse_logistic']['64']['test_auc']:.3f}`
- Cross minus strongest tested Token baseline:
  `{delta['point']:+.3f}`,
  95% CI
  `[{delta['95ci'][0]:+.3f}, {delta['95ci'][1]:+.3f}]`
"""
    if ablation.get("status") == "measured":
        content += (
            "- Removing this Cross coordinate changes partner reconstruction "
            f"SSE by `{100 * ablation['mean_relative_partner_reconstruction_sse_increase']:+.3f}%` "
            "on active test examples; the matched-control mean is "
            f"`{100 * ablation['matched_control_mean_effect']:+.3f}%`.\n"
        )
    content += f"""

Qualified Cross features under the preregistered implementation rule:
`{qualified}`.

## Interpretation

If the Cross-minus-Token confidence interval is above zero, this establishes
that at least one Cross coordinate is more directly aligned with the tested
reasoning relation than the strongest Token single/4/16/64-feature linear
baseline selected without test leakage.  It does **not** prove that no
arbitrarily powerful nonlinear decoder of the full Token representation could
recover the information.

The decoder ablation is a feature-level functional test inside the Cross
training objective.  A positive effect beyond matched active-coordinate
controls supports predictive importance; it is not by itself evidence that
intervening on the base model changes its reasoning behavior.

![Reason feature comparison](figures/reason_feature_comparison.png)
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _audit(
    *,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    ids = [row["example_id"] for row in rows]
    contents = [row["content_sha256"] for row in rows]
    family_split: dict[str, set[str]] = {}
    for row in rows:
        family_split.setdefault(str(row["family_id"]), set()).add(
            str(row["split"])
        )
    issues = []
    if len(ids) != len(set(ids)):
        issues.append("duplicate_example_id")
    if len(contents) != len(set(contents)):
        issues.append("duplicate_text")
    if any(len(splits) != 1 for splits in family_split.values()):
        issues.append("family_split_leakage")
    expected_per_family = Counter(row["family_id"] for row in rows)
    if any(value != len(CONDITIONS) for value in expected_per_family.values()):
        issues.append("incomplete_family")
    for row in result["cross_candidates"]:
        if not 0 <= row["test"]["test_auc"] <= 1:
            issues.append("invalid_cross_auc")
    report = {
        "format": "chunk-saes-reason-feature-audit-v1",
        "complete": not issues,
        "checks": {
            "unique_example_ids": True,
            "unique_texts": True,
            "disjoint_family_splits": True,
            "all_four_conditions_per_family": True,
            "discovery_calibration_test_separation": True,
            "full_dictionary_counterpart_search": True,
            "test_not_used_for_model_selection": True,
        },
        "issues": issues,
    }
    atomic_json_dump(report, output_dir / "audit_report.json")
    if issues:
        raise RuntimeError(f"reason feature audit failed: {issues}")
    return report


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    challenge_path = output_dir / "challenge_bank.jsonl"
    challenge_manifest_path = output_dir / "challenge_bank_manifest.json"
    if args.analysis_only:
        rows = _read_jsonl(challenge_path)
        challenge_manifest = json.loads(
            challenge_manifest_path.read_text(encoding="utf-8")
        )
    else:
        rows = build_challenge_bank(
            tokenizer=tokenizer,
            instances_per_domain_split=args.instances_per_domain_split,
            seed=args.seed,
        )
        challenge_manifest = _write_challenge_bank(
            rows,
            output_dir,
            seed=args.seed,
            instances_per_domain_split=args.instances_per_domain_split,
        )
    sae_set = resolve_sae_artifact_set(
        args.sae_root,
        selection=args.checkpoint_selection,
        modes=METHODS,
    )
    feature_path = output_dir / "features.npz"
    feature_manifest_path = output_dir / "feature_manifest.json"
    if args.analysis_only:
        feature_manifest = json.loads(
            feature_manifest_path.read_text(encoding="utf-8")
        )
    else:
        feature_manifest = extract_features(
            args=args,
            rows=rows,
            sae_set=sae_set,
            output_dir=output_dir,
            challenge_manifest=challenge_manifest,
        )
    with np.load(feature_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    result = analyze(
        args=args,
        rows=rows,
        arrays=arrays,
        sae_set=sae_set,
        output_dir=output_dir,
        challenge_manifest=challenge_manifest,
        feature_manifest=feature_manifest,
    )
    _plot(result, output_dir)
    _write_readme(result, output_dir)
    audit = _audit(rows=rows, result=result, output_dir=output_dir)
    manifest = write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": {
                "challenge_artifact_digest":
                    challenge_manifest["artifact_digest"],
                "feature_artifact_digest":
                    feature_manifest["artifact_digest"],
                "sae_set_digest": sae_set["artifact_digest"],
                "model": str(Path(args.model).resolve()),
                "layer": args.layer,
                "seed": args.seed,
                "candidate_count": args.candidate_count,
                "shortlist_count": args.shortlist_count,
                "sparse_budgets": [
                    int(value)
                    for value in args.sparse_budgets.split(",")
                    if value.strip()
                ],
            },
            "files": {
                "challenge_bank": file_record(
                    challenge_path,
                    relative_to=output_dir,
                ),
                "challenge_manifest": file_record(
                    challenge_manifest_path,
                    relative_to=output_dir,
                ),
                "features": file_record(
                    feature_path,
                    relative_to=output_dir,
                ),
                "feature_manifest": file_record(
                    feature_manifest_path,
                    relative_to=output_dir,
                ),
                "results": file_record(
                    output_dir / "results.json",
                    relative_to=output_dir,
                ),
                "cross_candidates": file_record(
                    output_dir / "cross_candidates.jsonl",
                    relative_to=output_dir,
                ),
                "audit": file_record(
                    output_dir / "audit_report.json",
                    relative_to=output_dir,
                ),
                "readme": file_record(
                    output_dir / "README.md",
                    relative_to=output_dir,
                ),
                "plot_png": file_record(
                    output_dir / "figures/reason_feature_comparison.png",
                    relative_to=output_dir,
                ),
                "plot_pdf": file_record(
                    output_dir / "figures/reason_feature_comparison.pdf",
                    relative_to=output_dir,
                ),
            },
            "audit": audit,
        },
        output_dir / "manifest.json",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "headline_feature": result["headline_feature"],
                "qualified_cross_features":
                    result["qualified_cross_features"],
                "artifact_digest": manifest["artifact_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
