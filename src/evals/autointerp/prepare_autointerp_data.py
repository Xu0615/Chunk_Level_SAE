#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.stats import poisson
from transformers import AutoTokenizer

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.autointerp import AUTOINTERP_PROTOCOL, AutoInterpConfig
from chunk_saes.corpus import discover_dataset_shards
from chunk_saes.data import iter_dataset_jsonl
from chunk_saes.evaluation_protocol import (
    full_dictionary_feature_widths,
    validate_full_dictionary_width,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.runtime import (
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.sample_plan import PreparedDocument, solve_exact_cell_counts
from chunk_saes.streaming_plan import (
    StreamingExactPlanner,
    StreamingRankAllocator,
    iter_tokenized_prepared_documents,
)
from chunk_saes.utils import (
    atomic_json_dump,
    content_hash,
    dtype_from_name,
    log,
    parse_int_csv,
    tokenizer_fingerprint,
)


DATA_FORMAT = "chunk-saes-autointerp-data-v2"
TOKEN_METHODS = ("token", "temporal")
CHUNK_METHODS = ("mean", "cross")
METHODS = TOKEN_METHODS + CHUNK_METHODS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare SAEBench token-window AutoInterp data for Token/Temporal "
            "and complete training-style chunk data for Mean/Cross."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--checkpoint-selection", choices=("best", "final"), default="best")
    p.add_argument("--token-total-tokens", type=int, default=32_000_000)
    p.add_argument("--token-context-size", type=int, default=128)
    p.add_argument("--chunk-total-tokens", type=int, default=256_000_000)
    p.add_argument("--chunk-lengths", default="32,64,128,256,512")
    p.add_argument("--chunk-plan-shard-tokens", type=int, default=32_000_000)
    p.add_argument("--max-document-reuses", type=int, default=64)
    p.add_argument("--tokenizer-batch-size", type=int, default=1024)
    p.add_argument("--tokenizer-batch-chars", type=int, default=8_000_000)
    p.add_argument("--forward-batch-size", type=int, default=1024)
    p.add_argument("--forward-token-budget", type=int, default=98_304)
    p.add_argument("--feature-block-size", type=int, default=256)
    p.add_argument("--coverage-sequence-batch-size", type=int, default=64)
    p.add_argument("--coverage-chunk-batch-size", type=int, default=8192)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--activation-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--feature-sample-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coverage-family-alpha", type=float, default=0.01)
    p.add_argument("--keep-coverage-tensors", action="store_true")
    p.add_argument("--skip-analytic-budget-check", action="store_true")
    p.add_argument("--allow-incomplete-coverage", action="store_true")
    p.add_argument(
        "--resume-existing-forward",
        action="store_true",
        help=(
            "Reuse a complete token/chunk plan and forward outputs in output-dir, "
            "then resume at full-dictionary coverage auditing."
        ),
    )
    p.add_argument(
        "--reuse-existing-chunk-forward",
        action="store_true",
        help=(
            "Regenerate/extend only the Token/Temporal pool while reusing an "
            "already complete chunk plan, chunk means, and chunk activations."
        ),
    )
    p.add_argument(
        "--recompute-chunk-means-for-coverage",
        action="store_true",
        help=(
            "With --reuse-existing-chunk-forward, rerun the existing chunk "
            "plan to materialize native chunk means and re-audit all 65,536 "
            "Mean/Cross features under the exact AutoInterp active threshold."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def _lightweight_dataset_identity(dataset: str | Path) -> dict[str, Any]:
    root, shards = discover_dataset_shards(dataset)
    digest = hashlib.sha256()
    for shard in shards:
        relative = shard.path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(shard.size).encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(Path(dataset).resolve()),
        "root": str(root),
        "shard_count": len(shards),
        "total_bytes": sum(shard.size for shard in shards),
        "identity_algorithm": "sha256(relative_path,size)",
        "identity_digest": digest.hexdigest(),
    }


def _load_checkpoint_parameters(
    checkpoint_dir: Path,
    *,
    feature_ids: list[int] | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    with safe_open(
        str(checkpoint_dir / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        names = set(handle.keys())
        encoder_weight = handle.get_tensor("encoder_weight")
        encoder_bias = handle.get_tensor("encoder_bias")
        counts = handle.get_tensor("feature_counts")
        dict_size = int(counts.numel())
        if int(encoder_weight.shape[0]) != dict_size:
            raise ValueError(
                f"{checkpoint_dir} encoder exposes {encoder_weight.shape[0]} "
                f"features but the checkpoint dictionary has {dict_size}"
            )
        if feature_ids is not None:
            ids = torch.tensor(feature_ids, dtype=torch.long)
            encoder_weight = encoder_weight.index_select(0, ids)
            encoder_bias = encoder_bias.index_select(0, ids)
        decoder_bias = handle.get_tensor("decoder_bias")
        if "pre_bias" in names:
            pre_bias = handle.get_tensor("pre_bias")
        elif config.get("sae_parameter_schema_version") == SAE_PARAMETER_SCHEMA_VERSION:
            raise ValueError(f"{checkpoint_dir} lacks required pre_bias")
        else:
            pre_bias = decoder_bias
        result = {
            "config": config,
            "encoder_weight": encoder_weight,
            "encoder_bias": encoder_bias,
            "pre_bias": pre_bias,
            "decoder_bias": decoder_bias,
            "threshold": float(handle.get_tensor("threshold")),
            "scale": float(handle.get_tensor("activation_scale")),
            "feature_counts": counts,
            "dict_size": dict_size,
        }
    if dtype is not None:
        for key in ("encoder_weight", "encoder_bias", "pre_bias", "decoder_bias"):
            result[key] = result[key].to(dtype=dtype)
    return result
def _alive_feature_ids(checkpoint_dir: Path) -> list[int]:
    with safe_open(
        str(checkpoint_dir / "sae.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        counts = handle.get_tensor("feature_counts")
    return torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()


def _sample_alive_features(
    checkpoint_dir: Path,
    *,
    sample_size: int,
    seed: int,
) -> list[int]:
    """Backward-compatible helper for sampling one checkpoint independently."""

    alive = _alive_feature_ids(checkpoint_dir)
    if len(alive) < sample_size:
        raise ValueError(
            f"{checkpoint_dir} has only {len(alive)} alive features; "
            f"cannot sample {sample_size}"
        )
    return sorted(random.Random(seed).sample(alive, k=sample_size))


def _required_poisson_mean(
    *,
    required_events: int,
    features: int,
    family_alpha: float,
) -> float:
    per_feature = family_alpha / features
    lower, upper = float(required_events), float(required_events)
    while poisson.cdf(required_events - 1, upper) > per_feature:
        upper *= 2
    for _ in range(80):
        middle = (lower + upper) / 2
        if poisson.cdf(required_events - 1, middle) > per_feature:
            lower = middle
        else:
            upper = middle
    return upper


def _round_up(value: float, unit: int) -> int:
    return int(math.ceil(value / unit) * unit)


def _budget_report(
    sae_set: dict[str, Any],
    *,
    protocol: AutoInterpConfig,
    chunk_lengths: list[int],
    family_alpha: float,
    chosen_token_budget: int,
    chosen_chunk_budget: int,
) -> dict[str, Any]:
    required = protocol.n_top_total + protocol.n_importance_total
    widths = []
    min_counts = {}
    for method in METHODS:
        checkpoint = Path(sae_set["modes"][method]["checkpoint_path"])
        parameters = _load_checkpoint_parameters(checkpoint)
        widths.append(parameters["dict_size"])
        alive_counts = parameters["feature_counts"][
            parameters["feature_counts"] > 0
        ]
        min_counts[method] = int(alive_counts.min().item())
    if len(set(widths)) != 1:
        raise ValueError(f"SAE dictionary widths differ: {widths}")
    width = widths[0]
    lam = _required_poisson_mean(
        required_events=required,
        features=width,
        family_alpha=family_alpha,
    )
    # Token/Temporal AutoInterp only permits centers outside the 10-token
    # context buffer on each side.
    eligible_token_fraction = (
        protocol.context_size - 2 * protocol.buffer
    ) / protocol.context_size
    token_bounds = {
        method: (
            lam
            * 1_000_000_000
            / min_counts[method]
            / eligible_token_fraction
        )
        for method in TOKEN_METHODS
    }
    # Training feature_counts for Mean/Cross are occurrence weighted: one
    # active chunk contributes once per token in that chunk. Dividing by the
    # maximum possible chunk length gives a conservative lower bound on
    # independent active chunks for any mixture of the training lengths.
    maximum_chunk_length = max(chunk_lengths)
    chunk_bounds = {
        method: lam * maximum_chunk_length * 1_000_000_000 / min_counts[method]
        for method in CHUNK_METHODS
    }
    recommended_token = _round_up(max(token_bounds.values()), 1_000_000)
    recommended_chunk = _round_up(max(chunk_bounds.values()), 32_000_000)
    return {
        "model": "Poisson lower-tail with Bonferroni family-wise coverage",
        "family_alpha": family_alpha,
        "dictionary_features": width,
        "required_positive_examples": required,
        "required_expected_events_per_rarest_feature": lam,
        "training_min_feature_counts": min_counts,
        "token_budget_lower_bounds": token_bounds,
        "chunk_budget_lower_bounds_conservative_max_length": chunk_bounds,
        "maximum_chunk_length": maximum_chunk_length,
        "eligible_token_fraction_after_context_buffer": eligible_token_fraction,
        "recommended_token_budget": recommended_token,
        "recommended_chunk_budget": recommended_chunk,
        "chosen_token_budget": chosen_token_budget,
        "chosen_chunk_budget": chosen_chunk_budget,
        "chosen_budgets_meet_analytic_bounds": (
            chosen_token_budget >= recommended_token
            and chosen_chunk_budget >= recommended_chunk
        ),
        "note": (
            "The analytic bound is followed by an exact empirical two-pass "
            "coverage scan over every feature. LLM scoring is blocked unless "
            "all 65,536 features satisfy the support criterion."
        ),
    }


def _tokenize_token_pool(
    dataset: Path,
    tokenizer,
    *,
    context_size: int,
    total_tokens: int,
) -> torch.Tensor:
    target = total_tokens + context_size + 1
    eos = (
        [int(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id is not None
        else []
    )
    tokens: list[int] = []
    for _shard, _ordinal, record in iter_dataset_jsonl(dataset):
        text = record.get("text")
        if not isinstance(text, str) or len(text) <= 100:
            continue
        if tokens and eos:
            tokens.extend(eos)
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(tokens) >= target:
            break
    if len(tokens) <= total_tokens:
        raise ValueError(
            f"dataset provided only {len(tokens)} tokens, need > {total_tokens}"
        )
    usable = min(len(tokens), target)
    usable -= usable % context_size
    tensor = torch.tensor(tokens[:usable], dtype=torch.long).reshape(
        -1, context_size
    )
    if tokenizer.bos_token_id is not None:
        tensor[:, 0] = int(tokenizer.bos_token_id)
    return tensor


class _ChunkPlanWriter:
    def __init__(self, output_dir: Path, shard_token_limit: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_token_limit = max(1, int(shard_token_limit))
        self.token_ids: list[int] = []
        self.offsets = [0]
        self.lengths: list[int] = []
        self.chunk_ids: list[int] = []
        self.pair_ids: list[int] = []
        self.sides: list[int] = []
        self.doc_hashes: list[list[int]] = []
        self.shards: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.total_chunks = 0
        self.length_counts: Counter[int] = Counter()

    def add_pair(self, row) -> None:
        for side, ids in enumerate((row.input_ids_a, row.input_ids_b)):
            self.token_ids.extend(int(value) for value in ids)
            self.offsets.append(len(self.token_ids))
            self.lengths.append(len(ids))
            self.chunk_ids.append(self.total_chunks)
            self.pair_ids.append(int(row.pair_id))
            self.sides.append(side)
            self.doc_hashes.append(list(row.doc_hash))
            self.total_chunks += 1
            self.total_tokens += len(ids)
            self.length_counts[len(ids)] += 1
        if len(self.token_ids) >= self.shard_token_limit:
            self.flush()

    def flush(self) -> None:
        if not self.lengths:
            return
        shard_id = len(self.shards)
        path = self.output_dir / f"shard-{shard_id:05d}.safetensors"
        tensors = {
            "token_ids": torch.tensor(self.token_ids, dtype=torch.int32),
            "offsets": torch.tensor(self.offsets, dtype=torch.int64),
            "lengths": torch.tensor(self.lengths, dtype=torch.int32),
            "chunk_ids": torch.tensor(self.chunk_ids, dtype=torch.int64),
            "pair_ids": torch.tensor(self.pair_ids, dtype=torch.int64),
            "sides": torch.tensor(self.sides, dtype=torch.int8),
            "doc_hashes": torch.tensor(self.doc_hashes, dtype=torch.uint8),
        }
        save_file(tensors, str(path))
        self.shards.append(
            {
                "path": path.name,
                "tokens": len(self.token_ids),
                "chunks": len(self.lengths),
                "chunk_id_min": min(self.chunk_ids),
                "chunk_id_max": max(self.chunk_ids),
                "bytes": path.stat().st_size,
            }
        )
        self.token_ids = []
        self.offsets = [0]
        self.lengths = []
        self.chunk_ids = []
        self.pair_ids = []
        self.sides = []
        self.doc_hashes = []

    def finish(self) -> dict[str, Any]:
        self.flush()
        return {
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "length_counts": {
                str(length): count
                for length, count in sorted(self.length_counts.items())
            },
            "shards": self.shards,
        }


def _local_prepared_documents(dataset: Path) -> Iterable[PreparedDocument]:
    for stream_ordinal, (shard, ordinal, record) in enumerate(
        iter_dataset_jsonl(dataset)
    ):
        text = record.get("text")
        if not isinstance(text, str) or len(text) <= 100:
            continue
        digest = content_hash(text)
        yield PreparedDocument(
            stream_id=0,
            ordinal=stream_ordinal,
            doc_id=f"local:{shard}:{ordinal}:{digest}",
            source="Pile-Uncopyrighted",
            text=text,
            content_hash=digest,
        )


def _build_chunk_plan(
    dataset: Path,
    tokenizer,
    output_dir: Path,
    *,
    target_tokens: int,
    lengths: list[int],
    seed: int,
    max_document_reuses: int,
    tokenizer_batch_size: int,
    tokenizer_batch_chars: int,
    shard_tokens: int,
) -> dict[str, Any]:
    counts = solve_exact_cell_counts(
        target_tokens,
        ["Pile-Uncopyrighted"],
        lengths,
        seed=seed,
    )
    allocator = StreamingRankAllocator(
        counts,
        world_size=1,
        target_tokens=target_tokens,
        seed=seed,
    )
    planner = StreamingExactPlanner(
        counts,
        seed=seed,
        max_document_reuses=max_document_reuses,
        rank_allocator=allocator,
    )
    writer = _ChunkPlanWriter(output_dir, shard_tokens)
    documents = _local_prepared_documents(dataset)
    tokenized = iter_tokenized_prepared_documents(
        documents,
        tokenizer,
        batch_size=tokenizer_batch_size,
        batch_chars=tokenizer_batch_chars,
    )
    for document, token_ids in tokenized:
        for row in planner.add_document(document, token_ids):
            writer.add_pair(row)
        if planner.documents_scanned % 10_000 == 0:
            print(
                "[chunk-plan] "
                f"documents={planner.documents_scanned} "
                f"candidate_tokens={planner.candidate_tokens} "
                f"selected={planner.selected_tokens}/{target_tokens}",
                flush=True,
            )
        if planner.complete:
            break
    planner_stats = planner.finish(
        target_tokens=target_tokens,
        expected_pairs=sum(counts.values()),
    )
    allocation = allocator.finish()
    written = writer.finish()
    if written["total_tokens"] != target_tokens:
        raise RuntimeError(
            f"chunk plan wrote {written['total_tokens']} != {target_tokens}"
        )
    manifest = {
        "format": "chunk-saes-autointerp-chunk-plan-v1",
        "complete": True,
        "target_tokens": target_tokens,
        "chunk_lengths": lengths,
        "sample_seed": seed,
        "planner_stats": planner_stats,
        "rank_allocation": allocation,
        **written,
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    return manifest


def _selected_scores(
    hidden: torch.Tensor,
    parameters: dict[str, Any],
) -> torch.Tensor:
    weight = parameters["encoder_weight"].to(
        hidden.device,
        dtype=hidden.dtype,
    )
    bias = parameters["encoder_bias"].to(
        hidden.device,
        dtype=hidden.dtype,
    )
    pre_bias = parameters["pre_bias"].to(
        hidden.device,
        dtype=hidden.dtype,
    )
    pre = F.relu(
        F.linear(hidden * parameters["scale"] - pre_bias, weight, bias)
    )
    return pre * (pre > parameters["threshold"])


def _move_selected_parameters(
    parameters: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    result = dict(parameters)
    for key in ("encoder_weight", "encoder_bias", "pre_bias", "decoder_bias"):
        result[key] = result[key].to(device=device, dtype=dtype)
    return result


def _run_token_pool(
    args,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    sae_set: dict[str, Any],
    feature_ids: dict[str, list[int]],
    tokens_path: Path,
    output_dir: Path,
    extractor: TargetLayerExtractor,
) -> None:
    """Materialize native token activations for Token and Temporal SAEs."""

    del local_rank
    tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
    start = len(tokens) * rank // world_size
    stop = len(tokens) * (rank + 1) // world_size
    local_tokens = tokens[start:stop]
    dtype = dtype_from_name(args.activation_dtype)
    parameters = {
        method: _move_selected_parameters(
            _load_checkpoint_parameters(
                Path(sae_set["modes"][method]["checkpoint_path"]),
                feature_ids=feature_ids[method],
            ),
            device=extractor.device,
            dtype=dtype,
        )
        for method in TOKEN_METHODS
    }
    expected_widths = full_dictionary_feature_widths(
        sae_set, modes=TOKEN_METHODS
    )
    for method in TOKEN_METHODS:
        validate_full_dictionary_width(
            method,
            parameters[method]["dict_size"],
            expected_widths[method],
        )
    batch_limit = max(
        1,
        min(
            args.forward_batch_size,
            args.forward_token_budget // args.token_context_size,
        ),
    )
    hidden_parts: list[torch.Tensor] = []
    activation_parts = {method: [] for method in TOKEN_METHODS}
    for batch_start in range(0, len(local_tokens), batch_limit):
        rows = local_tokens[batch_start : batch_start + batch_limit].tolist()
        batch = extractor.forward_ids(rows)
        hidden = batch.hidden
        hidden_parts.append(hidden.to("cpu", dtype=dtype))
        flat = hidden.reshape(-1, hidden.shape[-1])
        for method in TOKEN_METHODS:
            scores = _selected_scores(flat, parameters[method]).reshape(
                hidden.shape[0], hidden.shape[1], -1
            )
            # Store feature-major so one feature can be read contiguously by
            # the AutoInterp sampler. Sequence-major [B,L,F] would require
            # rereading the complete multi-GB shard for every feature.
            activation_parts[method].append(
                scores.permute(2, 0, 1).to("cpu", dtype=dtype)
            )
        log(
            f"token AutoInterp rank={rank} "
            f"{min(batch_start + batch_limit, len(local_tokens))}/"
            f"{len(local_tokens)} sequences",
            rank=rank,
        )
    hidden_tensor = (
        torch.cat(hidden_parts, dim=0).contiguous()
        if hidden_parts
        else torch.empty(
            (0, args.token_context_size, extractor.hidden_size), dtype=dtype
        )
    )
    valid_mask = torch.ones_like(local_tokens, dtype=torch.bool)
    for token_id in getattr(args, "special_token_ids", ()):
        valid_mask &= local_tokens != int(token_id)
    hidden_dir = output_dir / "work" / "token_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hidden": hidden_tensor,
            "valid_mask": valid_mask.contiguous(),
            "sequence_start": torch.tensor(start, dtype=torch.long),
        },
        str(hidden_dir / f"rank-{rank:03d}.safetensors"),
    )
    for method in TOKEN_METHODS:
        method_dir = output_dir / "token_pool" / "activations" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        values = (
            torch.cat(activation_parts[method], dim=1).contiguous()
            if activation_parts[method]
            else torch.empty(
                (len(feature_ids[method]), 0, args.token_context_size),
                dtype=dtype,
            )
        )
        save_file(
            {
                "activations": values,
                "feature_ids": torch.tensor(feature_ids[method], dtype=torch.long),
                "sequence_start": torch.tensor(start, dtype=torch.long),
                "layout_version": torch.tensor(2, dtype=torch.long),
            },
            str(method_dir / f"rank-{rank:03d}.safetensors"),
        )
def _chunk_rows(
    token_ids: torch.Tensor,
    offsets: torch.Tensor,
    indices: list[int],
) -> list[list[int]]:
    return [
        token_ids[int(offsets[index]) : int(offsets[index + 1])].tolist()
        for index in indices
    ]


def _run_chunk_pool(
    args,
    *,
    rank: int,
    world_size: int,
    sae_set: dict[str, Any],
    feature_ids: dict[str, list[int]],
    chunk_plan: dict[str, Any],
    plan_dir: Path,
    output_dir: Path,
    extractor: TargetLayerExtractor,
) -> None:
    dtype = dtype_from_name(args.activation_dtype)
    parameters = {
        method: _move_selected_parameters(
            _load_checkpoint_parameters(
                Path(sae_set["modes"][method]["checkpoint_path"]),
                feature_ids=feature_ids[method],
            ),
            device=extractor.device,
            dtype=dtype,
        )
        for method in CHUNK_METHODS
    }
    for shard_index, item in enumerate(chunk_plan["shards"]):
        if shard_index % world_size != rank:
            continue
        path = plan_dir / item["path"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            token_ids = handle.get_tensor("token_ids")
            offsets = handle.get_tensor("offsets")
            lengths = handle.get_tensor("lengths")
            chunk_ids = handle.get_tensor("chunk_ids")
        means = torch.empty(
            (len(lengths), extractor.hidden_size),
            dtype=dtype,
        )
        selected = {
            method: torch.empty(
                (len(lengths), len(feature_ids[method])),
                dtype=dtype,
            )
            for method in CHUNK_METHODS
        }
        for length in sorted(set(lengths.tolist())):
            indices = torch.nonzero(
                lengths == length,
                as_tuple=False,
            ).flatten().tolist()
            limit = max(
                1,
                min(
                    args.forward_batch_size,
                    args.forward_token_budget // int(length),
                ),
            )
            for begin in range(0, len(indices), limit):
                current = indices[begin : begin + limit]
                batch = extractor.forward_ids(
                    _chunk_rows(token_ids, offsets, current)
                )
                current_means = batch.means()
                means[current] = current_means.to("cpu", dtype=dtype)
                for method in CHUNK_METHODS:
                    selected[method][current] = _selected_scores(
                        current_means,
                        parameters[method],
                    ).to("cpu", dtype=dtype)
        work_dir = output_dir / "work" / "chunk_means"
        work_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "means": means.contiguous(),
                "chunk_ids": chunk_ids.contiguous(),
            },
            str(work_dir / f"shard-{shard_index:05d}.safetensors"),
        )
        activation_dir = output_dir / "chunk_pool" / "activations"
        activation_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "chunk_ids": chunk_ids.contiguous(),
                "mean_activations": selected["mean"].contiguous(),
                "cross_activations": selected["cross"].contiguous(),
                "mean_feature_ids": torch.tensor(
                    feature_ids["mean"],
                    dtype=torch.long,
                ),
                "cross_feature_ids": torch.tensor(
                    feature_ids["cross"],
                    dtype=torch.long,
                ),
            },
            str(activation_dir / f"shard-{shard_index:05d}.safetensors"),
        )
        log(
            f"chunk AutoInterp rank={rank} shard={shard_index + 1}/"
            f"{len(chunk_plan['shards'])} chunks={len(lengths)}",
            rank=rank,
        )


def _merge_top(
    current: torch.Tensor,
    values: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    take = min(k, values.shape[0])
    local = values.topk(take, dim=0).values
    return torch.cat((current, local), dim=0).topk(k, dim=0).values


def _coverage_scan(
    *,
    method: str,
    checkpoint: Path,
    shard_paths: list[Path],
    tensor_name: str,
    unit: str,
    feature_block_size: int,
    unit_batch_size: int,
    protocol: AutoInterpConfig,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Audit exact support required by the active-example protocol.

    Pass 1 finds each feature's maximum activation. Pass 2 applies the same
    ``activation_threshold_fraction * max`` threshold used by example
    construction. For token features, 12 active sequences conservatively
    guarantee 12 disjoint top windows and 19 active positions guarantee seven
    additional importance candidates after excluding those top centers. For a
    chunk feature, each active chunk is one independent candidate, so 19 active
    chunks guarantee all 12 top and seven importance examples.
    """

    parameters = _load_checkpoint_parameters(checkpoint)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    weight = parameters["encoder_weight"].to(device=device, dtype=dtype)
    bias = parameters["encoder_bias"].to(device=device, dtype=dtype)
    pre_bias = parameters["pre_bias"].to(device=device, dtype=dtype)
    width = parameters["dict_size"]
    maximum = torch.zeros(width, dtype=torch.float32, device=device)
    local_paths = shard_paths[rank::world_size]
    if not local_paths:
        raise ValueError(
            f"rank {rank} received no {method} coverage shards from "
            f"{len(shard_paths)} paths"
        )

    def batches(path: Path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            source = handle.get_slice(tensor_name)
            shape = source.get_shape()
            expected_rank = 3 if unit == "token" else 2
            if len(shape) != expected_rank:
                raise ValueError(
                    f"{method} {tensor_name} has rank {len(shape)}, "
                    f"expected {expected_rank}"
                )
            valid_source = (
                handle.get_slice("valid_mask")
                if unit == "token" and "valid_mask" in handle.keys()
                else None
            )
            for batch_start in range(0, shape[0], unit_batch_size):
                batch_stop = min(batch_start + unit_batch_size, shape[0])
                values = source[batch_start:batch_stop].to(
                    device=device, dtype=dtype
                )
                if unit == "token":
                    eligible = values[
                        :, protocol.buffer : shape[1] - protocol.buffer
                    ]
                    if valid_source is not None:
                        valid = valid_source[batch_start:batch_stop][
                            :, protocol.buffer : shape[1] - protocol.buffer
                        ].to(device=device, dtype=torch.bool)
                    else:
                        valid = torch.ones(
                            eligible.shape[:2],
                            dtype=torch.bool,
                            device=device,
                        )
                    yield (
                        eligible.reshape(-1, shape[-1]),
                        batch_stop - batch_start,
                        eligible.shape[1],
                        valid.reshape(-1),
                    )
                else:
                    yield values, batch_stop - batch_start, 1, None

    for path in local_paths:
        for flat, _units, _positions_per_unit, valid_flat in batches(path):
            centered = flat * parameters["scale"] - pre_bias
            for begin in range(0, width, feature_block_size):
                end = min(begin + feature_block_size, width)
                pre = F.relu(
                    F.linear(centered, weight[begin:end], bias[begin:end])
                )
                scores = pre * (pre > parameters["threshold"])
                if valid_flat is not None:
                    scores = scores.masked_fill(
                        ~valid_flat.unsqueeze(1),
                        0,
                    )
                maximum[begin:end] = torch.maximum(
                    maximum[begin:end],
                    scores.float().amax(dim=0),
                )
                del pre, scores
            del flat, centered
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    active_threshold = maximum * protocol.activation_threshold_fraction

    active_units = torch.zeros(width, dtype=torch.int64, device=device)
    active_positions = torch.zeros(width, dtype=torch.int64, device=device)
    for path in local_paths:
        for flat, units, positions_per_unit, valid_flat in batches(path):
            centered = flat * parameters["scale"] - pre_bias
            for begin in range(0, width, feature_block_size):
                end = min(begin + feature_block_size, width)
                pre = F.relu(
                    F.linear(centered, weight[begin:end], bias[begin:end])
                )
                scores = pre * (pre > parameters["threshold"])
                if valid_flat is not None:
                    scores = scores.masked_fill(
                        ~valid_flat.unsqueeze(1),
                        0,
                    )
                active = scores.float() > active_threshold[begin:end].unsqueeze(0)
                active_positions[begin:end] += active.sum(dim=0)
                if unit == "token":
                    active_units[begin:end] += active.reshape(
                        units, positions_per_unit, end - begin
                    ).any(dim=1).sum(dim=0)
                else:
                    active_units[begin:end] += active.sum(dim=0)
                del pre, scores, active
            del flat, centered
    if dist.is_initialized():
        dist.all_reduce(active_units, op=dist.ReduceOp.SUM)
        dist.all_reduce(active_positions, op=dist.ReduceOp.SUM)

    required_positive_candidates = (
        protocol.n_top_total + protocol.n_importance_total
    )
    if unit == "token":
        scorable = (
            (maximum > 0)
            & (active_units >= protocol.n_top_total)
            & (active_positions >= required_positive_candidates)
        )
    else:
        scorable = (
            (maximum > 0)
            & (active_units >= required_positive_candidates)
        )
    tensors = {
        "positive_units": active_units.cpu(),
        "positive_positions": active_positions.cpu(),
        "maximum_activation": maximum.cpu(),
        "active_threshold": active_threshold.cpu(),
        "scorable": scorable.cpu(),
    }
    quantiles = torch.tensor(
        [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    )
    criterion = (
        {
            "distinct_active_sequences": protocol.n_top_total,
            "active_token_positions": required_positive_candidates,
            "eligible_token_positions_per_sequence":
                protocol.context_size - 2 * protocol.buffer,
            "active_definition":
                f"activation > {protocol.activation_threshold_fraction} * max",
        }
        if unit == "token"
        else {
            "active_chunks": required_positive_candidates,
            "active_definition":
                f"activation > {protocol.activation_threshold_fraction} * max",
        }
    )
    report = {
        "method": method,
        "unit": unit,
        "granularity": (
            "token-centered-window" if unit == "token"
            else "complete-variable-length-chunk"
        ),
        "features": width,
        "scorable_features": int(scorable.sum().item()),
        "unscorable_features": int((~scorable).sum().item()),
        "zero_positive_features": int((maximum == 0).sum().item()),
        "insufficient_top_units": int(
            (active_units < protocol.n_top_total).sum().item()
        ),
        "insufficient_positive_candidates": int(
            (active_positions < required_positive_candidates).sum().item()
        ),
        "positive_unit_quantiles": torch.quantile(
            active_units.float().cpu(), quantiles
        ).tolist(),
        "positive_position_quantiles": torch.quantile(
            active_positions.float().cpu(), quantiles
        ).tolist(),
        "coverage_complete": bool(scorable.all().item()),
        "criterion": criterion,
    }
    del weight, bias, pre_bias, maximum, active_threshold
    torch.cuda.empty_cache()
    return report, tensors


def _coverage_and_cleanup(
    args,
    *,
    output_dir: Path,
    sae_set: dict[str, Any],
    protocol: AutoInterpConfig,
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict[str, Any]:
    token_paths = sorted(
        (output_dir / "work" / "token_hidden").glob("rank-*.safetensors")
    )
    if not token_paths:
        raise RuntimeError("token hidden states are missing for coverage auditing")
    reports: dict[str, Any] = {}
    coverage_dir = output_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    for method in TOKEN_METHODS:
        report, tensors = _coverage_scan(
            method=method,
            checkpoint=Path(sae_set["modes"][method]["checkpoint_path"]),
            shard_paths=token_paths,
            tensor_name="hidden",
            unit="token",
            feature_block_size=args.feature_block_size,
            unit_batch_size=args.coverage_sequence_batch_size,
            protocol=protocol,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        reports[method] = report
        if rank == 0:
            save_file(tensors, str(coverage_dir / f"{method}.safetensors"))

    chunk_paths = sorted(
        (output_dir / "work" / "chunk_means").glob("shard-*.safetensors")
    )
    existing_methods: dict[str, Any] = {}
    report_path = output_dir / "coverage_report.json"
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            existing_methods = (json.load(handle).get("methods") or {})
    for method in CHUNK_METHODS:
        if chunk_paths:
            report, tensors = _coverage_scan(
                method=method,
                checkpoint=Path(sae_set["modes"][method]["checkpoint_path"]),
                shard_paths=chunk_paths,
                tensor_name="means",
                unit="chunk",
                feature_block_size=args.feature_block_size,
                unit_batch_size=args.coverage_chunk_batch_size,
                protocol=protocol,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
            reports[method] = report
            if rank == 0:
                save_file(tensors, str(coverage_dir / f"{method}.safetensors"))
        else:
            reusable = existing_methods.get(method)
            if (
                not isinstance(reusable, dict)
                or reusable.get("unit") != "chunk"
                or reusable.get("coverage_complete") is not True
                or not (coverage_dir / f"{method}.safetensors").is_file()
            ):
                raise RuntimeError(
                    f"{method} native chunk coverage is unavailable and cannot be reused"
                )
            reports[method] = reusable

    complete = all(reports[method]["coverage_complete"] for method in METHODS)
    payload = {"complete_dictionary_coverage": complete, "methods": reports}
    if rank == 0:
        atomic_json_dump(payload, output_dir / "coverage_report.json")
    if not complete and not args.allow_incomplete_coverage:
        failures = {
            method: reports[method]["unscorable_features"]
            for method in METHODS
            if not reports[method]["coverage_complete"]
        }
        raise RuntimeError(
            "AutoInterp pools do not cover the complete dictionaries: "
            f"{failures}. Increase the relevant token budget."
        )
    if rank == 0 and not args.keep_coverage_tensors:
        shutil.rmtree(output_dir / "work", ignore_errors=True)
    return payload


def _materialized_token_budget(total_tokens: int, context_size: int) -> int:
    return ((total_tokens + context_size + 1) // context_size) * context_size


def _identity(
    args,
    *,
    sae_set: dict[str, Any],
    feature_ids: dict[str, list[int]],
    feature_widths: dict[str, int],
    tokenizer,
    chunk_lengths: list[int],
    budget: dict[str, Any],
    world_size: int,
    protocol: AutoInterpConfig,
) -> dict[str, Any]:
    return {
        "protocol": AUTOINTERP_PROTOCOL,
        "protocol_config": protocol.to_dict(),
        "model": str(Path(args.model).resolve()),
        "tokenizer_fingerprint": tokenizer_fingerprint(args.model),
        "layer": args.layer,
        "dataset": _lightweight_dataset_identity(args.dataset),
        "token_pool": {
            "requested_total_tokens": args.token_total_tokens,
            "total_tokens": _materialized_token_budget(
                args.token_total_tokens, args.token_context_size
            ),
            "context_size": args.token_context_size,
            "granularity": "saebench-token-centered-window",
        },
        "chunk_pool": {
            "total_tokens": args.chunk_total_tokens,
            "chunk_lengths": chunk_lengths,
            "granularity": "complete-variable-length-chunk",
            "length_distribution":
                "exactly balanced over ordered training length pairs",
            "sample_seed": args.seed + 17,
        },
        "feature_selection": {
            "sample_size_per_method": args.feature_sample_size,
            "population": "complete training-alive dictionary",
            "algorithm": "python random.Random(seed).sample",
            "seed": args.seed,
            "temporal_prefix_restriction": False,
        },
        "feature_ids": feature_ids,
        "feature_widths": feature_widths,
        "granularity_protocol": {
            "token": "token-centered-window",
            "temporal": "token-centered-window",
            "mean": "complete-variable-length-chunk",
            "cross": "complete-variable-length-chunk",
        },
        "sae_set_digest": sae_set["artifact_digest"],
        "checkpoint_selection": args.checkpoint_selection,
        "world_size": world_size,
        "activation_dtype": args.activation_dtype,
        "model_dtype": args.model_dtype,
        "budget_analysis": budget,
    }


def _validate_reusable_chunk_forward(
    output_dir: Path,
    *,
    expected_chunk_tokens: int,
    feature_ids: dict[str, list[int]],
) -> dict[str, Any]:
    plan_path = output_dir / "chunk_pool" / "plan" / "manifest.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if int(plan.get("total_tokens", -1)) != expected_chunk_tokens:
        raise ValueError("existing chunk plan budget differs from requested budget")
    paths = sorted(
        (output_dir / "chunk_pool" / "activations").glob("shard-*.safetensors")
    )
    if len(paths) != len(plan.get("shards") or []):
        raise RuntimeError("existing chunk activations are incomplete")
    with safe_open(str(paths[0]), framework="pt", device="cpu") as handle:
        for method in CHUNK_METHODS:
            stored = list(
                map(int, handle.get_tensor(f"{method}_feature_ids").tolist())
            )
            if stored != feature_ids[method]:
                raise ValueError(f"existing {method} feature IDs differ")
    report_path = output_dir / "coverage_report.json"
    with report_path.open(encoding="utf-8") as handle:
        methods = (json.load(handle).get("methods") or {})
    for method in CHUNK_METHODS:
        if (
            methods.get(method, {}).get("unit") != "chunk"
            or methods[method].get("coverage_complete") is not True
            or not (output_dir / "coverage" / f"{method}.safetensors").is_file()
        ):
            raise RuntimeError(f"existing {method} chunk coverage is invalid")
    return plan


def _write_manifest(
    *,
    output_dir: Path,
    identity: dict[str, Any],
    feature_ids: dict[str, list[int]],
    coverage: dict[str, Any],
    chunk_plan: dict[str, Any],
) -> None:
    token_sequences = int(
        torch.load(
            output_dir / "token_pool" / "tokens.pt",
            map_location="cpu",
            weights_only=True,
        ).shape[0]
    )
    files = {
        "token_ids": file_record(
            output_dir / "token_pool" / "tokens.pt", relative_to=output_dir
        ),
        "chunk_plan_manifest": file_record(
            output_dir / "chunk_pool" / "plan" / "manifest.json",
            relative_to=output_dir,
        ),
        "budget_analysis": file_record(
            output_dir / "budget_analysis.json", relative_to=output_dir
        ),
        "coverage_report": file_record(
            output_dir / "coverage_report.json", relative_to=output_dir
        ),
    }
    groups = (
        ("token_activation", (output_dir / "token_pool" / "activations").rglob("*.safetensors")),
        ("chunk_plan", (output_dir / "chunk_pool" / "plan").glob("shard-*.safetensors")),
        ("chunk_activation", (output_dir / "chunk_pool" / "activations").glob("shard-*.safetensors")),
        ("coverage", (output_dir / "coverage").glob("*.safetensors")),
    )
    for prefix, paths in groups:
        for index, path in enumerate(sorted(paths)):
            files[f"{prefix}_{index:04d}"] = file_record(
                path, relative_to=output_dir
            )
    write_artifact_manifest(
        {
            "format": DATA_FORMAT,
            "complete": True,
            "identity": identity,
            "feature_ids": feature_ids,
            "pool_layouts": {
                "token": "token_pool/activations/token/rank-*.safetensors",
                "temporal": "token_pool/activations/temporal/rank-*.safetensors",
                "mean": "chunk_pool/activations/shard-*.safetensors:mean_activations",
                "cross": "chunk_pool/activations/shard-*.safetensors:cross_activations",
            },
            "coverage": coverage,
            "token_sequences": token_sequences,
            "chunk_count": int(chunk_plan["total_chunks"]),
            "features_per_method": {
                method: len(ids) for method, ids in feature_ids.items()
            },
            "files": files,
        },
        output_dir / "data_manifest.json",
    )


def main() -> None:
    args = parser().parse_args()
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    bind_local_rank_cpu_affinity(
        local_rank=local_rank,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
    )
    output_dir = Path(args.output_dir)
    chunk_lengths = sorted(set(parse_int_csv(args.chunk_lengths)))
    if chunk_lengths != [32, 64, 128, 256, 512]:
        raise ValueError("chunk lengths must match training: 32,64,128,256,512")
    protocol = AutoInterpConfig(
        context_size=args.token_context_size, buffer=10, seed=args.seed
    )

    if rank == 0:
        sae_set = resolve_sae_artifact_set(
            args.sae_root,
            selection=args.checkpoint_selection,
            modes=METHODS,
        )
        alive_by_method = {
            method: set(
                _alive_feature_ids(
                    Path(sae_set["modes"][method]["checkpoint_path"])
                )
            )
            for method in METHODS
        }
        common_alive = sorted(set.intersection(*alive_by_method.values()))
        if len(common_alive) < args.feature_sample_size:
            raise ValueError(
                f"only {len(common_alive)} feature IDs are alive in all four "
                f"SAEs; cannot sample {args.feature_sample_size}"
            )
        shared_feature_ids = sorted(
            random.Random(args.seed).sample(
                common_alive,
                k=args.feature_sample_size,
            )
        )
        feature_ids = {
            method: list(shared_feature_ids) for method in METHODS
        }
        feature_widths = full_dictionary_feature_widths(sae_set, modes=METHODS)
        if set(feature_widths.values()) != {65_536}:
            raise ValueError(f"expected four 65,536 dictionaries: {feature_widths}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        special_token_ids = sorted(set(map(int, tokenizer.all_special_ids)))
        budget = _budget_report(
            sae_set,
            protocol=protocol,
            chunk_lengths=chunk_lengths,
            family_alpha=args.coverage_family_alpha,
            chosen_token_budget=args.token_total_tokens,
            chosen_chunk_budget=args.chunk_total_tokens,
        )
        if (
            not budget["chosen_budgets_meet_analytic_bounds"]
            and not args.skip_analytic_budget_check
        ):
            raise ValueError(
                "budgets below analytic bounds: "
                f"token>={budget['recommended_token_budget']}, "
                f"chunk>={budget['recommended_chunk_budget']}"
            )
        identity = _identity(
            args,
            sae_set=sae_set,
            feature_ids=feature_ids,
            feature_widths=feature_widths,
            tokenizer=tokenizer,
            chunk_lengths=chunk_lengths,
            budget=budget,
            world_size=world_size,
            protocol=protocol,
        )
        marker = output_dir / "data_manifest.json"
        existing = None
        if (
            marker.exists()
            and not args.overwrite
            and not args.resume_existing_forward
            and not args.reuse_existing_chunk_forward
        ):
            existing = ensure_reusable_artifact(
                marker,
                expected_format=DATA_FORMAT,
                expected_identity=identity,
            )
        if args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        skip = existing is not None
        if args.resume_existing_forward and args.reuse_existing_chunk_forward:
            raise ValueError("resume and reuse-existing-chunk are mutually exclusive")
        if skip:
            chunk_plan = None
        elif args.reuse_existing_chunk_forward:
            chunk_plan = _validate_reusable_chunk_forward(
                output_dir,
                expected_chunk_tokens=args.chunk_total_tokens,
                feature_ids=feature_ids,
            )
            shutil.rmtree(output_dir / "token_pool" / "activations", ignore_errors=True)
            shutil.rmtree(output_dir / "work" / "token_hidden", ignore_errors=True)
            shutil.rmtree(output_dir / "matched_chunk_pool", ignore_errors=True)
            tokens = _tokenize_token_pool(
                Path(args.dataset),
                tokenizer,
                context_size=args.token_context_size,
                total_tokens=args.token_total_tokens,
            )
            (output_dir / "token_pool").mkdir(parents=True, exist_ok=True)
            torch.save(tokens, output_dir / "token_pool" / "tokens.pt")
            atomic_json_dump(budget, output_dir / "budget_analysis.json")
        elif args.resume_existing_forward:
            chunk_plan = _validate_reusable_chunk_forward(
                output_dir,
                expected_chunk_tokens=args.chunk_total_tokens,
                feature_ids=feature_ids,
            )
            hidden = sorted(
                (output_dir / "work" / "token_hidden").glob("rank-*.safetensors")
            )
            acts = {
                method: sorted(
                    (output_dir / "token_pool" / "activations" / method).glob(
                        "rank-*.safetensors"
                    )
                )
                for method in TOKEN_METHODS
            }
            if len(hidden) != world_size or any(
                len(paths) != world_size for paths in acts.values()
            ):
                raise RuntimeError("resume found incomplete token forward outputs")
            atomic_json_dump(budget, output_dir / "budget_analysis.json")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            tokens = _tokenize_token_pool(
                Path(args.dataset),
                tokenizer,
                context_size=args.token_context_size,
                total_tokens=args.token_total_tokens,
            )
            (output_dir / "token_pool").mkdir(parents=True, exist_ok=True)
            torch.save(tokens, output_dir / "token_pool" / "tokens.pt")
            chunk_plan = _build_chunk_plan(
                Path(args.dataset),
                tokenizer,
                output_dir / "chunk_pool" / "plan",
                target_tokens=args.chunk_total_tokens,
                lengths=chunk_lengths,
                seed=args.seed + 17,
                max_document_reuses=args.max_document_reuses,
                tokenizer_batch_size=args.tokenizer_batch_size,
                tokenizer_batch_chars=args.tokenizer_batch_chars,
                shard_tokens=args.chunk_plan_shard_tokens,
            )
            atomic_json_dump(budget, output_dir / "budget_analysis.json")
    else:
        sae_set = feature_ids = identity = skip = chunk_plan = None
        special_token_ids = None

    sae_set = broadcast_object(sae_set, rank=rank)
    feature_ids = broadcast_object(feature_ids, rank=rank)
    identity = broadcast_object(identity, rank=rank)
    special_token_ids = broadcast_object(special_token_ids, rank=rank)
    args.special_token_ids = special_token_ids
    skip = bool(broadcast_object(skip, rank=rank))
    if skip:
        log(f"reusing verified AutoInterp data at {output_dir}", rank=rank, main_only=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return
    if rank != 0:
        with (output_dir / "chunk_pool" / "plan" / "manifest.json").open(
            encoding="utf-8"
        ) as handle:
            chunk_plan = json.load(handle)
    chunk_plan = broadcast_object(chunk_plan if rank == 0 else None, rank=rank)
    if dist.is_initialized():
        dist.barrier()

    if not args.resume_existing_forward:
        extractor = TargetLayerExtractor(
            args.model,
            args.layer,
            f"cuda:{local_rank}",
            dtype=args.model_dtype,
            attn_implementation=args.attn_implementation,
        )
        _run_token_pool(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            sae_set=sae_set,
            feature_ids=feature_ids,
            tokens_path=output_dir / "token_pool" / "tokens.pt",
            output_dir=output_dir,
            extractor=extractor,
        )
        if (
            not args.reuse_existing_chunk_forward
            or args.recompute_chunk_means_for_coverage
        ):
            _run_chunk_pool(
                args,
                rank=rank,
                world_size=world_size,
                sae_set=sae_set,
                feature_ids=feature_ids,
                chunk_plan=chunk_plan,
                plan_dir=output_dir / "chunk_pool" / "plan",
                output_dir=output_dir,
                extractor=extractor,
            )
        extractor.close()
    if dist.is_initialized():
        dist.barrier()

    coverage = _coverage_and_cleanup(
        args,
        output_dir=output_dir,
        sae_set=sae_set,
        protocol=protocol,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        _write_manifest(
            output_dir=output_dir,
            identity=identity,
            feature_ids=feature_ids,
            coverage=coverage,
            chunk_plan=chunk_plan,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
