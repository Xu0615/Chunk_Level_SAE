#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from chunk_saes.artifacts import (
    ensure_reusable_artifact,
    file_record,
    json_digest,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.data import tokenize_document
from chunk_saes.evaluation_protocol import (
    fixed_chunk_protocol_metadata,
    full_dictionary_feature_widths,
    mean_after_threshold,
)
from chunk_saes.hf_corpus import (
    HF_PILE_DATASET,
    HF_PILE_REVISION,
    HF_PILE_SOURCE,
    HFStreamingCorpus,
    iter_hf_documents,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.runtime import (
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.utils import (
    append_jsonl,
    atomic_json_dump,
    content_hash,
    document_split,
    log,
    parse_int_csv,
)


EVIDENCE_FORMAT = "chunk-saes-feature-evidence-v2"


class FeatureScorer:
    def __init__(self, checkpoint_dir: Path, sample_size: int, seed: int, device: str) -> None:
        with (checkpoint_dir / "config.json").open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        weights_path = checkpoint_dir / "sae.safetensors"
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            tensor_names = set(handle.keys())
            counts = handle.get_tensor("feature_counts")
            self.dictionary_width = int(counts.numel())
            encoder_shape = handle.get_slice("encoder_weight").get_shape()
            if int(encoder_shape[0]) != self.dictionary_width:
                raise ValueError(
                    f"{checkpoint_dir} encoder width {encoder_shape[0]} does not "
                    f"match full dictionary width {self.dictionary_width}"
                )
            alive = torch.nonzero(
                counts > 0,
                as_tuple=False,
            ).flatten()
            if len(alive) == 0:
                raise ValueError(f"No alive features in {checkpoint_dir}")
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(alive), generator=generator)
            self.feature_ids = alive[permutation[: min(sample_size, len(alive))]].sort().values
            self.encoder_weight = handle.get_tensor("encoder_weight")[self.feature_ids].to(device)
            self.encoder_bias = handle.get_tensor("encoder_bias")[self.feature_ids].to(device)
            self.decoder_bias = handle.get_tensor("decoder_bias").to(device)
            if "pre_bias" in tensor_names:
                self.pre_bias = handle.get_tensor("pre_bias").to(device)
            elif (
                self.config.get("sae_parameter_schema_version")
                == SAE_PARAMETER_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{checkpoint_dir} declares {SAE_PARAMETER_SCHEMA_VERSION} "
                    "but sae.safetensors lacks pre_bias"
                )
            else:
                # Legacy checkpoints used decoder_bias for both input
                # centering and decoder output bias.
                self.pre_bias = self.decoder_bias
            self.threshold = float(handle.get_tensor("threshold"))
            self.scale = float(handle.get_tensor("activation_scale"))

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.encoder_weight.dtype)
        scaled = hidden * self.scale
        pre = F.relu(
            F.linear(scaled - self.pre_bias, self.encoder_weight, self.encoder_bias)
        )
        return pre * (pre > self.threshold)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build matched, blinded top-chunk feature evidence.")
    p.add_argument("--model", required=True)
    p.add_argument("--hf-dataset", default=HF_PILE_DATASET)
    p.add_argument("--hf-dataset-revision", default=HF_PILE_REVISION)
    p.add_argument("--hf-physical-split", default="train")
    p.add_argument("--document-shuffle-seed", type=int, default=71)
    p.add_argument("--stream-shuffle-buffer", type=int, default=16_384)
    p.add_argument("--corpus-source", default=HF_PILE_SOURCE)
    p.add_argument("--sae-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=52)
    p.add_argument("--chunk-lengths", default="32,64,128,256,512")
    p.add_argument("--evidence-chunks", type=int, default=20000)
    p.add_argument("--feature-sample-size", type=int, default=1000)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-passes", type=int, default=100)
    p.add_argument("--chunks-per-document", type=int, default=8)
    p.add_argument("--checkpoint-selection", choices=("best", "final"), default="best")
    p.add_argument("--overwrite", action="store_true")
    return p


def score_batch(extractor, scorers, chunks, all_scores) -> None:
    layer_batch = extractor.forward_ids([chunk["input_ids"] for chunk in chunks])
    means = layer_batch.means()
    valid_hidden = layer_batch.hidden
    mask = layer_batch.mask.bool()
    for mode, scorer in scorers.items():
        if mode in {"token", "temporal"}:
            batch, length, hidden = valid_hidden.shape
            scores = scorer.scores(valid_hidden.reshape(batch * length, hidden))
            scores = scores.reshape(batch, length, -1)
            scores = mean_after_threshold(scores, mask)
        else:
            scores = scorer.scores(means)
        all_scores[mode].append(scores.float().cpu())


def merge_outputs(
    args,
    world_size: int,
    scorers: dict[str, FeatureScorer],
    output_dir: Path,
    identity: Mapping[str, object],
) -> None:
    chunks: dict[int, dict] = {}
    for rank in range(world_size):
        path = output_dir / "partials" / f"chunks-rank{rank:03d}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                global_id = int(row["global_chunk_id"])
                if global_id in chunks:
                    raise ValueError(f"duplicate global_chunk_id={global_id}")
                if not isinstance(row.get("input_ids"), list) or len(
                    row["input_ids"]
                ) != int(row["length"]):
                    raise ValueError(
                        f"chunk {global_id} lacks exact fixed-length input_ids"
                    )
                chunks[global_id] = row

    unblinded = []
    for mode, scorer in scorers.items():
        rank_scores, rank_indices = [], []
        for rank in range(world_size):
            path = output_dir / "partials" / f"candidates-rank{rank:03d}.safetensors"
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                rank_scores.append(handle.get_tensor(f"{mode}_scores"))
                rank_indices.append(handle.get_tensor(f"{mode}_indices"))
        scores = torch.cat(rank_scores, dim=1)
        indices = torch.cat(rank_indices, dim=1)
        top = scores.topk(min(args.top_n, scores.shape[1]), dim=1)
        selected_indices = indices.gather(1, top.indices)
        for row_index, feature_id in enumerate(scorer.feature_ids.tolist()):
            examples = []
            for score, chunk_id in zip(top.values[row_index].tolist(), selected_indices[row_index].tolist()):
                chunk = chunks[int(chunk_id)]
                examples.append(
                    {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "length": chunk["length"],
                        "score": score,
                    }
                )
            unblinded.append({"method": mode, "feature_id": feature_id, "examples": examples})

    random.Random(args.sample_seed + 991).shuffle(unblinded)
    blinded_rows, blind_map = [], {}
    for index, row in enumerate(unblinded):
        blind_id = f"feature-{index:06d}"
        examples = list(row["examples"])
        random.Random(args.sample_seed + index).shuffle(examples)
        blinded_rows.append(
            {
                "blind_id": blind_id,
                "activating_chunks": [example["text"] for example in examples],
            }
        )
        blind_map[blind_id] = {
            "method": row["method"],
            "feature_id": row["feature_id"],
            "scores": [example["score"] for example in examples],
            "sources": [example["source"] for example in examples],
        }
    evidence_path = output_dir / "evidence_blinded.jsonl"
    if evidence_path.exists():
        evidence_path.unlink()
    append_jsonl(evidence_path, blinded_rows)
    atomic_json_dump(blind_map, output_dir / "blind_map.json")
    partial_files = sorted((output_dir / "partials").glob("*"))
    ordered_chunks = [chunks[index] for index in sorted(chunks)]
    chunk_lengths = sorted({int(chunk["length"]) for chunk in ordered_chunks})
    feature_widths = {
        mode: scorer.dictionary_width for mode, scorer in scorers.items()
    }
    protocol = fixed_chunk_protocol_metadata(
        feature_widths=feature_widths,
        chunk_lengths=chunk_lengths,
    )
    write_artifact_manifest(
        {
            "format": EVIDENCE_FORMAT,
            "complete": True,
            "identity": dict(identity),
            "model": args.model,
            "layer": args.layer,
            "heldout_split": "test",
            "evidence_chunks": len(chunks),
            "features_per_method": {
                mode: len(scorer.feature_ids) for mode, scorer in scorers.items()
            },
            "feature_widths": feature_widths,
            "top_n": args.top_n,
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
            "chunks_per_document_per_pass": args.chunks_per_document,
            "judge_visible_fields": ["blind_id", "activating_chunks"],
            "representation_protocol": protocol,
            "candidate_pool": {
                "shared_across_methods": True,
                "global_chunk_ids_digest": json_digest(
                    [int(chunk["global_chunk_id"]) for chunk in ordered_chunks]
                ),
                "input_ids_digest": json_digest(
                    [chunk["input_ids"] for chunk in ordered_chunks]
                ),
                "counts_by_length": {
                    str(length): sum(
                        int(chunk["length"]) == length for chunk in ordered_chunks
                    )
                    for length in chunk_lengths
                },
            },
            "token_sae_chunk_score": "mean_after_threshold",
            "complete_chunks": True,
            "files": {
                "evidence": file_record(evidence_path, relative_to=output_dir),
                "blind_map": file_record(
                    output_dir / "blind_map.json", relative_to=output_dir
                ),
                **{
                    f"partial_{index:03d}": file_record(path, relative_to=output_dir)
                    for index, path in enumerate(partial_files)
                },
            },
        },
        output_dir / "evidence_manifest.json",
    )


def _broadcast(value, rank: int):
    return broadcast_object(value, rank=rank)


def main() -> None:
    args = parser().parse_args()
    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    bind_local_rank_cpu_affinity(
        local_rank=local_rank,
        local_world_size=int(
            os.environ.get("LOCAL_WORLD_SIZE", world_size)
        ),
    )
    output_dir = Path(args.output_dir)
    if rank == 0:
        sae_set = resolve_sae_artifact_set(
            args.sae_root, selection=args.checkpoint_selection
        )
        if int(sae_set["common"]["layer"]) != args.layer:
            raise ValueError("SAE checkpoint layer does not match --layer")
        if Path(str(sae_set["common"]["model"])).resolve() != Path(args.model).resolve():
            raise ValueError("SAE checkpoint model does not match --model")
        corpus = HFStreamingCorpus(
            dataset=args.hf_dataset,
            revision=args.hf_dataset_revision,
            physical_split=args.hf_physical_split,
            logical_split="test",
            split_seed=args.split_seed,
            shuffle_seed=args.document_shuffle_seed,
            shuffle_buffer=args.stream_shuffle_buffer,
            source=args.corpus_source,
        )
        representation_protocol = fixed_chunk_protocol_metadata(
            feature_widths=full_dictionary_feature_widths(sae_set),
            chunk_lengths=parse_int_csv(args.chunk_lengths),
        )
        identity = {
            "sae_set_digest": sae_set["artifact_digest"],
            "sae_selection": args.checkpoint_selection,
            "sae_modes": sae_set["modes"],
            "model": str(Path(args.model).resolve()),
            "layer": args.layer,
            "corpus": corpus.identity(),
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
            "chunk_lengths": parse_int_csv(args.chunk_lengths),
            "evidence_chunks": args.evidence_chunks,
            "feature_sample_size": args.feature_sample_size,
            "top_n": args.top_n,
            "chunks_per_document": args.chunks_per_document,
            "max_passes": args.max_passes,
            "representation_protocol": representation_protocol,
            "scoring": {
                mode: {
                    "token": "mean_after_thresholded_token_activation",
                    "temporal": (
                        "mean_after_thresholded_token_activation_full_dictionary"
                    ),
                    "mean": "chunk_mean_activation",
                    "cross": "chunk_mean_activation",
                }[mode]
                for mode in sae_set["modes"]
            },
            "feature_sampling_scope": {
                mode: "full_training_alive_dictionary"
                for mode in sae_set["modes"]
            },
        }
        existing = None
        marker = output_dir / "evidence_manifest.json"
        if marker.exists() and not args.overwrite:
            existing = ensure_reusable_artifact(
                marker,
                expected_format=EVIDENCE_FORMAT,
                expected_identity=identity,
            )
        if args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        skip = existing is not None
    else:
        sae_set = None
        corpus = None
        identity = None
        skip = None
    sae_set = _broadcast(sae_set, rank)
    corpus = _broadcast(corpus, rank)
    identity = _broadcast(identity, rank)
    skip = bool(_broadcast(skip, rank))
    if skip:
        log(f"reusing verified feature evidence at {output_dir}", rank=rank, main_only=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return
    if dist.is_initialized():
        dist.barrier()
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    scorer_seed = args.sample_seed + 101
    scorers = {
        mode: FeatureScorer(
            Path(sae_set["modes"][mode]["checkpoint_path"]),
            args.feature_sample_size,
            scorer_seed,
            f"cuda:{local_rank}",
        )
        for mode in sae_set["modes"]
    }
    expected_feature_widths = full_dictionary_feature_widths(sae_set)
    for mode, scorer in scorers.items():
        if scorer.dictionary_width != expected_feature_widths[mode]:
            raise ValueError(
                f"{mode} checkpoint tensor width {scorer.dictionary_width} does "
                f"not match configured width {expected_feature_widths[mode]}"
            )
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        f"cuda:{local_rank}",
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    lengths = parse_int_csv(args.chunk_lengths)
    cells = [
        (args.corpus_source, length)
        for length in lengths
    ]
    shuffled_cells = list(cells)
    random.Random(args.sample_seed).shuffle(shuffled_cells)
    base, remainder = divmod(args.evidence_chunks, len(cells))
    global_quotas = {cell: base + int(cell in set(shuffled_cells[:remainder])) for cell in cells}
    quotas = {
        cell: global_quotas[cell] // world_size + int(rank < global_quotas[cell] % world_size)
        for cell in cells
    }
    rank_target = sum(quotas.values())
    used = defaultdict(int)
    generator = torch.Generator().manual_seed(args.sample_seed + rank * 1_000_003)
    chunks, pending = [], []
    all_scores: dict[str, list[torch.Tensor]] = defaultdict(list)

    document_iterator = lambda: iter_hf_documents(
        corpus,
        rank=rank,
        world_size=world_size,
    )
    for pass_index in range(args.max_passes):
        before = len(chunks)
        current_documents = document_iterator()
        try:
            for document in current_documents:
                token_ids = tokenize_document(extractor.tokenizer, document.text)
                for _ in range(args.chunks_per_document):
                    eligible = [
                        cell for cell in cells
                        if cell[0] == document.source
                        and used[cell] < quotas[cell]
                        and len(token_ids) >= cell[1]
                    ]
                    if not eligible:
                        break
                    cell = eligible[int(torch.randint(len(eligible), (), generator=generator))]
                    length = cell[1]
                    start = int(torch.randint(len(token_ids) - length + 1, (), generator=generator))
                    selected = token_ids[start : start + length]
                    local_id = len(chunks)
                    global_id = (rank << 32) | local_id
                    chunk = {
                        "global_chunk_id": global_id,
                        "doc_id": document.doc_id,
                        "source": document.source,
                        "length": length,
                        "text": extractor.tokenizer.decode(selected, skip_special_tokens=True),
                        "input_ids": [int(value) for value in selected],
                    }
                    if document_split(document.text_hash, args.split_seed) != "test":
                        raise ValueError("feature evidence document is not in the test split")
                    if content_hash(document.text) != document.text_hash:
                        raise ValueError("feature evidence document content hash is invalid")
                    chunks.append(chunk)
                    pending.append(chunk)
                    used[cell] += 1
                    if len(pending) >= args.forward_batch_size:
                        score_batch(extractor, scorers, pending, all_scores)
                        pending.clear()
                    if len(chunks) >= rank_target:
                        break
                if len(chunks) >= rank_target:
                    break
        finally:
            current_documents.close()
        log(f"evidence pass={pass_index + 1} chunks={len(chunks)}/{rank_target}", rank=rank)
        if len(chunks) >= rank_target:
            break
        if len(chunks) == before:
            raise RuntimeError("No progress while building held-out evidence chunks")
    if pending:
        score_batch(extractor, scorers, pending, all_scores)
    extractor.close()

    chunks_path = partial_dir / f"chunks-rank{rank:03d}.jsonl"
    if chunks_path.exists():
        chunks_path.unlink()
    append_jsonl(chunks_path, chunks)
    candidate_tensors = {}
    for mode, scorer in scorers.items():
        scores = torch.cat(all_scores[mode], dim=0).T
        top = scores.topk(min(args.top_n, scores.shape[1]), dim=1)
        global_ids = torch.tensor([chunk["global_chunk_id"] for chunk in chunks], dtype=torch.int64)
        candidate_tensors[f"{mode}_scores"] = top.values.contiguous()
        candidate_tensors[f"{mode}_indices"] = global_ids[top.indices].contiguous()
        candidate_tensors[f"{mode}_feature_ids"] = scorer.feature_ids.contiguous()
    save_file(candidate_tensors, str(partial_dir / f"candidates-rank{rank:03d}.safetensors"))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        merge_outputs(args, world_size, scorers, output_dir, identity)
        log(f"blinded feature evidence written to {output_dir}", rank=rank)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # See extract_streaming_chunk_activations.py: explicit process exit avoids
    # a known remote IterableDataset helper-thread finalization crash.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
