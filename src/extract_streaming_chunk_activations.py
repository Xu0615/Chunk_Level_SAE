#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist

from chunk_saes.cache import (
    ACTIVATION_CACHE_V2_FORMAT,
    ActivationCacheV2Writer,
    merge_v2_rank_manifests,
)
from chunk_saes.forward import (
    ForwardSchedule,
    process_pairs,
    process_window_by_length,
)
from chunk_saes.hf_corpus import (
    HF_PILE_DATASET,
    HF_PILE_REVISION,
    HF_PILE_SOURCE,
    HFStreamingCorpus,
    iter_hf_prepared_documents,
)
from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.runtime import (
    PerformanceCounters,
    all_gather_objects,
    bind_local_rank_cpu_affinity,
    broadcast_object,
    configure_cpu_threads,
    initialize_distributed,
)
from chunk_saes.sample_plan import (
    PLAN_ROW_HASH_VERSION,
    SAMPLE_PLAN_FORMAT,
    plan_row_hash,
    solve_exact_cell_counts,
)
from chunk_saes.streaming_plan import (
    STREAMING_BOUNDARY_VERSION,
    STREAMING_SAMPLING_VERSION,
    StreamingExactPlanner,
    StreamingRankAllocator,
    iter_tokenized_prepared_documents,
)
from chunk_saes.streaming_dispatch import RowWindowReceiver, send_rows
from chunk_saes.utils import (
    atomic_json_dump,
    model_fingerprint,
    parse_int_csv,
    tokenizer_fingerprint,
)

FORWARD_PIPELINE_VERSION = "hf-central-length-buffered-v3"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stream EleutherAI/the_pile_deduplicated once, select exact no-overlap "
            "Chunk-SAE pairs online, immediately run independent A/B forwards, "
            "and publish a complete activation cache without a sample-plan artifact."
        )
    )
    p.add_argument("--model", required=True)
    p.add_argument("--hf-dataset", default=HF_PILE_DATASET)
    p.add_argument("--hf-dataset-revision", default=HF_PILE_REVISION)
    p.add_argument("--hf-physical-split", default="train")
    p.add_argument(
        "--logical-split",
        choices=("train", "validation", "test"),
        required=True,
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--document-shuffle-seed", type=int, default=71)
    p.add_argument("--stream-shuffle-buffer", type=int, default=16_384)
    p.add_argument("--corpus-source", default=HF_PILE_SOURCE)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--target-tokens", required=True, type=int)
    p.add_argument("--chunk-lengths", default="32,64,128,256,512")
    p.add_argument("--sample-seed", type=int, default=43)
    p.add_argument("--max-document-reuses", type=int, default=64)
    p.add_argument("--tokenizer-threads", type=int, default=16)
    p.add_argument("--tokenizer-batch-size", type=int, default=1024)
    p.add_argument("--tokenizer-batch-chars", type=int, default=8_000_000)
    p.add_argument("--forward-batch-size", type=int, default=64)
    p.add_argument("--forward-token-budget", type=int, default=65_536)
    p.add_argument("--writer-batch-tokens", type=int, default=131_072)
    p.add_argument(
        "--forward-scheduler",
        choices=("length", "cell"),
        default="length",
    )
    p.add_argument(
        "--planner-mode",
        choices=("central", "replicated"),
        default="central",
    )
    p.add_argument("--dispatch-queue-depth", type=int, default=4)
    p.add_argument("--scheduler-window-pairs", type=int, default=8192)
    p.add_argument("--async-write-batches", type=int, default=2)
    p.add_argument("--shard-write-workers", type=int, default=2)
    p.add_argument("--max-pending-shards", type=int, default=4)
    p.add_argument("--cache-shard-tokens", type=int, default=131_072)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument(
        "--activation-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--progress-every-documents", type=int, default=100_000)
    p.add_argument("--verify-cache-payloads", action="store_true")
    p.add_argument("--verify-cache-activation-rows", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _activation_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _broadcast(value, *, rank: int):
    return broadcast_object(value, rank=rank)


def _all_gather(value, *, world_size: int):
    return all_gather_objects(value, world_size=world_size)


def _cache_matches(
    cache_dir: Path,
    *,
    target_tokens: int,
    model: str,
    layer: int,
    corpus_identity_digest: str,
) -> bool:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(
        (
            manifest.get("format") == ACTIVATION_CACHE_V2_FORMAT,
            manifest.get("complete") is True,
            manifest.get("target_token_occurrences") == target_tokens,
            manifest.get("token_occurrences") == target_tokens,
            manifest.get("model") == model,
            manifest.get("layer") == layer,
            manifest.get("streaming_sampling") == STREAMING_SAMPLING_VERSION,
            manifest.get("forward_pipeline_version")
            == FORWARD_PIPELINE_VERSION,
            manifest.get("corpus_identity_digest") == corpus_identity_digest,
        )
    )


def _forward_limit(args: argparse.Namespace, key: tuple[int, int]) -> int:
    if args.forward_token_budget <= 0:
        return args.forward_batch_size
    result = max(1, args.forward_token_budget // max(key))
    if args.forward_batch_size > 0:
        result = min(result, args.forward_batch_size)
    return result


def _flush_window(
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    window: list[tuple[object, int]],
    *,
    args: argparse.Namespace,
    expected_order: int,
) -> dict[str, float | int]:
    if not window:
        return {}
    if args.forward_scheduler == "length":
        stats = process_window_by_length(
            extractor,
            writer,
            window,
            schedule=ForwardSchedule(
                max_batch_size=args.forward_batch_size,
                token_budget=args.forward_token_budget,
                writer_batch_tokens=args.writer_batch_tokens,
            ),
        ).as_dict()
    else:
        from collections import defaultdict

        grouped: dict[tuple[int, int], list[tuple[object, int]]] = defaultdict(list)
        for row, order_index in window:
            grouped[(row.length_a, row.length_b)].append((row, order_index))
        calls = 0
        for key, entries in grouped.items():
            limit = _forward_limit(args, key)
            for start in range(0, len(entries), limit):
                batch = entries[start : start + limit]
                process_pairs(
                    extractor,
                    writer,
                    [item[0] for item in batch],
                    [item[1] for item in batch],
                )
                calls += 2
        writer.wait_until_order(expected_order)
        stats = {
            "pairs": len(window),
            "tokens": sum(item[0].token_count for item in window),
            "model_calls": calls,
        }
    window.clear()
    return stats


def _build_configuration(
    args: argparse.Namespace,
    *,
    corpus: HFStreamingCorpus,
    tokenizer_hash: str,
    world_size: int,
) -> tuple[dict, dict, list[str], dict[str, int]]:
    sources = [corpus.source]
    source_weights = {corpus.source: 1}
    lengths = sorted(set(parse_int_csv(args.chunk_lengths)))
    counts = solve_exact_cell_counts(
        args.target_tokens,
        sources,
        lengths,
        seed=args.sample_seed,
    )
    corpus_identity = corpus.identity()
    corpus_identity_digest = hashlib.sha256(
        _canonical_json(corpus_identity)
    ).hexdigest()
    identity = {
        "format_version": "chunk-saes-streaming-universe",
        "target_tokens": int(args.target_tokens),
        "sample_seed": int(args.sample_seed),
        "tokenizer_hash": tokenizer_hash,
        "corpus_identity": corpus_identity,
        "corpus_identity_digest": corpus_identity_digest,
        "sources": sources,
        "chunk_lengths": lengths,
        "document_priority": corpus_identity["shuffle"],
        "boundary_sampling": STREAMING_BOUNDARY_VERSION,
        "row_hash": PLAN_ROW_HASH_VERSION,
        "source_weighting": "single_deduplicated_stream",
        "source_weight_basis": "EleutherAI/the_pile_deduplicated_stream",
        "source_token_weights": source_weights,
        "length_pair_balanced": True,
        "execution_world_size": int(world_size),
        "streaming_sampling": STREAMING_SAMPLING_VERSION,
        "forward_fused": True,
        "planner_execution": args.planner_mode,
        "forward_pipeline_version": FORWARD_PIPELINE_VERSION,
    }
    return identity, counts, sources, source_weights


def _forward_row_window(
    rows: list,
    *,
    local_order: int,
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    args: argparse.Namespace,
    performance: PerformanceCounters,
) -> tuple[int, int]:
    if not rows:
        return local_order, 0
    entries = [
        (row, local_order + index)
        for index, row in enumerate(rows)
    ]
    local_order += len(rows)
    with performance.timer("forward_window"):
        stats = _flush_window(
            extractor,
            writer,
            entries,
            args=args,
            expected_order=local_order,
        )
    return local_order, int(stats.get("model_calls", 0))


def _run_replicated_planner(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    corpus: HFStreamingCorpus,
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    planner: StreamingExactPlanner,
    expected_pairs: int,
) -> dict:
    rows_digest = hashlib.sha256()
    local_order = 0
    window: list = []
    last_progress = 0
    performance = PerformanceCounters()
    forward_windows = 0
    model_calls = 0
    documents = iter_hf_prepared_documents(corpus)
    tokenized_documents = iter_tokenized_prepared_documents(
        documents,
        extractor.tokenizer,
        batch_size=args.tokenizer_batch_size,
        batch_chars=args.tokenizer_batch_chars,
    )
    try:
        for document, token_ids in tokenized_documents:
            for row in planner.add_document(document, token_ids):
                rows_digest.update(plan_row_hash(row))
                if row.execution_rank == rank:
                    window.append(row)
                    if len(window) >= args.scheduler_window_pairs:
                        local_order, calls = _forward_row_window(
                            window,
                            local_order=local_order,
                            extractor=extractor,
                            writer=writer,
                            args=args,
                            performance=performance,
                        )
                        model_calls += calls
                        forward_windows += 1
                        window = []
            if (
                rank == 0
                and args.progress_every_documents > 0
                and planner.documents_scanned - last_progress
                >= args.progress_every_documents
            ):
                last_progress = planner.documents_scanned
                print(
                    "[streaming-forward] "
                    f"documents={planner.documents_scanned} "
                    f"candidate_tokens={planner.candidate_tokens} "
                    f"selected_pairs={planner.selected_pairs}/{expected_pairs} "
                    f"selected_tokens={planner.selected_tokens}/{args.target_tokens} "
                    f"rank0_forward_pairs={local_order}",
                    flush=True,
                )
            if planner.complete:
                break
    finally:
        tokenized_documents.close()
        documents.close()
    planner_stats = planner.finish(
        target_tokens=args.target_tokens,
        expected_pairs=expected_pairs,
    )
    if window:
        local_order, calls = _forward_row_window(
            window,
            local_order=local_order,
            extractor=extractor,
            writer=writer,
            args=args,
            performance=performance,
        )
        model_calls += calls
        forward_windows += 1
    return {
        "rows_digest": rows_digest.hexdigest(),
        "planner_stats": planner_stats,
        "planner_summary": {
            "pairs": planner.selected_pairs,
            "tokens": planner.selected_tokens,
            "documents_scanned": planner.documents_scanned,
            "candidate_tokens": planner.candidate_tokens,
        },
        "local_order": local_order,
        "forward_windows": forward_windows,
        "model_calls": model_calls,
        "runtime": performance.summary(),
    }


def _run_central_planner(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    corpus: HFStreamingCorpus,
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    planner: StreamingExactPlanner | None,
    expected_pairs: int,
    source_to_id: dict[str, int],
    dispatch_group,
) -> dict:
    performance = PerformanceCounters()
    local_order = 0
    forward_windows = 0
    model_calls = 0
    logical_payload = None
    if rank == 0:
        assert planner is not None
        rows_digest = hashlib.sha256()
        buffers: dict[int, list] = {peer: [] for peer in range(world_size)}
        last_progress = 0
        documents = iter_hf_prepared_documents(corpus)
        tokenized_documents = iter_tokenized_prepared_documents(
            documents,
            extractor.tokenizer,
            batch_size=args.tokenizer_batch_size,
            batch_chars=args.tokenizer_batch_chars,
        )
        try:
            for document, token_ids in tokenized_documents:
                for row in planner.add_document(document, token_ids):
                    rows_digest.update(plan_row_hash(row))
                    destination = int(row.execution_rank)
                    buffers[destination].append(row)
                    if len(buffers[destination]) >= args.scheduler_window_pairs:
                        ready = buffers[destination]
                        buffers[destination] = []
                        if destination == 0:
                            local_order, calls = _forward_row_window(
                                ready,
                                local_order=local_order,
                                extractor=extractor,
                                writer=writer,
                                args=args,
                                performance=performance,
                            )
                            model_calls += calls
                            forward_windows += 1
                        else:
                            with performance.timer("dispatch_send"):
                                send_rows(
                                    ready,
                                    destination=destination,
                                    group=dispatch_group,
                                    source_to_id=source_to_id,
                                )
                if (
                    args.progress_every_documents > 0
                    and planner.documents_scanned - last_progress
                    >= args.progress_every_documents
                ):
                    last_progress = planner.documents_scanned
                    print(
                        "[streaming-forward-central] "
                        f"documents={planner.documents_scanned} "
                        f"candidate_tokens={planner.candidate_tokens} "
                        f"selected_pairs={planner.selected_pairs}/{expected_pairs} "
                        f"selected_tokens={planner.selected_tokens}/{args.target_tokens}",
                        flush=True,
                    )
                if planner.complete:
                    break
        finally:
            tokenized_documents.close()
            documents.close()
        planner_stats = planner.finish(
            target_tokens=args.target_tokens,
            expected_pairs=expected_pairs,
        )
        for destination in range(world_size):
            ready = buffers[destination]
            if ready:
                if destination == 0:
                    local_order, calls = _forward_row_window(
                        ready,
                        local_order=local_order,
                        extractor=extractor,
                        writer=writer,
                        args=args,
                        performance=performance,
                    )
                    model_calls += calls
                    forward_windows += 1
                else:
                    send_rows(
                        ready,
                        destination=destination,
                        group=dispatch_group,
                        source_to_id=source_to_id,
                    )
            if destination:
                send_rows(
                    None,
                    destination=destination,
                    group=dispatch_group,
                    source_to_id=source_to_id,
                )
        logical_payload = {
            "rows_digest": rows_digest.hexdigest(),
            "planner_stats": planner_stats,
            "planner_summary": {
                "pairs": planner.selected_pairs,
                "tokens": planner.selected_tokens,
                "documents_scanned": planner.documents_scanned,
                "candidate_tokens": planner.candidate_tokens,
            },
        }
    else:
        receiver = RowWindowReceiver(
            source=0,
            group=dispatch_group,
            id_to_source=[
                source
                for source, _index in sorted(
                    source_to_id.items(), key=lambda item: item[1]
                )
            ],
            queue_depth=args.dispatch_queue_depth,
        )
        for rows in receiver:
            local_order, calls = _forward_row_window(
                rows,
                local_order=local_order,
                extractor=extractor,
                writer=writer,
                args=args,
                performance=performance,
            )
            model_calls += calls
            forward_windows += 1
    logical_payload = _broadcast(logical_payload, rank=rank)
    if not isinstance(logical_payload, dict):
        raise RuntimeError("central planner failed to publish its logical result")
    return {
        **logical_payload,
        "local_order": local_order,
        "forward_windows": forward_windows,
        "model_calls": model_calls,
        "runtime": performance.summary(),
    }


def main() -> None:
    args = parser().parse_args()
    if args.tokenizer_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    configure_cpu_threads(default=4)
    rank, world_size, local_rank = initialize_distributed()
    bind_local_rank_cpu_affinity(
        local_rank=local_rank,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
    )
    cache_dir = Path(args.cache_dir)
    if args.target_tokens <= 0 or args.target_tokens % world_size:
        raise ValueError("target tokens must be positive and divisible by world size")
    if args.scheduler_window_pairs <= 0:
        raise ValueError("scheduler_window_pairs must be positive")
    corpus = HFStreamingCorpus(
        dataset=args.hf_dataset,
        revision=args.hf_dataset_revision,
        physical_split=args.hf_physical_split,
        logical_split=args.logical_split,
        split_seed=args.split_seed,
        shuffle_seed=args.document_shuffle_seed,
        shuffle_buffer=args.stream_shuffle_buffer,
        source=args.corpus_source,
    )
    corpus_identity_digest = hashlib.sha256(
        _canonical_json(corpus.identity())
    ).hexdigest()

    skip = _cache_matches(
        cache_dir,
        target_tokens=args.target_tokens,
        model=args.model,
        layer=args.layer,
        corpus_identity_digest=corpus_identity_digest,
    )
    skip = bool(_broadcast(skip if rank == 0 else None, rank=rank))
    if skip and not args.overwrite:
        if rank == 0:
            print(f"[streaming-forward] complete cache exists: {cache_dir}", flush=True)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    tokenizer_hash = tokenizer_fingerprint(args.model)
    identity, counts, sources, source_weights = _build_configuration(
        args,
        corpus=corpus,
        tokenizer_hash=tokenizer_hash,
        world_size=world_size,
    )
    expected_pairs = sum(counts.values())
    source_tokens: Counter[str] = Counter()
    for (source, length_a, length_b), count in counts.items():
        source_tokens[source] += (length_a + length_b) * count
    requested_cell_counts = {
        f"{source}\t{length_a}\t{length_b}": int(count)
        for (source, length_a, length_b), count in sorted(counts.items())
    }
    sampling_identity_digest = hashlib.sha256(_canonical_json(identity)).hexdigest()

    if rank == 0:
        if args.overwrite and cache_dir.exists():
            shutil.rmtree(cache_dir)
        else:
            (cache_dir / "manifest.json").unlink(missing_ok=True)
            (cache_dir / "streaming_universe.json").unlink(missing_ok=True)
    if dist.is_initialized():
        dist.barrier()
    rank_dir = cache_dir / f"rank{rank:03d}"
    if rank_dir.exists():
        shutil.rmtree(rank_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        expected_model_hash = model_fingerprint(args.model)
    else:
        expected_model_hash = None
    expected_model_hash = _broadcast(expected_model_hash, rank=rank)
    if not isinstance(expected_model_hash, str):
        raise RuntimeError("Failed to compute/broadcast model fingerprint")

    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        f"cuda:{local_rank}",
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    writer = ActivationCacheV2Writer(
        cache_dir,
        rank=rank,
        world_size=world_size,
        hidden_size=extractor.hidden_size,
        plan_digest=sampling_identity_digest,
        target_token_occurrences=args.target_tokens,
        shard_token_limit=args.cache_shard_tokens,
        source_to_id={source: index for index, source in enumerate(sources)},
        activation_dtype=_activation_dtype(args.activation_dtype),
        async_write_batches=args.async_write_batches,
        shard_write_workers=args.shard_write_workers,
        max_pending_shards=args.max_pending_shards,
    )
    planner = None
    if args.planner_mode == "replicated" or rank == 0:
        rank_allocator = StreamingRankAllocator(
            counts,
            world_size=world_size,
            target_tokens=args.target_tokens,
            seed=args.sample_seed,
        )
        planner = StreamingExactPlanner(
            counts,
            seed=args.sample_seed,
            max_document_reuses=args.max_document_reuses,
            rank_allocator=rank_allocator,
        )
    dispatch_group = None
    if args.planner_mode == "central" and world_size > 1:
        dispatch_group = dist.new_group(backend="gloo")

    rank_manifest = None
    execution = None
    try:
        if args.planner_mode == "central" and world_size > 1:
            execution = _run_central_planner(
                args=args,
                rank=rank,
                world_size=world_size,
                corpus=corpus,
                extractor=extractor,
                writer=writer,
                planner=planner,
                expected_pairs=expected_pairs,
                source_to_id={
                    source: index for index, source in enumerate(sources)
                },
                dispatch_group=dispatch_group,
            )
        else:
            assert planner is not None
            execution = _run_replicated_planner(
                args=args,
                rank=rank,
                world_size=world_size,
                corpus=corpus,
                extractor=extractor,
                writer=writer,
                planner=planner,
                expected_pairs=expected_pairs,
            )
        rank_manifest = writer.finish()
        expected_rank_tokens = args.target_tokens // world_size
        if int(rank_manifest["token_occurrences"]) != expected_rank_tokens:
            raise RuntimeError(
                f"rank {rank} wrote {rank_manifest['token_occurrences']} tokens, "
                f"expected {expected_rank_tokens}"
            )
    finally:
        writer.close()
        extractor.close()

    assert execution is not None
    rows_digest_value = str(execution["rows_digest"])
    planner_stats = execution["planner_stats"]
    planner_summary = execution["planner_summary"]
    gathered = _all_gather(
        {
            "rows_digest": rows_digest_value,
            "pairs": int(planner_summary["pairs"]),
            "tokens": int(planner_summary["tokens"]),
            "documents_scanned": int(planner_summary["documents_scanned"]),
            "candidate_tokens": int(planner_summary["candidate_tokens"]),
            "forward_windows": int(execution["forward_windows"]),
            "model_calls": int(execution["model_calls"]),
            "runtime": execution["runtime"],
        },
        world_size=world_size,
    )
    if len({item["rows_digest"] for item in gathered}) != 1:
        raise RuntimeError("Streaming ranks produced different logical row universes")
    if any(
        item["pairs"] != expected_pairs or item["tokens"] != args.target_tokens
        for item in gathered
    ):
        raise RuntimeError("Streaming rank planners disagree on exact coverage")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        assert planner_stats is not None
        streaming_universe_digest = hashlib.sha256(
            _canonical_json(identity) + bytes.fromhex(rows_digest_value)
        ).hexdigest()

        universe_manifest = {
            "format": SAMPLE_PLAN_FORMAT,
            "complete": True,
            "identity": identity,
            # Direct streaming has no standalone plan artifact. The legacy
            # plan_digest slot consistently carries the pre-forward sampling
            # identity in shard/rank/global metadata, while rows_digest and
            # streaming_universe_digest bind the realized row universe.
            "plan_digest": sampling_identity_digest,
            "rows_digest": rows_digest_value,
            "streaming_universe_digest": streaming_universe_digest,
            "target_token_occurrences": int(args.target_tokens),
            "token_occurrences": int(args.target_tokens),
            "pairs": int(expected_pairs),
            "rank_independent": True,
            "occurrence_ids": {
                "start": 0,
                "stop": int(args.target_tokens),
                "contiguous": True,
            },
            "source_pairs": {
                source: sum(
                    count
                    for (cell_source, _a, _b), count in counts.items()
                    if cell_source == source
                )
                for source in sources
            },
            "source_tokens": dict(sorted(source_tokens.items())),
            "cell_counts": requested_cell_counts,
            "requested_cell_counts": requested_cell_counts,
            "source_weighting": "single_deduplicated_stream",
            "source_weight_basis": "EleutherAI/the_pile_deduplicated_stream",
            "source_token_weights": source_weights,
            "max_document_reuses": int(args.max_document_reuses),
            "corpus_position_overlap_policy": "forbidden",
            "corpus_position_overlap_verified": True,
            "unique_corpus_token_positions": int(args.target_tokens),
            "document_sampling": {
                key: value
                for key, value in planner_stats.items()
                if key != "execution_rank_assignment"
            },
            "execution_rank_assignment": planner_stats[
                "execution_rank_assignment"
            ],
            "input_provenance": {
                **corpus.identity(),
                "corpus_identity_digest": corpus_identity_digest,
            },
            "shards": [],
        }
        atomic_json_dump(
            universe_manifest,
            cache_dir / "streaming_universe.json",
        )
        merged = merge_v2_rank_manifests(
            cache_dir,
            world_size=world_size,
            plan_manifest=universe_manifest,
            verify_payload_checksums=args.verify_cache_payloads,
            verify_activation_rows=args.verify_cache_activation_rows,
            extra={
                "project": "chunk-saes",
                "model": args.model,
                "model_hash": expected_model_hash,
                "layer": args.layer,
                "tokenizer_hash": tokenizer_hash,
                "hf_corpus": corpus.identity(),
                "corpus_identity_digest": corpus_identity_digest,
                "streaming_sampling": STREAMING_SAMPLING_VERSION,
                "streaming_forward_fused": True,
                "forward_pipeline_version": FORWARD_PIPELINE_VERSION,
                "planner_execution": args.planner_mode,
                "sampling_identity_digest": sampling_identity_digest,
                "streaming_universe_digest": streaming_universe_digest,
                "independent_forwards": True,
                "padding_side": "right",
                "padding_rows_stored": False,
                "position_policy": "reset_to_zero_per_independent_chunk",
                "activation_dtype": args.activation_dtype,
                "forward_batching": {
                    "scheduler": args.forward_scheduler,
                    "max_chunk_batch_size": args.forward_batch_size,
                    "token_budget_per_independent_forward": (
                        args.forward_token_budget
                    ),
                    "scheduler_window_pairs": args.scheduler_window_pairs,
                    "writer_batch_tokens": args.writer_batch_tokens,
                },
                "forward_runtime_by_rank": gathered,
                "candidate_qwen_tokens": planner.candidate_tokens,
                "prepared_documents_scanned": planner.documents_scanned,
                "candidate_to_selected_token_ratio": (
                    planner.candidate_tokens / args.target_tokens
                ),
            },
        )
        print(
            "[streaming-forward] "
            + json.dumps(
                {
                    "cache_dir": str(cache_dir),
                    "sampling_identity_digest": sampling_identity_digest,
                    "streaming_universe_digest": streaming_universe_digest,
                    "activation_digest": merged["activation_digest"],
                    "pairs": merged["pairs"],
                    "token_occurrences": merged["token_occurrences"],
                    "documents_scanned": planner.documents_scanned,
                    "candidate_qwen_tokens": planner.candidate_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # Remote IterableDataset may retain an aiohttp/fsspec helper thread whose
    # Python-3.12 interpreter-finalization hook is unsafe. All CUDA, writer and
    # process-group resources are explicitly closed by main().
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
