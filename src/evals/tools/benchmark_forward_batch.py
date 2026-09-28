#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from chunk_saes.modeling import TargetLayerExtractor
from chunk_saes.utils import parse_int_csv


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark target-layer forward batch sizes.")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--batch-sizes", default="32,48,64")
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--measure-iters", type=int, default=3)
    p.add_argument("--model-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reserve-memory-gib", type=float, default=2.0)
    return p


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extractor = TargetLayerExtractor(
        args.model,
        args.layer,
        str(device),
        dtype=args.model_dtype,
        attn_implementation=args.attn_implementation,
    )
    token_id = extractor.tokenizer.eos_token_id
    if token_id is None:
        token_id = 0
    sequence = [int(token_id)] * args.sequence_length
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserve = int(args.reserve_memory_gib * 1024**3)
    results = []

    for batch_size in parse_int_csv(args.batch_sizes):
        sequences = [sequence] * batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for _ in range(args.warmup_iters):
                extractor.forward_ids(sequences)
            torch.cuda.synchronize(device)
            durations = []
            for _ in range(args.measure_iters):
                started = time.perf_counter()
                extractor.forward_ids(sequences)
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - started)
            peak = torch.cuda.max_memory_reserved(device)
            row = {
                "batch_size": batch_size,
                "sequence_length": args.sequence_length,
                "median_seconds": statistics.median(durations),
                "min_seconds": min(durations),
                "sequences_per_second": batch_size / statistics.median(durations),
                "tokens_per_second": batch_size * args.sequence_length / statistics.median(durations),
                "peak_reserved_bytes": peak,
                "peak_reserved_gib": peak / 1024**3,
                "memory_fraction": peak / total_memory,
                "safe_with_reserve": peak <= total_memory - reserve,
                "status": "ok",
            }
        except torch.OutOfMemoryError as exc:
            row = {
                "batch_size": batch_size,
                "sequence_length": args.sequence_length,
                "status": "oom",
                "error": str(exc),
            }
            torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    valid = [row for row in results if row["status"] == "ok" and row["safe_with_reserve"]]
    best = max(valid, key=lambda row: row["tokens_per_second"]) if valid else None
    print(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "device": str(device),
                "best_safe_batch_size": None if best is None else best["batch_size"],
                "best_safe_tokens_per_second": None if best is None else best["tokens_per_second"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    extractor.close()


if __name__ == "__main__":
    main()
