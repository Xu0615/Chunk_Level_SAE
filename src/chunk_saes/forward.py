from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch

from .cache import ActivationCacheV2Writer
from .modeling import TargetLayerExtractor
from .sample_plan import PlanRow


@dataclass(frozen=True)
class ForwardSchedule:
    max_batch_size: int
    token_budget: int
    writer_batch_tokens: int

    def batch_limit(self, sequence_length: int) -> int:
        token_limit = (
            self.max_batch_size
            if self.token_budget <= 0
            else max(1, self.token_budget // int(sequence_length))
        )
        if self.max_batch_size > 0:
            token_limit = min(token_limit, self.max_batch_size)
        return max(1, token_limit)


@dataclass
class ForwardWindowStats:
    pairs: int = 0
    tokens: int = 0
    model_calls: int = 0
    forward_seconds: float = 0.0
    writer_submit_seconds: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "pairs": self.pairs,
            "tokens": self.tokens,
            "model_calls": self.model_calls,
            "forward_seconds": self.forward_seconds,
            "writer_submit_seconds": self.writer_submit_seconds,
        }


def process_pairs(
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    pairs: list[PlanRow],
    order_indices: list[int] | None = None,
) -> None:
    """Compatibility fixed-cell path used by small tests and legacy callers."""

    if not pairs:
        return
    keys = {(pair.length_a, pair.length_b) for pair in pairs}
    if len(keys) != 1:
        start = 0
        while start < len(pairs):
            key = (pairs[start].length_a, pairs[start].length_b)
            stop = start + 1
            while stop < len(pairs) and (
                pairs[stop].length_a,
                pairs[stop].length_b,
            ) == key:
                stop += 1
            process_pairs(
                extractor,
                writer,
                pairs[start:stop],
                None if order_indices is None else order_indices[start:stop],
            )
            start = stop
        return
    batch_a = extractor.forward_ids([pair.input_ids_a for pair in pairs])
    batch_b = extractor.forward_ids([pair.input_ids_b for pair in pairs])
    length_a = pairs[0].length_a
    length_b = pairs[0].length_b
    if batch_a.hidden.shape[1] != length_a or batch_b.hidden.shape[1] != length_b:
        raise ValueError("forward result length differs from the fixed streaming cell")
    packed = torch.cat((batch_a.hidden, batch_b.hidden), dim=1).reshape(
        -1, extractor.hidden_size
    )
    means = torch.stack(
        (
            batch_a.hidden.float().mean(dim=1),
            batch_b.hidden.float().mean(dim=1),
        ),
        dim=1,
    )
    writer.add_batch(pairs, packed, means, order_indices=order_indices)


def _submit_pair_major_batches(
    writer: ActivationCacheV2Writer,
    rows: Sequence[PlanRow],
    order_indices: Sequence[int],
    hidden_a: list[torch.Tensor | None],
    hidden_b: list[torch.Tensor | None],
    mean_a: list[torch.Tensor | None],
    mean_b: list[torch.Tensor | None],
    *,
    writer_batch_tokens: int,
) -> None:
    start = 0
    while start < len(rows):
        stop = start
        tokens = 0
        while stop < len(rows):
            row_tokens = rows[stop].token_count
            if stop > start and writer_batch_tokens > 0 and (
                tokens + row_tokens > writer_batch_tokens
            ):
                break
            tokens += row_tokens
            stop += 1
        pair_hidden: list[torch.Tensor] = []
        pair_means: list[torch.Tensor] = []
        for index in range(start, stop):
            current_a = hidden_a[index]
            current_b = hidden_b[index]
            current_mean_a = mean_a[index]
            current_mean_b = mean_b[index]
            if (
                current_a is None
                or current_b is None
                or current_mean_a is None
                or current_mean_b is None
            ):
                raise RuntimeError(f"missing staged forward result for row {index}")
            pair_hidden.extend((current_a, current_b))
            pair_means.append(torch.stack((current_mean_a, current_mean_b), dim=0))
        packed = torch.cat(pair_hidden, dim=0)
        means = torch.stack(pair_means, dim=0)
        writer.add_batch(
            list(rows[start:stop]),
            packed,
            means,
            order_indices=list(order_indices[start:stop]),
        )
        for index in range(start, stop):
            hidden_a[index] = None
            hidden_b[index] = None
            mean_a[index] = None
            mean_b[index] = None
        start = stop


def process_window_by_length(
    extractor: TargetLayerExtractor,
    writer: ActivationCacheV2Writer,
    entries: Sequence[tuple[PlanRow, int]],
    *,
    schedule: ForwardSchedule,
) -> ForwardWindowStats:
    """Forward a scheduler window using five independent length buckets.

    A and B remain independent sequences with positions reset to zero, but short
    sides are no longer forced to use the batch size selected for the longer
    partner. Results are staged on GPU until they can be submitted to the writer
    in the original rank-local order, preserving physical cache ordering.
    """

    if not entries:
        return ForwardWindowStats()
    rows = [item[0] for item in entries]
    order_indices = [int(item[1]) for item in entries]
    expected = list(range(order_indices[0], order_indices[0] + len(order_indices)))
    if order_indices != expected:
        raise ValueError("length-bucket forwarding requires a contiguous order window")

    hidden_a: list[torch.Tensor | None] = [None] * len(rows)
    hidden_b: list[torch.Tensor | None] = [None] * len(rows)
    mean_a: list[torch.Tensor | None] = [None] * len(rows)
    mean_b: list[torch.Tensor | None] = [None] * len(rows)
    buckets: dict[int, list[tuple[int, bool, Sequence[int]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[row.length_a].append((index, False, row.input_ids_a))
        buckets[row.length_b].append((index, True, row.input_ids_b))

    stats = ForwardWindowStats(
        pairs=len(rows),
        tokens=sum(row.token_count for row in rows),
    )
    for length in sorted(buckets, reverse=True):
        tasks = buckets[length]
        limit = schedule.batch_limit(length)
        for start in range(0, len(tasks), limit):
            current = tasks[start : start + limit]
            started = time.perf_counter()
            batch = extractor.forward_ids([task[2] for task in current])
            stats.forward_seconds += time.perf_counter() - started
            stats.model_calls += 1
            if batch.hidden.shape[:2] != (len(current), length):
                raise ValueError(
                    "forward result shape differs from the length-bucket request"
                )
            batch_means = batch.hidden.float().mean(dim=1)
            for batch_row, (row_index, is_b, _ids) in enumerate(current):
                value = batch.hidden[batch_row]
                mean = batch_means[batch_row]
                if is_b:
                    hidden_b[row_index] = value
                    mean_b[row_index] = mean
                else:
                    hidden_a[row_index] = value
                    mean_a[row_index] = mean

    started = time.perf_counter()
    _submit_pair_major_batches(
        writer,
        rows,
        order_indices,
        hidden_a,
        hidden_b,
        mean_a,
        mean_b,
        writer_batch_tokens=schedule.writer_batch_tokens,
    )
    stats.writer_submit_seconds += time.perf_counter() - started
    writer.wait_until_order(order_indices[-1] + 1)
    return stats
