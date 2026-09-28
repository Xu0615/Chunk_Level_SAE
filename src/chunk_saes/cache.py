from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .sample_plan import (
    SAMPLE_PLAN_FORMAT,
    PlanRow,
    plan_row_hash,
    tensor_payload_sha256,
)
from .utils import atomic_json_dump


ACTIVATION_CACHE_V2_FORMAT = "chunk-saes-activation-cache-v2"
ACTIVATION_CACHE_V2_SHARD_FORMAT = "chunk-saes-activation-cache-shard-v2"


def _activation_row_hashes(
    pair_ids: torch.Tensor,
    plan_row_hashes: torch.Tensor,
    token_hidden: torch.Tensor,
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
) -> torch.Tensor:
    result: list[list[int]] = []
    for index, pair_id in enumerate(pair_ids.tolist()):
        digest = hashlib.sha256()
        digest.update(b"chunk-saes-activation-row-v2")
        digest.update(int(pair_id).to_bytes(8, "little", signed=False))
        digest.update(bytes(plan_row_hashes[index].tolist()))
        for tensor in (token_hidden[index], mean_a[index], mean_b[index]):
            byte_view = tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(byte_view))
        result.append(list(digest.digest()))
    return torch.tensor(result, dtype=torch.uint8)


class ActivationCacheV2Writer:
    """Stream complete flattened token activations and pair metadata to v2 shards.

    Every pair is self-contained in one shard. ``token_hidden`` contains exactly
    ``len(A)+len(B)`` valid, unpadded activations in A-then-B order. The logical
    occurrence ID for a row is ``occurrence_start + local_token_offset``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        rank: int,
        world_size: int,
        hidden_size: int,
        plan_digest: str,
        target_token_occurrences: int,
        shard_token_limit: int,
        source_to_id: dict[str, int],
        activation_dtype: torch.dtype = torch.bfloat16,
        async_write_batches: int = 2,
        shard_write_workers: int = 2,
        max_pending_shards: int = 4,
    ) -> None:
        if shard_token_limit <= 0:
            raise ValueError("shard_token_limit must be positive")
        if activation_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported activation cache dtype: {activation_dtype}")
        self.root = Path(root)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.plan_digest = plan_digest
        self.target_token_occurrences = int(target_token_occurrences)
        self.shard_token_limit = int(shard_token_limit)
        self.source_to_id = dict(source_to_id)
        self.activation_dtype = activation_dtype
        self.async_write_batches = max(1, int(async_write_batches))
        self.rank_dir = self.root / f"rank{self.rank:03d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[PlanRow, torch.Tensor, torch.Tensor, bytes]] = []
        self._token_buffer: torch.Tensor | None = None
        self.buffered_tokens = 0
        self.shards: list[dict] = []
        self.pairs = 0
        self.tokens = 0
        self.source_tokens: dict[str, int] = {}
        self.cell_counts: dict[str, int] = {}
        self.occurrence_ranges: list[tuple[int, int]] = []
        self.target_sums = {
            mode: torch.zeros(self.hidden_size, dtype=torch.float64)
            for mode in (
                "token",
                "temporal",
                "mean",
                "cross",
                "cross_a_to_b",
                "cross_b_to_a",
            )
        }
        self.target_counts = {
            mode: 0
            for mode in self.target_sums
        }
        self._copy_stream: torch.cuda.Stream | None = None
        self._write_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"activation-writer-r{self.rank:03d}",
        )
        self._pending_writes: list[concurrent.futures.Future[None]] = []
        self._shard_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(shard_write_workers)),
            thread_name_prefix=f"activation-shard-io-r{self.rank:03d}",
        )
        self._pending_shard_writes: list[
            concurrent.futures.Future[dict]
        ] = []
        self._max_pending_shards = max(1, int(max_pending_shards))
        self._ordered_rows: dict[
            int, tuple[PlanRow, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._next_order_index = 0
        self._closed = False

    def _wait_for_write_slot(self) -> None:
        while len(self._pending_writes) >= self.async_write_batches:
            self._pending_writes.pop(0).result()

    def _drain_writes(self) -> None:
        for future in self._pending_writes:
            future.result()
        self._pending_writes.clear()

    def _collect_next_shard(self) -> None:
        item = self._pending_shard_writes.pop(0).result()
        target_sums = item.pop("_target_sums")
        target_counts = item.pop("_target_counts")
        for mode, values in target_sums.items():
            self.target_sums[mode].add_(values)
            self.target_counts[mode] += int(target_counts[mode])
        shard_id = int(item.pop("_shard_id"))
        if shard_id != len(self.shards):
            raise RuntimeError(
                f"activation shard publication order mismatch: "
                f"expected={len(self.shards)}, found={shard_id}"
            )
        self.shards.append(item)

    def _wait_for_shard_slot(self) -> None:
        while len(self._pending_shard_writes) >= self._max_pending_shards:
            self._collect_next_shard()

    def _drain_shard_writes(self) -> None:
        while self._pending_shard_writes:
            self._collect_next_shard()

    def wait_until_order(self, expected_order_index: int) -> None:
        """Wait for a bounded scheduler window to be committed in legacy order."""

        self._drain_writes()
        if self._next_order_index != int(expected_order_index) or self._ordered_rows:
            raise RuntimeError(
                "activation writer did not close its scheduler window: "
                f"next={self._next_order_index}, expected={expected_order_index}, "
                f"buffered={len(self._ordered_rows)}"
            )

    def add_batch(
        self,
        rows: list[PlanRow],
        token_hidden: torch.Tensor,
        means: torch.Tensor,
        order_indices: list[int] | None = None,
    ) -> None:
        """Queue one forward batch with two large asynchronous D2H copies.

        ``token_hidden`` is packed in pair-major A-then-B order and ``means``
        has shape ``[pairs, 2, hidden]``. A single ordered worker performs the
        original pair-wise buffering and shard publication, preserving logical
        row order and pair-granular shard boundaries.
        """

        if self._closed:
            raise RuntimeError("activation cache writer is closed")
        if not rows:
            return
        expected_tokens = sum(row.token_count for row in rows)
        if token_hidden.shape != (expected_tokens, self.hidden_size):
            raise ValueError(
                f"packed token_hidden shape {tuple(token_hidden.shape)} does not "
                f"match ({expected_tokens}, {self.hidden_size})"
            )
        if means.shape != (len(rows), 2, self.hidden_size):
            raise ValueError(
                f"packed means shape {tuple(means.shape)} does not match "
                f"({len(rows)}, 2, {self.hidden_size})"
            )
        if order_indices is not None and len(order_indices) != len(rows):
            raise ValueError("order_indices do not match activation batch rows")
        self._wait_for_write_slot()
        if token_hidden.device.type == "cuda":
            device = token_hidden.device
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=device)
            producer_stream = torch.cuda.current_stream(device=device)
            cpu_hidden = torch.empty(
                token_hidden.shape,
                dtype=self.activation_dtype,
                device="cpu",
                pin_memory=True,
            )
            cpu_means = torch.empty(
                means.shape,
                dtype=self.activation_dtype,
                device="cpu",
                pin_memory=True,
            )
            cpu_finite = torch.empty(
                (),
                dtype=torch.bool,
                device="cpu",
                pin_memory=True,
            )
            with torch.cuda.device(device), torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(producer_stream)
                finite = torch.isfinite(token_hidden).all() & torch.isfinite(
                    means
                ).all()
                cpu_hidden.copy_(token_hidden, non_blocking=True)
                cpu_means.copy_(means, non_blocking=True)
                cpu_finite.copy_(finite, non_blocking=True)
                token_hidden.record_stream(self._copy_stream)
                means.record_stream(self._copy_stream)
                event = torch.cuda.Event()
                event.record(self._copy_stream)
        else:
            cpu_hidden = token_hidden.detach().to(
                device="cpu", dtype=self.activation_dtype
            ).contiguous()
            cpu_means = means.detach().to(
                device="cpu", dtype=self.activation_dtype
            ).contiguous()
            cpu_finite = torch.isfinite(cpu_hidden).all() & torch.isfinite(
                cpu_means
            ).all()
            event = None
        future = self._write_executor.submit(
            self._consume_batch,
            list(rows),
            cpu_hidden,
            cpu_means,
            cpu_finite,
            event,
            None if order_indices is None else list(order_indices),
        )
        self._pending_writes.append(future)

    def _consume_batch(
        self,
        rows: list[PlanRow],
        token_hidden: torch.Tensor,
        means: torch.Tensor,
        finite: torch.Tensor,
        event: torch.cuda.Event | None,
        order_indices: list[int] | None,
    ) -> None:
        if event is not None:
            event.synchronize()
        if not bool(finite.item()):
            raise ValueError(
                f"Non-finite activation batch containing pair_id={rows[0].pair_id}"
            )
        if order_indices is not None:
            contiguous = order_indices == list(
                range(self._next_order_index, self._next_order_index + len(rows))
            )
            if contiguous:
                self._add_cpu_batch(rows, token_hidden, means)
                self._next_order_index += len(rows)
                while self._next_order_index in self._ordered_rows:
                    self._add_cpu(
                        *self._ordered_rows.pop(self._next_order_index)
                    )
                    self._next_order_index += 1
                return
            if not contiguous:
                # Future legacy-order rows must not retain a multi-GB pinned D2H
                # buffer. Pageable staging keeps pinned memory bounded by the
                # in-flight copy depth while a scheduler window is reordered.
                pageable_hidden = torch.empty(
                    token_hidden.shape,
                    dtype=token_hidden.dtype,
                    device="cpu",
                )
                pageable_means = torch.empty(
                    means.shape,
                    dtype=means.dtype,
                    device="cpu",
                )
                pageable_hidden.copy_(token_hidden)
                pageable_means.copy_(means)
                token_hidden = pageable_hidden
                means = pageable_means
        offset = 0
        for index, row in enumerate(rows):
            stop = offset + row.token_count
            values = (
                row,
                token_hidden[offset:stop],
                means[index, 0],
                means[index, 1],
            )
            if order_indices is None:
                self._add_cpu(*values)
            else:
                order_index = int(order_indices[index])
                if order_index < self._next_order_index or order_index in self._ordered_rows:
                    raise ValueError(f"duplicate/stale activation order index {order_index}")
                if order_index == self._next_order_index:
                    self._add_cpu(*values)
                    self._next_order_index += 1
                    while self._next_order_index in self._ordered_rows:
                        self._add_cpu(*self._ordered_rows.pop(self._next_order_index))
                        self._next_order_index += 1
                else:
                    self._ordered_rows[order_index] = values
            offset = stop
        if offset != token_hidden.shape[0]:
            raise RuntimeError("activation batch token offsets did not consume payload")

    def add(
        self,
        row: PlanRow,
        token_hidden: torch.Tensor,
        mean_a: torch.Tensor,
        mean_b: torch.Tensor,
    ) -> None:
        """Compatibility path for callers that submit one pair at a time."""
        self._drain_writes()
        if row.source not in self.source_to_id:
            raise ValueError(f"Unknown source in activation row: {row.source}")
        if token_hidden.ndim != 2 or token_hidden.shape != (
            row.token_count,
            self.hidden_size,
        ):
            raise ValueError(
                f"token_hidden shape {tuple(token_hidden.shape)} does not match "
                f"({row.token_count}, {self.hidden_size})"
            )
        if mean_a.shape not in ((self.hidden_size,), (1, self.hidden_size)):
            raise ValueError(f"mean_a has invalid shape: {tuple(mean_a.shape)}")
        if mean_b.shape not in ((self.hidden_size,), (1, self.hidden_size)):
            raise ValueError(f"mean_b has invalid shape: {tuple(mean_b.shape)}")
        if not bool(torch.isfinite(token_hidden).all()):
            raise ValueError(f"Non-finite token activation for pair_id={row.pair_id}")
        if not bool(torch.isfinite(mean_a.float()).all()) or not bool(
            torch.isfinite(mean_b.float()).all()
        ):
            raise ValueError(f"Non-finite chunk mean for pair_id={row.pair_id}")
        if row.token_count > self.shard_token_limit:
            raise ValueError(
                f"Pair {row.pair_id} has {row.token_count} tokens, exceeding cache "
                f"shard limit {self.shard_token_limit}"
            )
        if self.rows and self.buffered_tokens + row.token_count > self.shard_token_limit:
            self._flush()
        cpu_hidden = token_hidden.detach().to(
            device="cpu", dtype=self.activation_dtype
        ).contiguous()
        cpu_mean_a = mean_a.detach().reshape(-1).to(
            device="cpu", dtype=self.activation_dtype
        ).contiguous()
        cpu_mean_b = mean_b.detach().reshape(-1).to(
            device="cpu", dtype=self.activation_dtype
        ).contiguous()
        self._add_cpu_batch(
            [row],
            cpu_hidden,
            torch.stack((cpu_mean_a, cpu_mean_b), dim=0).unsqueeze(0),
        )

    def _add_cpu(
        self,
        row: PlanRow,
        cpu_hidden: torch.Tensor,
        cpu_mean_a: torch.Tensor,
        cpu_mean_b: torch.Tensor,
    ) -> None:
        self._add_cpu_batch(
            [row],
            cpu_hidden,
            torch.stack((cpu_mean_a, cpu_mean_b), dim=0).unsqueeze(0),
        )

    def _ensure_token_buffer(self) -> torch.Tensor:
        if self._token_buffer is None:
            self._token_buffer = torch.empty(
                (self.shard_token_limit, self.hidden_size),
                dtype=self.activation_dtype,
                device="cpu",
            )
        return self._token_buffer

    def _record_row(self, row: PlanRow, mean_a: torch.Tensor, mean_b: torch.Tensor) -> None:
        if row.source not in self.source_to_id:
            raise ValueError(f"Unknown source in activation row: {row.source}")
        self.rows.append(
            (
                row,
                mean_a.reshape(-1).contiguous().clone(),
                mean_b.reshape(-1).contiguous().clone(),
                plan_row_hash(row),
            )
        )
        self.pairs += 1
        self.tokens += row.token_count
        self.source_tokens[row.source] = (
            self.source_tokens.get(row.source, 0) + row.token_count
        )
        cell = f"{row.source}\t{row.length_a}\t{row.length_b}"
        self.cell_counts[cell] = self.cell_counts.get(cell, 0) + 1
        self.occurrence_ranges.append((row.occurrence_start, row.occurrence_stop))

    def _add_cpu_batch(
        self,
        rows: list[PlanRow],
        token_hidden: torch.Tensor,
        means: torch.Tensor,
    ) -> None:
        if not rows:
            return
        expected_tokens = sum(row.token_count for row in rows)
        if token_hidden.shape != (expected_tokens, self.hidden_size):
            raise ValueError("CPU activation batch has an invalid token shape")
        if means.shape != (len(rows), 2, self.hidden_size):
            raise ValueError("CPU activation batch has an invalid mean shape")
        pair_start = 0
        token_start = 0
        while pair_start < len(rows):
            if self.rows and (
                self.buffered_tokens + rows[pair_start].token_count
                > self.shard_token_limit
            ):
                self._flush()
            room = self.shard_token_limit - self.buffered_tokens
            pair_stop = pair_start
            token_count = 0
            while pair_stop < len(rows):
                pair_tokens = rows[pair_stop].token_count
                if pair_tokens > self.shard_token_limit:
                    raise ValueError(
                        f"Pair {rows[pair_stop].pair_id} has {pair_tokens} tokens, "
                        f"exceeding shard limit {self.shard_token_limit}"
                    )
                if pair_stop > pair_start and token_count + pair_tokens > room:
                    break
                if pair_stop == pair_start and pair_tokens > room:
                    break
                token_count += pair_tokens
                pair_stop += 1
            if pair_stop == pair_start:
                self._flush()
                continue
            token_buffer = self._ensure_token_buffer()
            destination = token_buffer.narrow(
                0,
                self.buffered_tokens,
                token_count,
            )
            destination.copy_(
                token_hidden.narrow(0, token_start, token_count),
                non_blocking=False,
            )
            for index in range(pair_start, pair_stop):
                self._record_row(rows[index], means[index, 0], means[index, 1])
            self.buffered_tokens += token_count
            token_start += token_count
            pair_start = pair_stop
            if self.buffered_tokens == self.shard_token_limit:
                self._flush()
        if token_start != expected_tokens:
            raise RuntimeError("CPU activation batch was not fully consumed")

    def _flush(self) -> None:
        if not self.rows:
            return
        self._wait_for_shard_slot()
        rows = [item[0] for item in self.rows]
        if self._token_buffer is None:
            raise RuntimeError("activation writer has rows without a token buffer")
        token_hidden = self._token_buffer.narrow(
            0, 0, self.buffered_tokens
        ).contiguous()
        self._token_buffer = None
        mean_a = torch.stack([item[1] for item in self.rows], dim=0)
        mean_b = torch.stack([item[2] for item in self.rows], dim=0)
        plan_row_hashes = torch.tensor(
            [list(item[3]) for item in self.rows], dtype=torch.uint8
        )
        shard_id = len(self.shards) + len(self._pending_shard_writes)
        self._pending_shard_writes.append(
            self._shard_executor.submit(
                self._write_shard,
                shard_id,
                rows,
                token_hidden,
                mean_a,
                mean_b,
                plan_row_hashes,
            )
        )
        self.rows.clear()
        self.buffered_tokens = 0

    def _write_shard(
        self,
        shard_id: int,
        rows: list[PlanRow],
        token_hidden: torch.Tensor,
        mean_a: torch.Tensor,
        mean_b: torch.Tensor,
        plan_row_hashes: torch.Tensor,
    ) -> dict:
        lengths_a = torch.tensor(
            [row.length_a for row in rows], dtype=torch.float64
        ).unsqueeze(1)
        lengths_b = torch.tensor(
            [row.length_b for row in rows], dtype=torch.float64
        ).unsqueeze(1)
        target_sums = {
            "token": token_hidden.sum(dim=0, dtype=torch.float64),
            "temporal": token_hidden.sum(dim=0, dtype=torch.float64),
            "mean": (
                (mean_a.to(torch.float64) * lengths_a).sum(dim=0)
                + (mean_b.to(torch.float64) * lengths_b).sum(dim=0)
            ),
            "cross": (
                (mean_b.to(torch.float64) * lengths_a).sum(dim=0)
                + (mean_a.to(torch.float64) * lengths_b).sum(dim=0)
            ),
            "cross_a_to_b": (
                mean_b.to(torch.float64) * lengths_a
            ).sum(dim=0),
            "cross_b_to_a": (
                mean_a.to(torch.float64) * lengths_b
            ).sum(dim=0),
        }
        token_count = int(token_hidden.shape[0])
        target_counts = {
            "token": token_count,
            "temporal": token_count,
            "mean": token_count,
            "cross": token_count,
            "cross_a_to_b": int(lengths_a.sum()),
            "cross_b_to_a": int(lengths_b.sum()),
        }
        chunk_offsets = [0]
        for row in rows:
            chunk_offsets.append(chunk_offsets[-1] + row.length_a)
            chunk_offsets.append(chunk_offsets[-1] + row.length_b)
        pair_ids = torch.tensor([row.pair_id for row in rows], dtype=torch.int64)
        tensors = {
            "token_hidden": token_hidden,
            "chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.int64),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "pair_id": pair_ids,
            "occurrence_start": torch.tensor(
                [row.occurrence_start for row in rows], dtype=torch.int64
            ),
            "doc_hash": torch.tensor(
                [list(row.doc_hash) for row in rows], dtype=torch.uint8
            ),
            "content_hash": torch.tensor(
                [list(row.content_hash) for row in rows], dtype=torch.uint8
            ),
            "source_id": torch.tensor(
                [self.source_to_id[row.source] for row in rows], dtype=torch.int32
            ),
            "start_a": torch.tensor([row.start_a for row in rows], dtype=torch.int64),
            "start_b": torch.tensor([row.start_b for row in rows], dtype=torch.int64),
            "length_a": torch.tensor([row.length_a for row in rows], dtype=torch.int32),
            "length_b": torch.tensor([row.length_b for row in rows], dtype=torch.int32),
            "document_token_count": torch.tensor(
                [row.document_token_count for row in rows], dtype=torch.int64
            ),
            "document_reuse_index": torch.tensor(
                [row.document_reuse_index for row in rows], dtype=torch.int32
            ),
            "execution_rank": torch.tensor(
                [row.execution_rank for row in rows], dtype=torch.int32
            ),
            "plan_row_hash": plan_row_hashes,
        }
        pair_ids = tensors["pair_id"]
        plan_row_hashes = tensors["plan_row_hash"]
        tensors["activation_row_hash"] = _activation_row_hashes(
            pair_ids,
            plan_row_hashes,
            tensors["token_hidden"],
            tensors["mean_a"],
            tensors["mean_b"],
        )
        payload_sha256 = tensor_payload_sha256(tensors)
        final_path = self.rank_dir / f"shard-{shard_id:06d}.safetensors"
        partial_path = self.rank_dir / f".{final_path.name}.partial"
        partial_path.unlink(missing_ok=True)
        save_file(
            tensors,
            str(partial_path),
            metadata={
                "format": ACTIVATION_CACHE_V2_SHARD_FORMAT,
                "plan_digest": self.plan_digest,
            },
        )
        os.replace(partial_path, final_path)
        return {
            "_shard_id": shard_id,
            "_target_sums": target_sums,
            "_target_counts": target_counts,
            "path": str(final_path.relative_to(self.root)),
            "pairs": len(rows),
            "tokens": int(tensors["token_hidden"].shape[0]),
            "pair_id_min": min(row.pair_id for row in rows),
            "pair_id_max": max(row.pair_id for row in rows),
            "occurrence_min": min(row.occurrence_start for row in rows),
            "occurrence_max_exclusive": max(row.occurrence_stop for row in rows),
            "payload_sha256": payload_sha256,
            "bytes": final_path.stat().st_size,
        }

    def finish(self) -> dict:
        self._drain_writes()
        if self._ordered_rows:
            raise ValueError(
                f"activation writer is missing order_index={self._next_order_index}"
            )
        self._flush()
        self._drain_shard_writes()
        activation_digest = hashlib.sha256()
        for item in self.shards:
            activation_digest.update(bytes.fromhex(item["payload_sha256"]))
        manifest = {
            "format": ACTIVATION_CACHE_V2_FORMAT,
            "complete": True,
            "rank": self.rank,
            "world_size": self.world_size,
            "hidden_size": self.hidden_size,
            "activation_dtype": str(self.activation_dtype).removeprefix("torch."),
            "plan_digest": self.plan_digest,
            "target_token_occurrences": self.target_token_occurrences,
            "pairs": self.pairs,
            "token_occurrences": self.tokens,
            "source_tokens": dict(sorted(self.source_tokens.items())),
            "cell_counts": dict(sorted(self.cell_counts.items())),
            "assignment": "sample_plan_execution_rank",
            "activation_digest": activation_digest.hexdigest(),
            "target_sufficient_statistics": {
                "count": self.tokens,
                "count_by_mode": self.target_counts,
                "sum_by_mode": {
                    mode: values.tolist()
                    for mode, values in self.target_sums.items()
                },
                "accumulator_dtype": "float64",
                "source": "all_cached_target_rows",
            },
            "occurrence_range_digest": hashlib.sha256(
                b"".join(
                    start.to_bytes(8, "little", signed=False)
                    + stop.to_bytes(8, "little", signed=False)
                    for start, stop in sorted(self.occurrence_ranges)
                )
            ).hexdigest(),
            "shards": self.shards,
        }
        atomic_json_dump(manifest, self.rank_dir / "manifest.json")
        self.close()
        return manifest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._drain_writes()
        finally:
            try:
                self._drain_shard_writes()
            finally:
                self._write_executor.shutdown(wait=True, cancel_futures=False)
                self._shard_executor.shutdown(wait=True, cancel_futures=False)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _load_v2_shard(
    path: Path,
    names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != ACTIVATION_CACHE_V2_SHARD_FORMAT:
            raise ValueError(f"Not an activation-cache v2 shard: {path}")
        available = set(handle.keys())
        selected = available if names is None else set(names)
        missing = selected - available
        if missing:
            raise ValueError(f"Activation shard {path} lacks tensors {sorted(missing)}")
        return {name: handle.get_tensor(name) for name in selected}


def merge_v2_rank_manifests(
    root: str | Path,
    *,
    world_size: int,
    plan_manifest: dict,
    extra: dict | None = None,
    verify_payload_checksums: bool = False,
    verify_activation_rows: bool = False,
) -> dict:
    """Validate exact global coverage before publishing the cache manifest."""
    root = Path(root)
    if plan_manifest.get("format") != SAMPLE_PLAN_FORMAT:
        raise ValueError(
            f"Activation cache v2 requires a {SAMPLE_PLAN_FORMAT} manifest"
        )
    planned_world_size = int(
        plan_manifest.get("identity", {}).get("execution_world_size", 1)
    )
    if planned_world_size > 1 and planned_world_size != world_size:
        raise ValueError(
            f"sample plan execution world size {planned_world_size} != cache world size {world_size}"
        )
    ranks = []
    for rank in range(world_size):
        path = root / f"rank{rank:03d}" / "manifest.json"
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != ACTIVATION_CACHE_V2_FORMAT:
            raise ValueError(
                f"Rank {rank} cache is not v2 (found {manifest.get('format')!r}); "
                "v1 caches cannot be reused"
            )
        if not manifest.get("complete"):
            raise ValueError(f"Rank {rank} activation cache is incomplete")
        if manifest.get("rank") != rank or manifest.get("world_size") != world_size:
            raise ValueError(f"Rank metadata mismatch in {path}")
        if manifest.get("plan_digest") != plan_manifest.get("plan_digest"):
            raise ValueError(f"Rank {rank} plan digest mismatch")
        ranks.append(manifest)

    pairs = int(plan_manifest["pairs"])
    target_tokens = int(plan_manifest["target_token_occurrences"])
    seen = torch.zeros(pairs, dtype=torch.bool)
    occurrence_starts = torch.empty(pairs, dtype=torch.int64)
    token_counts = torch.empty(pairs, dtype=torch.int64)
    plan_hashes = torch.empty((pairs, 32), dtype=torch.uint8)
    global_activation_hashes = torch.empty((pairs, 32), dtype=torch.uint8)
    source_tokens: dict[str, int] = {}
    cell_counts: dict[str, int] = {}
    hidden_size = None
    activation_dtype = None
    target_sums: dict[str, torch.Tensor] | None = None
    target_stat_counts: dict[str, int] | None = None
    for manifest in ranks:
        if hidden_size is None:
            hidden_size = manifest["hidden_size"]
            activation_dtype = manifest["activation_dtype"]
        elif (
            manifest["hidden_size"] != hidden_size
            or manifest["activation_dtype"] != activation_dtype
        ):
            raise ValueError("Rank activation cache tensor schema mismatch")
        statistics = manifest.get("target_sufficient_statistics")
        if not isinstance(statistics, dict):
            raise ValueError("Rank activation cache lacks target sufficient statistics")
        rank_stat_count = int(statistics.get("count", -1))
        if rank_stat_count != int(manifest["token_occurrences"]):
            raise ValueError("Rank target-stat count differs from token occurrences")
        raw_sums = statistics.get("sum_by_mode")
        expected_stat_modes = {
            "token",
            "temporal",
            "mean",
            "cross",
            "cross_a_to_b",
            "cross_b_to_a",
        }
        if not isinstance(raw_sums, dict) or set(raw_sums) != expected_stat_modes:
            raise ValueError("Rank target statistics do not cover all SAE modes")
        raw_counts = statistics.get("count_by_mode")
        if not isinstance(raw_counts, dict) or set(raw_counts) != expected_stat_modes:
            raise ValueError("Rank target-stat counts do not cover all SAE modes")
        current_counts = {mode: int(raw_counts[mode]) for mode in expected_stat_modes}
        if any(count <= 0 for count in current_counts.values()):
            raise ValueError("Rank target-stat counts must be positive")
        if any(
            current_counts[mode] != rank_stat_count
            for mode in ("token", "temporal", "mean", "cross")
        ):
            raise ValueError("Rank primary target-stat counts differ from occurrences")
        if (
            current_counts["cross_a_to_b"]
            + current_counts["cross_b_to_a"]
            != rank_stat_count
        ):
            raise ValueError("Rank Cross directional counts do not cover occurrences")
        current_sums = {
            mode: torch.tensor(raw_sums[mode], dtype=torch.float64)
            for mode in expected_stat_modes
        }
        if any(values.numel() != int(hidden_size) for values in current_sums.values()):
            raise ValueError("Rank target-stat hidden size is invalid")
        if target_sums is None:
            target_sums = current_sums
            target_stat_counts = current_counts
        else:
            for mode, values in current_sums.items():
                target_sums[mode].add_(values)
                assert target_stat_counts is not None
                target_stat_counts[mode] += current_counts[mode]
        for source, count in manifest["source_tokens"].items():
            source_tokens[source] = source_tokens.get(source, 0) + int(count)
        for cell, count in manifest["cell_counts"].items():
            cell_counts[cell] = cell_counts.get(cell, 0) + int(count)
        for item in manifest["shards"]:
            path = root / item["path"]
            full_payload = verify_payload_checksums or verify_activation_rows
            tensors = _load_v2_shard(
                path,
                None
                if full_payload
                else {
                    "pair_id",
                    "execution_rank",
                    "chunk_offsets",
                    "length_a",
                    "length_b",
                    "occurrence_start",
                    "plan_row_hash",
                    "activation_row_hash",
                },
            )
            if (
                verify_payload_checksums
                and tensor_payload_sha256(tensors) != item["payload_sha256"]
            ):
                raise ValueError(f"Activation shard payload checksum mismatch: {path}")
            pair_id = tensors["pair_id"].long()
            if pair_id.numel() != int(item["pairs"]):
                raise ValueError(f"Pair count mismatch in {path}")
            if bool(((pair_id < 0) | (pair_id >= pairs)).any()):
                raise ValueError(f"Out-of-range pair ID in {path}")
            if bool(seen[pair_id].any()):
                raise ValueError(f"Duplicate pair ID in activation cache: {path}")
            if "execution_rank" not in tensors:
                raise ValueError(f"Activation shard lacks execution_rank: {path}")
            planned_rank = tensors["execution_rank"].long()
            expected_rank = (
                planned_rank
                if planned_world_size > 1
                else pair_id.remainder(world_size)
            )
            if bool((expected_rank != int(manifest["rank"])).any()):
                raise ValueError(f"Pair assigned to wrong rank in {path}")
            offsets = tensors["chunk_offsets"].long()
            if offsets.numel() != 2 * pair_id.numel() + 1:
                raise ValueError(f"Invalid chunk_offsets shape in {path}")
            declared_tokens = int(item["tokens"])
            if int(offsets[0]) != 0 or int(offsets[-1]) != declared_tokens:
                raise ValueError(f"chunk_offsets do not cover token_hidden in {path}")
            if bool((offsets[1:] <= offsets[:-1]).any()):
                raise ValueError(f"chunk_offsets are not strictly increasing in {path}")
            lengths_a = tensors["length_a"].long()
            lengths_b = tensors["length_b"].long()
            if not torch.equal(offsets[1::2] - offsets[:-1:2], lengths_a):
                raise ValueError(f"A lengths disagree with chunk_offsets in {path}")
            if not torch.equal(offsets[2::2] - offsets[1::2], lengths_b):
                raise ValueError(f"B lengths disagree with chunk_offsets in {path}")
            counts = lengths_a + lengths_b
            if verify_activation_rows:
                expected_activation_hashes = _activation_row_hashes(
                    pair_id,
                    tensors["plan_row_hash"],
                    tensors["token_hidden"],
                    tensors["mean_a"],
                    tensors["mean_b"],
                )
                if not torch.equal(
                    expected_activation_hashes, tensors["activation_row_hash"]
                ):
                    raise ValueError(f"Activation-row checksum mismatch in {path}")
            seen[pair_id] = True
            occurrence_starts[pair_id] = tensors["occurrence_start"]
            token_counts[pair_id] = counts
            plan_hashes[pair_id] = tensors["plan_row_hash"]
            global_activation_hashes[pair_id] = tensors["activation_row_hash"]
    if not bool(seen.all()):
        missing = (~seen).nonzero().flatten()[:10].tolist()
        raise ValueError(f"Activation cache is missing pair IDs, first missing={missing}")
    expected_starts = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(token_counts[:-1], dim=0)]
    )
    if not torch.equal(occurrence_starts, expected_starts):
        raise ValueError("Activation cache occurrence ranges are not contiguous")
    if int(token_counts.sum()) != target_tokens:
        raise ValueError(
            f"Activation cache contains {int(token_counts.sum())} token occurrences, "
            f"expected exact target {target_tokens}"
        )
    if target_sums is None or target_stat_counts is None:
        raise ValueError("Global target sufficient statistics do not cover the cache")
    if any(
        target_stat_counts[mode] != target_tokens
        for mode in ("token", "temporal", "mean", "cross")
    ):
        raise ValueError("Global primary target statistics do not cover the cache")
    if (
        target_stat_counts["cross_a_to_b"]
        + target_stat_counts["cross_b_to_a"]
        != target_tokens
    ):
        raise ValueError("Global Cross directional statistics do not cover the cache")
    target_means = {
        mode: (values / target_stat_counts[mode]).tolist()
        for mode, values in target_sums.items()
    }
    expected_rank_tokens = target_tokens // world_size
    if target_tokens % world_size:
        raise ValueError("Exact cache token target is not divisible by world size")
    observed_rank_tokens = [int(manifest["token_occurrences"]) for manifest in ranks]
    if observed_rank_tokens != [expected_rank_tokens] * world_size:
        raise ValueError(
            "Activation cache ranks do not have exact equal token coverage: "
            f"observed={observed_rank_tokens}, expected={expected_rank_tokens}"
        )
    plan_rows_digest = hashlib.sha256(memoryview(plan_hashes.contiguous().numpy())).hexdigest()
    if plan_rows_digest != plan_manifest["rows_digest"]:
        raise ValueError("Activation cache plan-row universe differs from sample plan")
    activation_digest = hashlib.sha256(
        memoryview(global_activation_hashes.contiguous().numpy())
    ).hexdigest()
    source_tokens = dict(sorted(source_tokens.items()))
    cell_counts = dict(sorted(cell_counts.items()))
    if source_tokens != plan_manifest.get("source_tokens"):
        raise ValueError("Activation cache source-token totals differ from sample plan")
    if cell_counts != plan_manifest.get("cell_counts"):
        raise ValueError("Activation cache source/length cell counts differ from sample plan")
    merged = {
        "format": ACTIVATION_CACHE_V2_FORMAT,
        "complete": True,
        "plan_digest": plan_manifest["plan_digest"],
        "plan_rows_digest": plan_rows_digest,
        "activation_digest": activation_digest,
        "world_size": world_size,
        "pairs": pairs,
        "target_token_occurrences": target_tokens,
        "token_occurrences": target_tokens,
        "hidden_size": hidden_size,
        "activation_dtype": activation_dtype,
        "source_tokens": source_tokens,
        "cell_counts": cell_counts,
        "occurrence_ids": {"start": 0, "stop": target_tokens, "contiguous": True},
        "corpus_position_overlap_policy": plan_manifest.get(
            "corpus_position_overlap_policy"
        ),
        "corpus_position_overlap_verified": plan_manifest.get(
            "corpus_position_overlap_verified"
        ),
        "unique_corpus_token_positions": plan_manifest.get(
            "unique_corpus_token_positions"
        ),
        "coverage": {
            "pair_ids_complete": True,
            "pair_ids_unique": True,
            "occurrence_ranges_complete": True,
            "token_hidden_rows_equal_occurrences": True,
            "plan_row_digest_matches": True,
        },
        "merge_verification": {
            "activation_rows_rehashed": bool(verify_activation_rows),
            "payloads_rehashed": bool(verify_payload_checksums),
            "default_mode": "metadata-and-writer-digests",
        },
        "rank_coverage": {
            "tokens_per_rank": [expected_rank_tokens] * world_size,
            "equal_token_coverage": True,
            "assignment": (
                "sample_plan_execution_rank" if planned_world_size > 1 else "pair_id_mod_world_size"
            ),
        },
        "target_sufficient_statistics": {
            "count": target_tokens,
            "count_by_mode": target_stat_counts,
            "mean_by_mode": target_means,
            "accumulator_dtype": "float64",
            "source": "all_cached_target_rows",
            "frozen_train_only": True,
        },
        "ranks": ranks,
        **(extra or {}),
    }
    atomic_json_dump(merged, root / "manifest.json")
    return merged


def load_activation_cache_v2_manifest(root: str | Path) -> dict:
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != ACTIVATION_CACHE_V2_FORMAT:
        raise ValueError(
            f"Expected {ACTIVATION_CACHE_V2_FORMAT}, found {manifest.get('format')!r}; "
            "v1 activation caches cannot provide exact token coverage"
        )
    if not manifest.get("complete"):
        raise ValueError(f"Activation cache v2 is incomplete: {root}")
    if manifest.get("token_occurrences") != manifest.get("target_token_occurrences"):
        raise ValueError("Activation cache v2 token total is not exact")
    return manifest
