from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .sample_plan import (
    BOUNDARY_VERSION,
    PLAN_ROW_HASH_VERSION,
    SAMPLE_PLAN_FORMAT,
    PlanRow,
    PreparedDocument,
    SamplePlanWriter,
    _hex_or_sha256,
    _keyed_digest,
    _solve_rank_weight_counts,
    iter_prepared_document_records,
    load_sample_plan_manifest,
    solve_exact_cell_counts,
    solve_proportional_cell_counts,
)


STREAMING_SAMPLING_VERSION = "prepared-hash-order-streaming-exact-quota"
STREAMING_BOUNDARY_VERSION = "prepared-streaming-nonoverlap-boundary"
STREAMING_RANK_VERSION = "prepared-streaming-weight-round-robin"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prepared_corpus_digest(manifest: Mapping[str, object]) -> str:
    """Hash the logical prepared corpus without depending on physical rank files."""

    logical = {
        key: manifest.get(key)
        for key in (
            "format",
            "dataset",
            "split_seed",
            "split",
            "excluded_sources",
            "min_chars",
            "generic_filtering",
            "cleaned_documents",
            "content_normalization",
            "split_key",
            "split_before_rank",
            "deduplication",
            "deduplicated_globally",
            "document_order",
            "documents",
            "raw_source_documents",
            "raw_source_text_bytes",
            "raw_source_weight_scope",
        )
    }
    return hashlib.sha256(_canonical_json(logical)).hexdigest()


def _tokenize_batch(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    if not texts:
        return []
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise ValueError("Batched tokenizer result has no input_ids field")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if len(input_ids) != len(texts):
        raise ValueError(
            f"Batched tokenizer returned {len(input_ids)} rows for {len(texts)} texts"
        )
    return [[int(token) for token in values] for values in input_ids]


def iter_tokenized_prepared_documents(
    documents: Iterable[PreparedDocument],
    tokenizer,
    *,
    batch_size: int,
    batch_chars: int,
) -> Iterator[tuple[PreparedDocument, list[int]]]:
    """Batch-tokenize a prepared stream while preserving its logical order."""

    if batch_size <= 0 or batch_chars <= 0:
        raise ValueError("streaming tokenizer batch limits must be positive")
    pending: list[PreparedDocument] = []
    pending_chars = 0

    def flush() -> Iterator[tuple[PreparedDocument, list[int]]]:
        nonlocal pending_chars
        if not pending:
            return
        token_rows = _tokenize_batch(tokenizer, [document.text for document in pending])
        for document, token_ids in zip(pending, token_rows, strict=True):
            yield document, token_ids
        pending.clear()
        pending_chars = 0

    for document in documents:
        text_chars = len(document.text)
        if pending and (
            len(pending) >= batch_size
            or pending_chars + text_chars > batch_chars
        ):
            yield from flush()
        pending.append(document)
        pending_chars += text_chars
    yield from flush()


class StreamingRankAllocator:
    """Assign pair weights so every execution rank receives an exact token total."""

    def __init__(
        self,
        counts: Mapping[tuple[str, int, int], int],
        *,
        world_size: int,
        target_tokens: int,
        seed: int,
    ) -> None:
        weight_counts: Counter[int] = Counter()
        for (_source, length_a, length_b), count in counts.items():
            weight_counts[int(length_a) + int(length_b)] += int(count)
        assignment = _solve_rank_weight_counts(
            dict(weight_counts),
            world_size=world_size,
            target_tokens=target_tokens,
        )
        self.world_size = int(world_size)
        self.target_tokens = int(target_tokens)
        self.remaining = {
            weight: {
                rank: int(assignment[rank].get(weight, 0))
                for rank in range(self.world_size)
            }
            for weight in sorted(weight_counts)
        }
        self.initial = {
            weight: dict(values) for weight, values in self.remaining.items()
        }
        self.rank_orders = {
            weight: sorted(
                range(self.world_size),
                key=lambda rank: _keyed_digest(
                    STREAMING_RANK_VERSION,
                    seed,
                    weight,
                    rank,
                    digest_size=8,
                ),
            )
            for weight in self.remaining
        }
        self.cursors = {weight: 0 for weight in self.remaining}
        self.rank_pairs = [0 for _ in range(self.world_size)]
        self.rank_tokens = [0 for _ in range(self.world_size)]

    def assign(self, weight: int) -> int:
        weight = int(weight)
        remaining = self.remaining.get(weight)
        if remaining is None:
            raise ValueError(f"No execution-rank quota for pair weight={weight}")
        order = self.rank_orders[weight]
        cursor = self.cursors[weight]
        checked = 0
        while remaining[order[cursor]] <= 0:
            cursor = (cursor + 1) % len(order)
            checked += 1
            if checked > len(order):
                raise RuntimeError(f"Execution-rank quota exhausted for weight={weight}")
        rank = order[cursor]
        remaining[rank] -= 1
        self.cursors[weight] = (cursor + 1) % len(order)
        self.rank_pairs[rank] += 1
        self.rank_tokens[rank] += weight
        return rank

    def finish(self) -> dict:
        leftovers = {
            str(weight): {str(rank): count for rank, count in values.items() if count}
            for weight, values in self.remaining.items()
            if any(values.values())
        }
        if leftovers:
            raise RuntimeError(
                "Streaming rank assignment did not consume every weight quota: "
                + json.dumps(leftovers, sort_keys=True)
            )
        expected = self.target_tokens // self.world_size
        if self.target_tokens % self.world_size:
            raise ValueError("target tokens are not divisible by execution world size")
        if self.rank_tokens != [expected] * self.world_size:
            raise RuntimeError(
                f"Streaming rank token totals differ: {self.rank_tokens}"
            )
        return {
            "world_size": self.world_size,
            "tokens_per_rank": self.rank_tokens,
            "pairs_per_rank": self.rank_pairs,
            "weight_counts_per_rank": {
                str(rank): {
                    str(weight): int(self.initial[weight][rank])
                    for weight in sorted(self.initial)
                }
                for rank in range(self.world_size)
            },
            "algorithm": STREAMING_RANK_VERSION,
        }


@dataclass(frozen=True)
class _SelectedCell:
    length_a: int
    length_b: int
    reuse_index: int

    @property
    def span(self) -> int:
        return self.length_a + self.length_b


class StreamingExactPlanner:
    """Consume prepared documents once and fill exact source/length quotas."""

    def __init__(
        self,
        counts: Mapping[tuple[str, int, int], int],
        *,
        seed: int,
        max_document_reuses: int,
        rank_allocator: StreamingRankAllocator,
    ) -> None:
        if max_document_reuses <= 0:
            raise ValueError("max_document_reuses must be positive")
        self.seed = int(seed)
        self.max_document_reuses = int(max_document_reuses)
        self.quotas = {
            (str(source), int(length_a), int(length_b)): int(count)
            for (source, length_a, length_b), count in counts.items()
        }
        self.remaining = dict(self.quotas)
        self.pending_pairs = sum(self.remaining.values())
        self.pending_tokens = sum(
            (length_a + length_b) * count
            for (_source, length_a, length_b), count in self.remaining.items()
        )
        self.cells_by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for source, length_a, length_b in self.remaining:
            self.cells_by_source[source].append((length_a, length_b))
        for source in self.cells_by_source:
            self.cells_by_source[source].sort()
        self.rank_allocator = rank_allocator
        self.next_pair_id = 0
        self.next_occurrence = 0
        self.documents_scanned = 0
        self.candidate_tokens = 0
        self.selected_documents = 0
        self.selected_pairs = 0
        self.selected_tokens = 0
        self.pair_histogram: Counter[int] = Counter()
        self.token_histogram: Counter[int] = Counter()

    @property
    def complete(self) -> bool:
        return self.pending_pairs == 0

    def _choose_cells(
        self,
        source: str,
        content_digest: bytes,
        token_count: int,
    ) -> list[_SelectedCell]:
        available = int(token_count)
        selected: list[_SelectedCell] = []
        cells = self.cells_by_source.get(source, ())
        for reuse_index in range(self.max_document_reuses):
            eligible = [
                cell
                for cell in cells
                if self.remaining[(source, cell[0], cell[1])] > 0
                and cell[0] + cell[1] <= available
            ]
            if not eligible:
                break

            def key(cell: tuple[int, int]) -> tuple:
                quota = self.quotas[(source, cell[0], cell[1])]
                remaining = self.remaining[(source, cell[0], cell[1])]
                tie = _keyed_digest(
                    "streaming-cell-choice",
                    self.seed,
                    content_digest,
                    reuse_index,
                    cell[0],
                    cell[1],
                    digest_size=8,
                )
                return (
                    cell[0] + cell[1],
                    remaining / max(1, quota),
                    tie,
                )

            length_a, length_b = max(eligible, key=key)
            span = length_a + length_b
            selected.append(
                _SelectedCell(
                    length_a=length_a,
                    length_b=length_b,
                    reuse_index=reuse_index,
                )
            )
            self.remaining[(source, length_a, length_b)] -= 1
            self.pending_pairs -= 1
            self.pending_tokens -= span
            available -= span
            if self.pending_pairs == 0:
                break
        return selected

    def _boundaries(
        self,
        content_digest: bytes,
        token_count: int,
        selected: Sequence[_SelectedCell],
    ) -> list[int]:
        if not selected:
            return []
        total = sum(item.span for item in selected)
        slack = int(token_count) - total
        if slack < 0:
            raise RuntimeError("Streaming selections exceed document capacity")
        order = sorted(
            range(len(selected)),
            key=lambda index: _keyed_digest(
                "streaming-span-order",
                self.seed,
                content_digest,
                index,
                selected[index].length_a,
                selected[index].length_b,
                selected[index].reuse_index,
                digest_size=16,
            ),
        )
        cuts = sorted(
            int.from_bytes(
                _keyed_digest(
                    STREAMING_BOUNDARY_VERSION,
                    self.seed,
                    content_digest,
                    gap_index,
                    digest_size=16,
                ),
                "big",
            )
            % (slack + 1)
            for gap_index in range(len(selected))
        )
        gaps: list[int] = []
        previous = 0
        for cut in cuts:
            gaps.append(cut - previous)
            previous = cut
        gaps.append(slack - previous)
        starts = [0 for _ in selected]
        cursor = gaps[0]
        for position, selected_index in enumerate(order):
            starts[selected_index] = cursor
            cursor += selected[selected_index].span + gaps[position + 1]
        if cursor != token_count:
            raise RuntimeError("Streaming boundary allocation did not consume layout")
        intervals = sorted(
            (starts[index], starts[index] + selected[index].span)
            for index in range(len(selected))
        )
        for previous_interval, current_interval in zip(intervals, intervals[1:]):
            if current_interval[0] < previous_interval[1]:
                raise RuntimeError("Streaming boundary allocation produced overlap")
        return starts

    def add_document(
        self,
        document: PreparedDocument,
        token_ids: Sequence[int],
    ) -> list[PlanRow]:
        self.documents_scanned += 1
        self.candidate_tokens += len(token_ids)
        if self.complete or document.source not in self.cells_by_source:
            return []
        content_digest = _hex_or_sha256(document.content_hash)
        selected = self._choose_cells(
            document.source,
            content_digest,
            len(token_ids),
        )
        if not selected:
            return []
        starts = self._boundaries(content_digest, len(token_ids), selected)
        doc_digest = hashlib.sha256(
            document.doc_id.encode("utf-8", errors="replace")
        ).digest()
        rows: list[PlanRow] = []
        for item, start_a in zip(selected, starts, strict=True):
            start_b = start_a + item.length_a
            stop_b = start_b + item.length_b
            execution_rank = self.rank_allocator.assign(item.span)
            row = PlanRow(
                pair_id=self.next_pair_id,
                occurrence_start=self.next_occurrence,
                doc_hash=doc_digest,
                content_hash=content_digest,
                source=document.source,
                input_ids_a=tuple(int(value) for value in token_ids[start_a:start_b]),
                input_ids_b=tuple(int(value) for value in token_ids[start_b:stop_b]),
                start_a=start_a,
                start_b=start_b,
                document_token_count=len(token_ids),
                document_reuse_index=item.reuse_index,
                execution_rank=execution_rank,
            )
            rows.append(row)
            self.next_pair_id += 1
            self.next_occurrence += item.span
        self.selected_documents += 1
        self.selected_pairs += len(rows)
        document_selected_tokens = sum(row.token_count for row in rows)
        self.selected_tokens += document_selected_tokens
        self.pair_histogram[len(rows)] += 1
        self.token_histogram[document_selected_tokens] += 1
        return rows

    @staticmethod
    def _percentile(histogram: Counter[int], fraction: float) -> int:
        total_count = sum(histogram.values())
        if not total_count:
            return 0
        target = int(fraction * (total_count - 1))
        cursor = 0
        for value, count in sorted(histogram.items()):
            cursor += count
            if cursor > target:
                return value
        return max(histogram)

    def finish(self, *, target_tokens: int, expected_pairs: int) -> dict:
        if not self.complete:
            deficits = {
                f"{source}\t{length_a}\t{length_b}": count
                for (source, length_a, length_b), count in sorted(
                    self.remaining.items()
                )
                if count
            }
            raise ValueError(
                "Prepared stream exhausted before exact quotas were filled: "
                + json.dumps(deficits, sort_keys=True)
            )
        if self.next_pair_id != expected_pairs:
            raise RuntimeError(
                f"Streaming planner selected {self.next_pair_id} pairs, "
                f"expected {expected_pairs}"
            )
        if self.next_occurrence != target_tokens:
            raise RuntimeError(
                f"Streaming planner selected {self.next_occurrence} tokens, "
                f"expected {target_tokens}"
            )
        rank_stats = self.rank_allocator.finish()
        return {
            "algorithm": STREAMING_SAMPLING_VERSION,
            "prepared_documents_scanned": self.documents_scanned,
            "candidate_qwen_tokens": self.candidate_tokens,
            "candidate_to_selected_token_ratio": (
                self.candidate_tokens / max(1, self.selected_tokens)
            ),
            "selected_documents": self.selected_documents,
            "selected_pairs": self.selected_pairs,
            "unique_corpus_token_positions": self.selected_tokens,
            "max_pairs_per_document": max(self.pair_histogram, default=0),
            "max_tokens_per_document": max(self.token_histogram, default=0),
            "pairs_per_document_p50": self._percentile(
                self.pair_histogram, 0.50
            ),
            "pairs_per_document_p95": self._percentile(
                self.pair_histogram, 0.95
            ),
            "pairs_per_document_p99": self._percentile(
                self.pair_histogram, 0.99
            ),
            "tokens_per_document_p50": self._percentile(
                self.token_histogram, 0.50
            ),
            "tokens_per_document_p95": self._percentile(
                self.token_histogram, 0.95
            ),
            "tokens_per_document_p99": self._percentile(
                self.token_histogram, 0.99
            ),
            "overlap_verified": True,
            "execution_rank_assignment": rank_stats,
        }


def build_streaming_sample_plan(
    document_root: str | Path,
    tokenizer,
    output_dir: str | Path,
    *,
    tokenizer_hash: str,
    target_tokens: int,
    lengths: Sequence[int],
    sample_seed: int,
    sources: Sequence[str] | None = None,
    excluded_sources: Iterable[str] = (),
    shard_token_limit: int = 1_000_000,
    max_document_reuses: int = 64,
    execution_world_size: int = 8,
    tokenizer_batch_size: int = 1024,
    tokenizer_batch_chars: int = 8_000_000,
    source_weighting: str = "corpus_proportional",
    overwrite: bool = False,
    progress_every_documents: int = 100_000,
) -> dict:
    """Build a standard immutable plan from one early-stopping prepared stream.

    Unlike the catalog planner, this function never tokenizes the full corpus
    before selection. It visits prepared documents in their deterministic
    content-hash order, fills exact quotas online, writes selected token IDs
    directly to final plan shards, and stops once all quotas are complete.
    """

    document_root = Path(document_root)
    output_dir = Path(output_dir)
    manifest_path = document_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        prepared_manifest = json.load(handle)
    if prepared_manifest.get("complete") is not True:
        raise ValueError(f"Prepared corpus is incomplete: {document_root}")
    deduplication = prepared_manifest.get("deduplication") or {}
    if (
        prepared_manifest.get("deduplicated_globally") is not True
        or deduplication.get("scope") != "global"
    ):
        raise ValueError("Streaming planning requires a globally deduplicated corpus")
    raw_weights = prepared_manifest.get("raw_source_text_bytes")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError("Prepared manifest lacks raw_source_text_bytes")
    if prepared_manifest.get("raw_source_weight_scope") != (
        "all_input_records_before_dedup_split_filter"
    ):
        raise ValueError("Prepared source weights have the wrong scope")

    canonical_lengths = sorted(set(int(length) for length in lengths))
    requested_sources = (
        sorted(set(str(source) for source in sources))
        if sources is not None
        else sorted(str(source) for source in raw_weights)
    )
    excluded = set(str(source) for source in excluded_sources)
    selected_sources = [
        source for source in requested_sources if source not in excluded
    ]
    if not selected_sources:
        raise ValueError("No sources remain after exclusions")
    source_weights = {
        source: int(raw_weights[source]) for source in selected_sources
    }
    if source_weighting == "corpus_proportional":
        counts = solve_proportional_cell_counts(
            target_tokens,
            selected_sources,
            canonical_lengths,
            source_weights,
            seed=sample_seed,
        )
        source_weight_basis = "raw_utf8_text_bytes_before_dedup_split_filter"
    elif source_weighting == "balanced":
        counts = solve_exact_cell_counts(
            target_tokens,
            selected_sources,
            canonical_lengths,
            seed=sample_seed,
        )
        source_weight_basis = "balanced_sources"
    else:
        raise ValueError(f"unsupported source_weighting={source_weighting!r}")

    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists() and not overwrite:
        existing = load_sample_plan_manifest(output_dir)
        expected = {
            "target_tokens": int(target_tokens),
            "sample_seed": int(sample_seed),
            "tokenizer_hash": tokenizer_hash,
            "execution_world_size": int(execution_world_size),
            "streaming_sampling": STREAMING_SAMPLING_VERSION,
        }
        mismatches = {
            key: {"expected": value, "actual": existing["identity"].get(key)}
            for key, value in expected.items()
            if existing["identity"].get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Existing streaming plan does not match requested configuration: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return existing
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("plan-*.safetensors", ".plan-*.partial", "manifest.json"):
        for stale in output_dir.glob(pattern):
            stale.unlink(missing_ok=True)

    expected_pairs = sum(counts.values())
    rank_allocator = StreamingRankAllocator(
        counts,
        world_size=execution_world_size,
        target_tokens=target_tokens,
        seed=sample_seed,
    )
    planner = StreamingExactPlanner(
        counts,
        seed=sample_seed,
        max_document_reuses=max_document_reuses,
        rank_allocator=rank_allocator,
    )
    writer = SamplePlanWriter(
        output_dir,
        source_to_id={
            source: index for index, source in enumerate(selected_sources)
        },
        target_tokens=target_tokens,
        shard_token_limit=shard_token_limit,
    )
    rows_digest = hashlib.sha256()
    last_progress = 0
    try:
        documents = iter_prepared_document_records(document_root)
        for document, token_ids in iter_tokenized_prepared_documents(
            documents,
            tokenizer,
            batch_size=tokenizer_batch_size,
            batch_chars=tokenizer_batch_chars,
        ):
            for row in planner.add_document(document, token_ids):
                rows_digest.update(writer.add(row))
            if (
                progress_every_documents > 0
                and planner.documents_scanned - last_progress
                >= progress_every_documents
            ):
                last_progress = planner.documents_scanned
                print(
                    "[streaming-plan] "
                    f"documents={planner.documents_scanned} "
                    f"candidate_tokens={planner.candidate_tokens} "
                    f"selected_pairs={planner.selected_pairs}/{expected_pairs} "
                    f"selected_tokens={planner.selected_tokens}/{target_tokens} "
                    f"pending_pairs={planner.pending_pairs}",
                    flush=True,
                )
            if planner.complete:
                break
        stats = planner.finish(
            target_tokens=target_tokens,
            expected_pairs=expected_pairs,
        )
        identity = {
            "format_version": SAMPLE_PLAN_FORMAT,
            "target_tokens": int(target_tokens),
            "sample_seed": int(sample_seed),
            "tokenizer_hash": tokenizer_hash,
            "corpus_digest": prepared_corpus_digest(prepared_manifest),
            "sources": selected_sources,
            "chunk_lengths": canonical_lengths,
            "document_priority": prepared_manifest.get("document_order"),
            "boundary_sampling": STREAMING_BOUNDARY_VERSION,
            "row_hash": PLAN_ROW_HASH_VERSION,
            "source_balanced": source_weighting == "balanced",
            "source_weighting": source_weighting,
            "source_weight_basis": source_weight_basis,
            "source_token_weights": source_weights,
            "length_pair_balanced": True,
            "rank_assignment_in_plan": True,
            "execution_world_size": int(execution_world_size),
            "streaming_sampling": STREAMING_SAMPLING_VERSION,
        }
        manifest = writer.finish(
            identity=identity,
            rows_digest=rows_digest.hexdigest(),
            extra={
                "catalog": {
                    "enabled": False,
                    "reason": "online prepared-stream exact-quota selection",
                },
                "requested_cell_counts": {
                    f"{source}\t{length_a}\t{length_b}": count
                    for (source, length_a, length_b), count in sorted(
                        counts.items()
                    )
                },
                "source_weighting": source_weighting,
                "source_weight_basis": source_weight_basis,
                "source_token_weights": source_weights,
                "max_document_reuses": int(max_document_reuses),
                "corpus_position_overlap_policy": "forbidden",
                "corpus_position_overlap_verified": True,
                "unique_corpus_token_positions": int(target_tokens),
                "document_sampling": {
                    "algorithm": STREAMING_SAMPLING_VERSION,
                    "max_document_reuses": int(max_document_reuses),
                    **{
                        key: value
                        for key, value in stats.items()
                        if key != "execution_rank_assignment"
                    },
                },
                "execution_rank_assignment": stats[
                    "execution_rank_assignment"
                ],
                "input_provenance": {
                    "document_shard_dir": str(document_root.resolve()),
                    "document_shard_format": prepared_manifest.get("format"),
                    "document_shard_complete": True,
                    "prepared_corpus_digest": prepared_corpus_digest(
                        prepared_manifest
                    ),
                },
                "physical_shard_order": "pair_id_contiguous",
            },
        )
        load_sample_plan_manifest(
            output_dir,
            verify_logical_coverage=True,
        )
        return manifest
    except Exception:
        (output_dir / "manifest.json").unlink(missing_ok=True)
        raise
