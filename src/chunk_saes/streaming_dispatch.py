from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from .sample_plan import PlanRow


_HEADER_TAG = 7100
_METADATA_TAG = 7101
_HASHES_TAG = 7102
_OFFSETS_TAG = 7103
_TOKENS_TAG = 7104


@dataclass
class SerializedRows:
    metadata: torch.Tensor
    hashes: torch.Tensor
    offsets: torch.Tensor
    token_ids: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.metadata.shape[0])


def serialize_rows(
    rows: Sequence[PlanRow],
    *,
    source_to_id: Mapping[str, int],
) -> SerializedRows:
    metadata = torch.empty((len(rows), 10), dtype=torch.int64)
    hashes = torch.empty((len(rows), 64), dtype=torch.uint8)
    offsets = [0]
    token_ids: list[int] = []
    for index, row in enumerate(rows):
        metadata[index] = torch.tensor(
            [
                row.pair_id,
                row.occurrence_start,
                row.start_a,
                row.start_b,
                row.document_token_count,
                row.document_reuse_index,
                row.execution_rank,
                row.length_a,
                row.length_b,
                int(source_to_id[row.source]),
            ],
            dtype=torch.int64,
        )
        hashes[index, :32] = torch.tensor(list(row.doc_hash), dtype=torch.uint8)
        hashes[index, 32:] = torch.tensor(list(row.content_hash), dtype=torch.uint8)
        token_ids.extend(row.input_ids_a)
        offsets.append(len(token_ids))
        token_ids.extend(row.input_ids_b)
        offsets.append(len(token_ids))
    return SerializedRows(
        metadata=metadata.contiguous(),
        hashes=hashes.contiguous(),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        token_ids=torch.tensor(token_ids, dtype=torch.int32),
    )


def deserialize_rows(
    payload: SerializedRows,
    *,
    id_to_source: Sequence[str],
) -> list[PlanRow]:
    result: list[PlanRow] = []
    metadata = payload.metadata.tolist()
    for index, values in enumerate(metadata):
        (
            pair_id,
            occurrence_start,
            start_a,
            start_b,
            document_token_count,
            reuse_index,
            execution_rank,
            length_a,
            length_b,
            source_id,
        ) = (int(value) for value in values)
        token_start = int(payload.offsets[2 * index])
        split = int(payload.offsets[2 * index + 1])
        token_stop = int(payload.offsets[2 * index + 2])
        if split - token_start != length_a or token_stop - split != length_b:
            raise ValueError("streaming dispatch offsets do not match chunk lengths")
        row_tokens = payload.token_ids[token_start:token_stop].tolist()
        result.append(
            PlanRow(
                pair_id=pair_id,
                occurrence_start=occurrence_start,
                doc_hash=bytes(payload.hashes[index, :32].tolist()),
                content_hash=bytes(payload.hashes[index, 32:].tolist()),
                source=id_to_source[source_id],
                input_ids_a=tuple(int(value) for value in row_tokens[:length_a]),
                input_ids_b=tuple(int(value) for value in row_tokens[length_a:]),
                start_a=start_a,
                start_b=start_b,
                document_token_count=document_token_count,
                document_reuse_index=reuse_index,
                execution_rank=execution_rank,
            )
        )
    return result


def send_rows(
    rows: Sequence[PlanRow] | None,
    *,
    destination: int,
    group,
    source_to_id: Mapping[str, int],
) -> None:
    if rows is None:
        header = torch.tensor([-1, 0], dtype=torch.int64)
        dist.send(header, dst=destination, group=group, tag=_HEADER_TAG)
        return
    payload = serialize_rows(rows, source_to_id=source_to_id)
    header = torch.tensor(
        [payload.rows, payload.token_ids.numel()],
        dtype=torch.int64,
    )
    dist.send(header, dst=destination, group=group, tag=_HEADER_TAG)
    dist.send(payload.metadata, dst=destination, group=group, tag=_METADATA_TAG)
    dist.send(payload.hashes, dst=destination, group=group, tag=_HASHES_TAG)
    dist.send(payload.offsets, dst=destination, group=group, tag=_OFFSETS_TAG)
    dist.send(payload.token_ids, dst=destination, group=group, tag=_TOKENS_TAG)


def receive_rows(
    *,
    source: int,
    group,
    id_to_source: Sequence[str],
) -> list[PlanRow] | None:
    header = torch.empty(2, dtype=torch.int64)
    dist.recv(header, src=source, group=group, tag=_HEADER_TAG)
    rows, tokens = (int(value) for value in header.tolist())
    if rows < 0:
        return None
    payload = SerializedRows(
        metadata=torch.empty((rows, 10), dtype=torch.int64),
        hashes=torch.empty((rows, 64), dtype=torch.uint8),
        offsets=torch.empty(2 * rows + 1, dtype=torch.int64),
        token_ids=torch.empty(tokens, dtype=torch.int32),
    )
    dist.recv(payload.metadata, src=source, group=group, tag=_METADATA_TAG)
    dist.recv(payload.hashes, src=source, group=group, tag=_HASHES_TAG)
    dist.recv(payload.offsets, src=source, group=group, tag=_OFFSETS_TAG)
    dist.recv(payload.token_ids, src=source, group=group, tag=_TOKENS_TAG)
    return deserialize_rows(payload, id_to_source=id_to_source)


class RowWindowReceiver:
    """Continuously receive Gloo windows while the main thread runs GPU forwards."""

    _END = object()

    def __init__(
        self,
        *,
        source: int,
        group,
        id_to_source: Sequence[str],
        queue_depth: int = 4,
    ) -> None:
        self.source = source
        self.group = group
        self.id_to_source = tuple(id_to_source)
        self.queue: queue.Queue[object] = queue.Queue(maxsize=max(1, queue_depth))
        self.failure: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="streaming-plan-receiver",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                rows = receive_rows(
                    source=self.source,
                    group=self.group,
                    id_to_source=self.id_to_source,
                )
                if rows is None:
                    self.queue.put(self._END)
                    return
                self.queue.put(rows)
        except BaseException as error:
            self.failure = error
            self.queue.put(self._END)

    def __iter__(self):
        return self

    def __next__(self) -> list[PlanRow]:
        value = self.queue.get()
        if value is self._END:
            self.thread.join(timeout=1.0)
            if self.failure is not None:
                raise self.failure
            raise StopIteration
        assert isinstance(value, list)
        return value
