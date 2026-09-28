from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .data import Document
from .sample_plan import PreparedDocument
from .utils import content_hash, document_split


HF_PILE_DATASET = "EleutherAI/the_pile_deduplicated"
HF_PILE_REVISION = "fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb"
HF_PILE_SOURCE = "Pile-Deduplicated"
HF_STREAMING_FORMAT = "chunk-saes-hf-deduplicated-stream-v1"


@dataclass(frozen=True)
class HFStreamingCorpus:
    dataset: str = HF_PILE_DATASET
    revision: str = HF_PILE_REVISION
    physical_split: str = "train"
    logical_split: str = "train"
    split_seed: int = 42
    shuffle_seed: int = 71
    shuffle_buffer: int = 16_384
    source: str = HF_PILE_SOURCE

    def identity(self) -> dict[str, object]:
        return {
            "format": HF_STREAMING_FORMAT,
            "dataset": self.dataset,
            "revision": self.revision,
            "physical_split": self.physical_split,
            "logical_split": self.logical_split,
            "split_seed": int(self.split_seed),
            "shuffle": {
                "algorithm": "datasets.IterableDataset.shuffle",
                "seed": int(self.shuffle_seed),
                "buffer_size": int(self.shuffle_buffer),
            },
            "already_deduplicated": True,
            "source_metadata_available": False,
            "source": self.source,
            "text_field": "text",
        }


def _iter_records(
    spec: HFStreamingCorpus,
    *,
    rank: int,
    world_size: int,
) -> Iterator[tuple[int, str, str]]:
    if spec.logical_split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported logical split: {spec.logical_split}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} outside [0, {world_size})")
    from datasets import load_dataset

    dataset = load_dataset(
        spec.dataset,
        split=spec.physical_split,
        revision=spec.revision,
        streaming=True,
    )
    if spec.shuffle_buffer > 1:
        dataset = dataset.shuffle(
            seed=spec.shuffle_seed,
            buffer_size=spec.shuffle_buffer,
        )
    if world_size > 1:
        dataset = dataset.shard(
            num_shards=world_size,
            index=rank,
            contiguous=False,
        )
    iterator = iter(dataset)
    try:
        for ordinal, record in enumerate(iterator):
            text = record.get("text")
            if not isinstance(text, str) or not text:
                continue
            digest = content_hash(text)
            if document_split(digest, spec.split_seed) != spec.logical_split:
                continue
            yield ordinal, text, digest
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def iter_hf_prepared_documents(
    spec: HFStreamingCorpus,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[PreparedDocument]:
    for ordinal, text, digest in _iter_records(
        spec,
        rank=rank,
        world_size=world_size,
    ):
        yield PreparedDocument(
            stream_id=rank,
            ordinal=ordinal,
            doc_id=f"hf:{digest}",
            source=spec.source,
            text=text,
            content_hash=digest,
        )


def iter_hf_documents(
    spec: HFStreamingCorpus,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[Document]:
    for _ordinal, text, digest in _iter_records(
        spec,
        rank=rank,
        world_size=world_size,
    ):
        yield Document(
            doc_id=f"hf:{digest}",
            source=spec.source,
            split=spec.logical_split,
            text=text,
            text_hash=digest,
        )


def verify_hf_stream(spec: HFStreamingCorpus) -> dict[str, object]:
    iterator = _iter_records(spec, rank=0, world_size=1)
    try:
        ordinal, text, digest = next(iterator)
    except StopIteration as error:
        raise ValueError(
            f"HF streaming corpus produced no {spec.logical_split} documents"
        ) from error
    finally:
        iterator.close()
    return {
        **spec.identity(),
        "verified": True,
        "first_matching_ordinal": ordinal,
        "first_matching_text_bytes": len(text.encode("utf-8")),
        "first_matching_content_hash": digest,
    }
