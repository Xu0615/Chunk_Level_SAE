from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import zstandard as zstd

from chunk_saes.cache import ActivationCacheV2Writer, merge_v2_rank_manifests
from chunk_saes.data import normalized_content_hash
from chunk_saes.forward import process_pairs
from chunk_saes.modeling import LayerBatch
from chunk_saes.rank_shards import prepare_rank_document_shards
from chunk_saes.sample_plan import (
    build_sample_plan,
    iter_prepared_document_records,
    iter_sample_plan_rows,
)
from chunk_saes.utils import document_split


@dataclass(frozen=True)
class TinyPile:
    root: Path
    split_seed: int
    expected_hashes: dict[str, frozenset[str]]
    duplicate_train_hash: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class TinyPipelineV2:
    pile: TinyPile
    tokenizer_hash: str
    train_document_root: Path
    validation_document_root: Path
    train_plan_root: Path
    validation_plan_root: Path
    train_cache_root: Path
    validation_cache_root: Path
    train_plan_manifest: dict
    validation_plan_manifest: dict
    train_cache_manifest: dict
    validation_cache_manifest: dict
    hidden_size: int


class StableWhitespaceTokenizer:
    """Order-independent tokenizer suitable for deterministic CPU tests."""

    @staticmethod
    def _token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") & 0x7FFF_FFFF

    def __call__(self, text: str, **_kwargs) -> dict[str, list[int]]:
        return {"input_ids": [self._token_id(token) for token in text.split()]}

    def encode(self, text: str, **kwargs) -> list[int]:
        return self(text, **kwargs)["input_ids"]

    @staticmethod
    def decode(ids, **_kwargs) -> str:
        return " ".join(str(value) for value in ids)


class FakeLayerExtractor:
    """Deterministic CPU-only stand-in for the target transformer layer."""

    hidden_size = 4

    def forward_ids(self, sequences) -> LayerBatch:
        rows = [tuple(int(token) for token in sequence) for sequence in sequences]
        max_length = max(map(len, rows))
        hidden = torch.zeros((len(rows), max_length, self.hidden_size), dtype=torch.float32)
        mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
        for row_index, token_ids in enumerate(rows):
            ids = torch.tensor(token_ids, dtype=torch.int64)
            positions = torch.arange(len(token_ids), dtype=torch.int64)
            context = int(ids.sum().item()) % 59
            values = torch.stack(
                (
                    (ids.remainder(101).float() - 50.0) / 25.0,
                    (positions.float() + 1.0) / 8.0,
                    torch.full((len(token_ids),), context / 59.0),
                    ((ids * (positions + 1)).remainder(67).float() - 33.0) / 17.0,
                ),
                dim=-1,
            )
            hidden[row_index, : len(token_ids)] = values
            mask[row_index, : len(token_ids)] = True
        return LayerBatch(hidden=hidden, mask=mask)

    @staticmethod
    def close() -> None:
        return None


def _record(text: str, identifier: str, source: str) -> dict:
    return {
        "text": text,
        "meta": {"id": identifier, "pile_set_name": source},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl_zst(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload.encode("utf-8")))


def _text_for_split(split: str, seed: int, label: str) -> str:
    for candidate in range(20_000):
        words = [
            f"{label}Candidate{candidate}Token{token_index}"
            for token_index in range(40)
        ]
        text = " ".join(words)
        if document_split(normalized_content_hash(text), seed) == split:
            return text
    raise AssertionError(f"could not synthesize document for split={split}")


@pytest.fixture
def fake_tokenizer() -> StableWhitespaceTokenizer:
    return StableWhitespaceTokenizer()


@pytest.fixture
def tiny_multishard_pile(tmp_path: Path) -> TinyPile:
    split_seed = 73
    sources = ("Pile-CC", "Wikipedia (en)")
    texts: dict[str, dict[str, list[str]]] = {
        split: {
            source: [
                _text_for_split(
                    split,
                    split_seed,
                    f"{split.replace('validation', 'valid')}{source.replace(' ', '')}{index}",
                )
                for index in range(count)
            ]
            for source in sources
        }
        for split, count in (("train", 5), ("validation", 3), ("test", 2))
    }

    shard_a: list[dict] = []
    shard_b: list[dict] = []
    for split in ("train", "validation", "test"):
        for source in sources:
            for index, text in enumerate(texts[split][source]):
                record = _record(text, f"{split}-{source}-metadata-{index}", source)
                (shard_a if index % 2 == 0 else shard_b).append(record)

    duplicate = texts["train"]["Pile-CC"][0]
    shard_b.insert(
        0,
        _record(
            " \n " + duplicate.upper().replace(" ", "   ") + "\n",
            "different-metadata-id-for-normalized-duplicate",
            "Pile-CC",
        ),
    )

    dataset_root = tmp_path / "tiny-pile"
    _write_jsonl(dataset_root / "part-b.jsonl", list(reversed(shard_a)))
    _write_jsonl_zst(
        dataset_root / "nested" / "part-a.jsonl.zst",
        list(reversed(shard_b)),
    )
    (dataset_root / "README.md").write_text("not a data shard", encoding="utf-8")

    expected = {
        split: frozenset(
            normalized_content_hash(text)
            for source in sources
            for text in texts[split][source]
        )
        for split in ("train", "validation", "test")
    }
    return TinyPile(
        root=dataset_root,
        split_seed=split_seed,
        expected_hashes=expected,
        duplicate_train_hash=normalized_content_hash(duplicate),
        sources=sources,
    )


def _build_activation_cache(
    *,
    plan_root: Path,
    cache_root: Path,
    plan_manifest: dict,
    split: str,
    hidden_size: int,
) -> dict:
    sources = list(plan_manifest["identity"]["sources"])
    writer = ActivationCacheV2Writer(
        cache_root,
        rank=0,
        world_size=1,
        hidden_size=hidden_size,
        plan_digest=plan_manifest["plan_digest"],
        target_token_occurrences=plan_manifest["target_token_occurrences"],
        shard_token_limit=17,
        source_to_id={source: index for index, source in enumerate(sources)},
        activation_dtype=torch.float32,
    )
    extractor = FakeLayerExtractor()
    rows = list(iter_sample_plan_rows(plan_root))
    for start in range(0, len(rows), 3):
        process_pairs(extractor, writer, rows[start : start + 3])
    writer.finish()
    return merge_v2_rank_manifests(
        cache_root,
        world_size=1,
        plan_manifest=plan_manifest,
        verify_payload_checksums=True,
        extra={
            "project": "chunk-saes-test",
            "model": "fake-model",
            "model_hash": "fake-model-v1",
            "tokenizer_hash": "fake-tokenizer-v1",
            "layer": 1,
            "split": split,
            "independent_forwards": True,
            "padding_rows_stored": False,
            "position_policy": "reset_to_zero_per_independent_chunk",
        },
    )


@pytest.fixture
def tiny_pipeline_v2(
    tmp_path: Path,
    tiny_multishard_pile: TinyPile,
    fake_tokenizer: StableWhitespaceTokenizer,
) -> TinyPipelineV2:
    train_document_root = tmp_path / "prepared-train"
    validation_document_root = tmp_path / "prepared-validation"
    for split, output in (
        ("train", train_document_root),
        ("validation", validation_document_root),
    ):
        prepare_rank_document_shards(
            tiny_multishard_pile.root,
            output,
            split_seed=tiny_multishard_pile.split_seed,
            split=split,
            world_size=2,
            excluded_sources=set(),
            min_chars=1,
            compression_level=0,
            tokenizer_hash="fake-tokenizer-v1",
            order_seed=991,
        )

    train_plan_root = tmp_path / "plan-train"
    validation_plan_root = tmp_path / "plan-validation"
    train_plan_manifest = build_sample_plan(
        lambda: iter_prepared_document_records(train_document_root),
        fake_tokenizer,
        train_plan_root,
        tokenizer_hash="fake-tokenizer-v1",
        target_tokens=48,
        lengths=[2, 4],
        sample_seed=101,
        sources=tiny_multishard_pile.sources,
        shard_token_limit=13,
        max_document_reuses=2,
        input_provenance={"split": "train"},
    )
    validation_plan_manifest = build_sample_plan(
        lambda: iter_prepared_document_records(validation_document_root),
        fake_tokenizer,
        validation_plan_root,
        tokenizer_hash="fake-tokenizer-v1",
        target_tokens=48,
        lengths=[2, 4],
        sample_seed=202,
        sources=tiny_multishard_pile.sources,
        shard_token_limit=13,
        max_document_reuses=2,
        input_provenance={"split": "validation"},
    )

    train_cache_root = tmp_path / "cache-train"
    validation_cache_root = tmp_path / "cache-validation"
    train_cache_manifest = _build_activation_cache(
        plan_root=train_plan_root,
        cache_root=train_cache_root,
        plan_manifest=train_plan_manifest,
        split="train",
        hidden_size=FakeLayerExtractor.hidden_size,
    )
    validation_cache_manifest = _build_activation_cache(
        plan_root=validation_plan_root,
        cache_root=validation_cache_root,
        plan_manifest=validation_plan_manifest,
        split="validation",
        hidden_size=FakeLayerExtractor.hidden_size,
    )

    return TinyPipelineV2(
        pile=tiny_multishard_pile,
        tokenizer_hash="fake-tokenizer-v1",
        train_document_root=train_document_root,
        validation_document_root=validation_document_root,
        train_plan_root=train_plan_root,
        validation_plan_root=validation_plan_root,
        train_cache_root=train_cache_root,
        validation_cache_root=validation_cache_root,
        train_plan_manifest=train_plan_manifest,
        validation_plan_manifest=validation_plan_manifest,
        train_cache_manifest=train_cache_manifest,
        validation_cache_manifest=validation_cache_manifest,
        hidden_size=FakeLayerExtractor.hidden_size,
    )
