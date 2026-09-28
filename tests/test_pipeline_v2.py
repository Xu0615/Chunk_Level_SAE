from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from chunk_saes.corpus import dataset_fingerprint
from chunk_saes.data import iter_partitioned_documents
from chunk_saes.sample_plan import iter_sample_plan_rows


def _all_split_hashes(pile, split: str, world_size: int) -> list[str]:
    return [
        document.text_hash
        for _rank, document in iter_partitioned_documents(
            pile.root,
            split_seed=pile.split_seed,
            wanted_split=split,
            world_size=world_size,
            min_chars=1,
            order_seed=991,
        )
    ]


def _load_train_module():
    module = importlib.import_module("train_chunk_saes")
    return importlib.reload(module)


def test_multishard_hash_first_split_and_global_dedup(tiny_multishard_pile) -> None:
    fingerprint = dataset_fingerprint(tiny_multishard_pile.root)
    assert fingerprint["shard_count"] == 2
    assert [item["path"] for item in fingerprint["shards"]] == [
        "nested/part-a.jsonl.zst",
        "part-b.jsonl",
    ]

    observed = {}
    for split in ("train", "validation", "test"):
        hashes = _all_split_hashes(tiny_multishard_pile, split, world_size=3)
        assert len(hashes) == len(set(hashes))
        assert set(hashes) == tiny_multishard_pile.expected_hashes[split]
        observed[split] = set(hashes)

    assert observed["train"].isdisjoint(observed["validation"])
    assert observed["train"].isdisjoint(observed["test"])
    assert observed["validation"].isdisjoint(observed["test"])
    assert (
        _all_split_hashes(tiny_multishard_pile, "train", world_size=1).count(
            tiny_multishard_pile.duplicate_train_hash
        )
        == 1
    )
    assert set(_all_split_hashes(tiny_multishard_pile, "train", world_size=1)) == set(
        _all_split_hashes(tiny_multishard_pile, "train", world_size=5)
    )


def test_exact_plan_cache_v2_and_three_modes_share_occurrence_ids(
    tiny_pipeline_v2,
) -> None:
    train = _load_train_module()
    plan = tiny_pipeline_v2.train_plan_manifest
    cache = tiny_pipeline_v2.train_cache_manifest

    assert plan["token_occurrences"] == plan["target_token_occurrences"] == 48
    assert set(plan["source_tokens"].values()) == {24}
    assert cache["token_occurrences"] == cache["target_token_occurrences"] == 48
    assert cache["plan_digest"] == plan["plan_digest"]
    assert cache["activation_dtype"] == "float32"
    assert cache["coverage"] == {
        "pair_ids_complete": True,
        "pair_ids_unique": True,
        "occurrence_ranges_complete": True,
        "token_hidden_rows_equal_occurrences": True,
        "plan_row_digest_matches": True,
    }

    plan_rows = list(iter_sample_plan_rows(tiny_pipeline_v2.train_plan_root))
    validation_plan_rows = list(
        iter_sample_plan_rows(tiny_pipeline_v2.validation_plan_root)
    )
    assert {row.content_hash for row in plan_rows}.isdisjoint(
        {row.content_hash for row in validation_plan_rows}
    )
    assert sum(row.token_count for row in plan_rows) == 48
    assert plan["corpus_position_overlap_policy"] == "forbidden"
    assert plan["corpus_position_overlap_verified"] is True
    assert plan["unique_corpus_token_positions"] == 48
    intervals: dict[bytes, list[tuple[int, int]]] = {}
    for row in plan_rows:
        intervals.setdefault(row.content_hash, []).append(
            (row.start_a, row.start_b + row.length_b)
        )
    for document_intervals in intervals.values():
        ordered = sorted(document_intervals)
        assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    assert [row.pair_id for row in plan_rows] == list(range(len(plan_rows)))
    cursor = 0
    for row in plan_rows:
        assert row.occurrence_start == cursor
        cursor = row.occurrence_stop
    assert cursor == 48

    total_hidden_rows = 0
    for item in cache["ranks"][0]["shards"]:
        path = tiny_pipeline_v2.train_cache_root / item["path"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        offsets = tensors["chunk_offsets"].long()
        assert offsets.numel() == 2 * tensors["pair_id"].numel() + 1
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == tensors["token_hidden"].shape[0]
        assert bool((offsets[1:] > offsets[:-1]).all())
        total_hidden_rows += tensors["token_hidden"].shape[0]
        for index in range(tensors["pair_id"].numel()):
            start = int(offsets[2 * index])
            split = int(offsets[2 * index + 1])
            stop = int(offsets[2 * index + 2])
            hidden = tensors["token_hidden"][start:stop].float()
            torch.testing.assert_close(
                hidden[: split - start].mean(dim=0),
                tensors["mean_a"][index].float(),
            )
            torch.testing.assert_close(
                hidden[split - start :].mean(dim=0),
                tensors["mean_b"][index].float(),
            )
    assert total_hidden_rows == 48

    occurrence_orders: dict[str, list[int]] = {}
    cross_differs = False
    for mode in ("token", "temporal", "mean", "cross"):
        adapter = train.OccurrenceLoaderAdapter(
            tiny_pipeline_v2.train_cache_root,
            cache,
            rank=0,
            world_size=1,
            batch_size=7,
            seed=31337,
            device="cpu",
            mode=mode,
            finite=True,
        )
        ids: list[int] = []
        all_targets: list[torch.Tensor] = []
        for batch in adapter.iter_epoch(0):
            inputs, targets, side = train.select_occurrence_view(mode, batch)
            all_targets.append(targets)
            ids.extend(int(value) for value in batch["occurrence_id"].tolist())
            assert inputs.shape == targets.shape
            assert inputs.shape[1] == tiny_pipeline_v2.hidden_size
            assert side is not None and side.shape[0] == inputs.shape[0]
            if mode in {"token", "temporal", "mean"}:
                torch.testing.assert_close(inputs, targets)
            if mode == "temporal":
                previous, temporal_mask = train.select_temporal_pair_view(batch)
                assert previous.shape == inputs.shape
                assert temporal_mask.shape == (inputs.shape[0],)
            else:
                cross_differs = cross_differs or not torch.equal(inputs, targets)
        occurrence_orders[mode] = ids
        statistics_mode = "token" if mode == "temporal" else mode
        expected_mean = torch.tensor(
            cache["target_sufficient_statistics"]["mean_by_mode"][
                statistics_mode
            ]
        )
        torch.testing.assert_close(
            torch.cat(all_targets).mean(dim=0), expected_mean, rtol=1e-5, atol=1e-6
        )

    assert occurrence_orders["token"] == occurrence_orders["mean"]
    assert occurrence_orders["token"] == occurrence_orders["cross"]
    assert occurrence_orders["token"] == occurrence_orders["temporal"]
    assert sorted(occurrence_orders["token"]) == list(range(48))
    assert len(set(occurrence_orders["token"])) == 48
    assert cross_differs

    resumed = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        rank=0,
        world_size=1,
        batch_size=7,
        seed=31337,
        device="cpu",
        mode="token",
        finite=True,
    )
    resumed_ids = [
        int(value)
        for batch in resumed.iter_epoch(0, start_offset=14)
        for value in batch["occurrence_id"].tolist()
    ]
    assert resumed_ids == occurrence_orders["token"][14:]


def test_lazy_prefetched_loader_preserves_order_and_cleans_local_stage(
    tiny_pipeline_v2,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _load_train_module()
    cache = tiny_pipeline_v2.train_cache_manifest
    reference = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        rank=0,
        world_size=1,
        batch_size=7,
        seed=2026,
        device="cpu",
        mode="cross",
        finite=True,
        prefetch_shards=1,
        prefetch_workers=1,
        prefetch_batches=0,
        materialize_shards=True,
    )
    expected = [
        {
            key: value.clone()
            for key, value in batch.items()
        }
        for batch in reference.iter_epoch(0)
    ]

    def full_expansion_must_not_run(*args, **kwargs):
        raise AssertionError("the streaming loader must not expand full shards")

    monkeypatch.setattr(
        train.PackedV2OccurrenceLoader,
        "_expand",
        staticmethod(full_expansion_must_not_run),
    )
    stage_root = tmp_path / "stage"
    optimized = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        rank=0,
        world_size=1,
        batch_size=7,
        seed=2026,
        device="cpu",
        mode="cross",
        finite=True,
        prefetch_shards=3,
        prefetch_workers=2,
        prefetch_batches=2,
        materialize_shards=True,
        local_cache_dir=stage_root,
    )
    observed = list(optimized.iter_epoch(0))
    assert len(observed) == len(expected)
    for left, right in zip(observed, expected, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])

    token_staged = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        rank=0,
        world_size=1,
        batch_size=7,
        seed=2026,
        device="cpu",
        mode="token",
        finite=True,
        prefetch_shards=3,
        prefetch_workers=2,
        prefetch_batches=2,
        materialize_shards=True,
        local_cache_dir=stage_root,
    )
    token_ids = [
        int(value)
        for batch in token_staged.iter_epoch(0)
        for value in batch["occurrence_id"].tolist()
    ]
    assert sorted(token_ids) == list(range(48))
    assert not list(stage_root.rglob("*.safetensors"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_resident_shard_loader_matches_cpu_order_and_values(
    tiny_pipeline_v2,
) -> None:
    train = _load_train_module()
    cache = tiny_pipeline_v2.train_cache_manifest
    common = {
        "rank": 0,
        "world_size": 1,
        "batch_size": 7,
        "seed": 2027,
        "mode": "cross",
        "finite": True,
        "prefetch_shards": 2,
        "prefetch_workers": 2,
        "prefetch_batches": 0,
        "materialize_shards": True,
        "pin_memory": True,
    }
    cpu = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        device="cpu",
        gpu_shards=0,
        **common,
    )
    gpu = train.OccurrenceLoaderAdapter(
        tiny_pipeline_v2.train_cache_root,
        cache,
        device="cuda:0",
        gpu_shards=2,
        **common,
    )
    expected = list(cpu.iter_epoch(0))
    observed = list(gpu.iter_epoch(0))
    assert len(observed) == len(expected)
    for left, right in zip(observed, expected, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key].cpu(), right[key])
    assert len(gpu.loader._gpu_cache) <= 2

def test_cpu_training_validation_metrics_and_tensorboard(
    tiny_pipeline_v2,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _load_train_module()
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    output_root = tmp_path / "trained"
    tensorboard_root = tmp_path / "runs"
    argv = [
        "train_chunk_saes.py",
        "--activation-cache-dir",
        str(tiny_pipeline_v2.train_cache_root),
        "--validation-cache-dir",
        str(tiny_pipeline_v2.validation_cache_root),
        "--output-dir",
        str(output_root),
        "--model",
        "fake-model",
        "--layer",
        "1",
        "--modes",
        "token,temporal,mean,cross",
        "--dict-size",
        "8",
        "--k",
        "2",
        "--global-batch-size",
        "8",
        "--steps",
        "6",
        "--lr",
        "0.001",
        "--warmup-steps",
        "1",
        "--threshold-beta",
        "0.9",
        "--autocast-dtype",
        "float32",
        "--normalization-samples",
        "0",
        "--log-every",
        "1",
        "--validate-every",
        "1",
        "--validation-samples",
        "0",
        "--validation-batch-size",
        "8",
        "--save-every",
        "1",
        "--run-name",
        "tiny-cpu-v2",
        "--tensorboard-dir",
        str(tensorboard_root),
        "--tensorboard-flush-secs",
        "1",
        "--tensorboard-max-queue",
        "1",
        "--fidelity-reference-fves",
        "token=1.0,temporal=1.0,mean=1.0,cross=1.0",
        "--overwrite-output",
        "--require-exact-coverage",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    for mode in ("token", "temporal", "mean", "cross"):
        mode_root = output_root / mode
        assert (mode_root / "complete.json").is_file()
        assert (mode_root / "sae.safetensors").is_file()
        assert (mode_root / "checkpoints" / "latest" / "training_state.pt").is_file()
        best_manifest = json.loads(
            (
                mode_root
                / "checkpoints"
                / "best"
                / "checkpoint_manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert best_manifest["resumable"] is False

        rows = [
            json.loads(line)
            for line in (mode_root / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        train_rows = [row for row in rows if row["split"] == "train"]
        validation_rows = [row for row in rows if row["split"] == "validation"]
        assert [row["step"] for row in train_rows] == [1, 2, 3, 4, 5, 6]
        assert [row["step"] for row in validation_rows] == [0, 1, 2, 3, 4, 5, 6]
        for row in validation_rows:
            mse = float(row["validation/raw_mse"])
            baseline = float(row["validation/mean_predictor_mse"])
            nmse = float(row["validation/normalized_mse"])
            fve = float(row["validation/fve"])
            assert math.isfinite(mse) and mse >= 0.0
            assert math.isfinite(baseline) and baseline > 0.0
            assert nmse == pytest.approx(mse / baseline, rel=1e-6, abs=1e-8)
            assert fve == pytest.approx(1.0 - nmse, rel=1e-6, abs=1e-8)

        event_root = tensorboard_root / "tiny-cpu-v2" / mode
        event_files = list(event_root.glob("events.out.tfevents.*"))
        assert event_files, f"missing TensorBoard event for {mode}"
        accumulator = EventAccumulator(str(event_root))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags()["scalars"])
        directional_tags = {
            tag
            for tag in train.CORE_TENSORBOARD_SCALARS
            if "/a_to_b/" in tag or "/b_to_a/" in tag
        }
        expected_tags = set(train.CORE_TENSORBOARD_SCALARS) - directional_tags
        if mode == "cross":
            expected_tags |= directional_tags
        else:
            expected_tags.discard("cross/validation_fve_volatility_20")
        temporal_tags = {
            "train/temporal_contrastive_loss",
            "train/temporal_contrastive_accuracy",
            "train/temporal_pairs",
        }
        if mode != "temporal":
            expected_tags -= temporal_tags
        assert scalar_tags == expected_tags
        assert [event.step for event in accumulator.Scalars("train/fve")] == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert [
            event.step for event in accumulator.Scalars("validation/fve")
        ] == [0, 1, 2, 3, 4, 5, 6]
        assert (
            accumulator.Scalars("progress/unique_occurrence_coverage")[-1].value
            == 1.0
        )
        complete = json.loads((mode_root / "complete.json").read_text(encoding="utf-8"))
        assert complete["exact_coverage"] is True
        assert complete["coverage_fraction"] == 1.0
        assert complete["final_full_validation_metrics"]["validation/samples"] == 48
        if mode == "cross":
            for tag in ("validation/a_to_b/fve", "validation/b_to_a/fve"):
                assert [
                    event.step for event in accumulator.Scalars(tag)
                ] == [0, 1, 2, 3, 4, 5, 6]
