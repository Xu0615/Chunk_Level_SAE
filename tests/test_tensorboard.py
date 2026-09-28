from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from chunk_saes.metrics import DASHBOARD_SCALAR_TAGS, TensorBoardLogger
from chunk_saes.sae import (
    BatchTopKSAE,
    load_training_checkpoint,
    save_inference_checkpoint,
    save_training_checkpoint,
    save_training_checkpoint_snapshot,
    snapshot_training_checkpoint,
)


def scalar_values(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def test_tensorboard_event_contains_required_scalars(tmp_path: Path) -> None:
    log_dir = tmp_path / "runs" / "run-1" / "cross"
    logger = TensorBoardLogger(log_dir, rank=0, flush_secs=1, max_queue=1)
    try:
        logger.add_scalars(
            {
                "train/scaled_objective": 2.5,
                "train/raw_mse": 0.25,
                "optimizer/lr": 1e-4,
                "optimizer/updates": 7,
                "progress/token_occurrences_seen": 1024,
                "validation/raw_mse": 0.2,
                "validation/mean_predictor_mse": 0.4,
                "validation/normalized_mse": 0.5,
                "validation/fve": 0.5,
                "validation/a_to_b/fve": 0.4,
                "validation/b_to_a/fve": 0.6,
            },
            step=7,
        )
    finally:
        logger.close()

    values = scalar_values(log_dir)
    required = {
        "train/scaled_objective",
        "train/raw_mse",
        "optimizer/lr",
        "optimizer/updates",
        "progress/token_occurrences_seen",
        "validation/raw_mse",
        "validation/mean_predictor_mse",
        "validation/normalized_mse",
        "validation/fve",
        "validation/a_to_b/fve",
        "validation/b_to_a/fve",
    }
    assert required <= values.keys()
    assert values["validation/fve"] == pytest.approx([(7, 0.5)])


def test_only_rank_zero_writes_tensorboard_events(tmp_path: Path) -> None:
    log_dir = tmp_path / "rank-one-must-not-exist"
    logger = TensorBoardLogger(log_dir, rank=1)
    logger.add_scalars({"train/raw_mse": 1.0}, step=1)
    logger.close()
    assert not log_dir.exists()


def test_tensorboard_scalar_allowlist_keeps_dashboard_compact(tmp_path: Path) -> None:
    log_dir = tmp_path / "core-scalars"
    logger = TensorBoardLogger(
        log_dir,
        rank=0,
        flush_secs=1,
        max_queue=1,
        scalar_allowlist={"train/fve", "validation/fve"},
    )
    try:
        logger.add_scalars(
            {
                "train/raw_mse": 0.25,
                "train/fve": 0.75,
                "validation/normalized_mse": 0.3,
                "validation/fve": 0.7,
            },
            step=10,
        )
    finally:
        logger.close()

    values = scalar_values(log_dir)
    assert values.keys() == {"train/fve", "validation/fve"}
    assert values["train/fve"][0][0] == 10
    assert values["train/fve"][0][1] == pytest.approx(0.75)
    assert values["validation/fve"][0][0] == 10
    assert values["validation/fve"][0][1] == pytest.approx(0.7)


def test_tensorboard_dashboard_derives_comparable_metrics(tmp_path: Path) -> None:
    log_dir = tmp_path / "dashboard"
    logger = TensorBoardLogger(
        log_dir,
        rank=0,
        flush_secs=1,
        max_queue=1,
        scalar_allowlist={
            "comparison/train_attainable_fidelity",
            "comparison/validation_attainable_fidelity",
            "comparison/generalization_gap",
            "comparison/remaining_regret",
            "optimization/grad_norm_to_clip",
            "sparsity/effective_l0",
            "cross/validation_fve_volatility_20",
        },
        mode="cross",
        fidelity_reference_fve=0.5,
        gradient_clip=1.0,
    )
    try:
        logger.add_scalars(
            {
                "train/fve": 0.4,
                "optimizer/grad_norm": 2.0,
                "train/effective_l0": 128.0,
            },
            step=1,
        )
        logger.add_scalars({"validation/fve": 0.25}, step=1)
        logger.add_scalars({"validation/fve": 0.35}, step=2)
    finally:
        logger.close()

    values = scalar_values(log_dir)
    assert values["comparison/train_attainable_fidelity"][-1][0] == 1
    assert values["comparison/train_attainable_fidelity"][-1][1] == pytest.approx(
        0.8
    )
    assert [
        step
        for step, _ in values["comparison/validation_attainable_fidelity"]
    ] == [1, 2]
    assert [
        value
        for _, value in values[
            "comparison/validation_attainable_fidelity"
        ]
    ] == pytest.approx([0.5, 0.7])
    assert values["comparison/generalization_gap"][-1][1] == pytest.approx(0.1)
    assert values["comparison/remaining_regret"][-1][1] == pytest.approx(0.3)
    assert values["optimization/grad_norm_to_clip"][-1][1] == pytest.approx(2.0)
    assert values["sparsity/effective_l0"][-1][1] == pytest.approx(128.0)
    assert values["cross/validation_fve_volatility_20"][-1][1] > 0.0


def test_complete_dashboard_scalar_surface_is_diagnostic_and_nonduplicative() -> None:
    required = {
        "train/scaled_objective",
        "train/fve",
        "train/auxiliary_loss",
        "validation/fve",
        "validation/nmse",
        "validation/a_to_b/fve",
        "validation/b_to_a/fve",
        "validation_full/fve",
        "sparsity/dead_features",
        "sparsity/dead_fraction",
        "sparsity/auxk_candidates",
        "optimizer/grad_norm",
        "optimization/grad_norm_to_clip",
        "progress/unique_occurrence_coverage",
        "filter/rejected_rows",
        "system/samples_per_second",
        "comparison/final_validation_attainable_fidelity",
        "cross/validation_fve_volatility_20",
    }
    assert required <= DASHBOARD_SCALAR_TAGS
    assert not any(
        tag.endswith(("/raw_sse", "/scaled_sse", "/elements", "/samples"))
        for tag in DASHBOARD_SCALAR_TAGS
    )


def test_tensorboard_values_and_steps_match_jsonl(tmp_path: Path) -> None:
    log_dir = tmp_path / "events"
    jsonl = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 1, "train/raw_mse": 3.0, "optimizer/lr": 1e-4},
        {"step": 5, "train/raw_mse": 2.0, "optimizer/lr": 5e-5},
    ]
    logger = TensorBoardLogger(log_dir, rank=0)
    try:
        with jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                logger.add_scalars(
                    {key: value for key, value in row.items() if key != "step"},
                    row["step"],
                )
    finally:
        logger.close()

    values = scalar_values(log_dir)
    assert [step for step, _ in values["train/raw_mse"]] == [1, 5]
    assert [value for _, value in values["train/raw_mse"]] == pytest.approx(
        [3.0, 2.0]
    )
    parsed = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert [row["step"] for row in parsed] == [1, 5]


def test_training_checkpoint_restores_model_optimizer_rng_and_config(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    reconstructed, _, _ = model(inputs, batch_topk=True)
    loss = reconstructed.square().mean()
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "latest"
    state = {
        "step": 13,
        "rng_by_rank": [{"torch_cpu": torch.get_rng_state()}],
        "config": {
            "mode": "token",
            "train_cache_digest": "abc",
            "run_id": "run-1",
        },
    }
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    save_training_checkpoint(model, optimizer, checkpoint, state)

    restored_model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9e-2)
    restored = load_training_checkpoint(
        checkpoint, restored_model, restored_optimizer
    )

    assert restored["step"] == 13
    assert restored["config"]["train_cache_digest"] == "abc"
    for key, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[key])
    assert restored_optimizer.state_dict()["state"]
    assert (checkpoint / "config.json").is_file()


def test_async_checkpoint_snapshot_is_detached_and_resumable(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    loss = model(inputs, batch_topk=True)[0].square().mean()
    loss.backward()
    optimizer.step()
    state = {
        "step": 7,
        "samples_seen": 224,
        "world_size": 1,
        "feature_counts_by_rank": [torch.arange(8, dtype=torch.int64)],
        "num_occurrences_since_fired_by_rank": [
            torch.arange(8, dtype=torch.int64) * 32
        ],
        "rng_by_rank": [{"torch_cpu": torch.get_rng_state()}],
        "config": {"mode": "cross", "run_id": "async-test"},
    }
    live_optimizer_tensors = {
        (parameter, key): value
        for parameter, values in optimizer.state.items()
        for key, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    tensors, optimizer_state, detached_state = snapshot_training_checkpoint(
        model, optimizer, state
    )
    for (parameter, key), value in live_optimizer_tensors.items():
        assert optimizer.state[parameter][key] is value
        assert optimizer.state[parameter][key].data_ptr() == value.data_ptr()
    expected_tensors = {key: value.clone() for key, value in tensors.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    checkpoint = tmp_path / "latest"
    save_training_checkpoint_snapshot(
        tensors, optimizer_state, checkpoint, detached_state
    )

    restored_model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9e-2)
    restored_state = load_training_checkpoint(
        checkpoint, restored_model, restored_optimizer
    )
    assert restored_state["step"] == 7
    assert restored_state["num_occurrences_since_fired_by_rank"][0].tolist() == [
        value * 32 for value in range(8)
    ]
    for key, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_tensors[key])
    assert restored_optimizer.state_dict()["state"]

    optimizer.zero_grad(set_to_none=True)
    model(inputs, batch_topk=True)[0].square().mean().backward()
    optimizer.step()


def test_async_checkpoint_local_staging_mirrors_verified_artifact(
    tmp_path: Path,
) -> None:
    model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    model(inputs, batch_topk=True)[0].square().mean().backward()
    optimizer.step()
    state = {
        "step": 3,
        "samples_seen": 96,
        "world_size": 1,
        "config": {"mode": "mean", "run_id": "staged-test"},
    }
    tensors, optimizer_state, detached_state = snapshot_training_checkpoint(
        model, optimizer, state
    )
    staging = tmp_path / "local" / "latest"
    published = tmp_path / "ceph" / "latest"
    save_training_checkpoint_snapshot(
        tensors,
        optimizer_state,
        published,
        detached_state,
        staging,
    )

    for name in (
        "sae.safetensors",
        "optimizer.pt",
        "training_state.pt",
        "config.json",
        "checkpoint_manifest.json",
    ):
        assert (staging / name).read_bytes() == (published / name).read_bytes()
    restored_model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=9e-2)
    restored = load_training_checkpoint(
        published, restored_model, restored_optimizer
    )
    assert restored["step"] == 3


def test_weights_only_best_checkpoint_is_not_resumable(tmp_path: Path) -> None:
    model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    tensors = model.checkpoint_tensors()
    checkpoint = tmp_path / "best"
    state = {
        "step": 17,
        "samples_seen": 136,
        "world_size": 1,
        "best_metric_value": 0.25,
        "best_step": 17,
        "config": {
            "mode": "token",
            "activation_dim": 4,
            "dict_size": 8,
            "k": 2,
        },
    }
    save_inference_checkpoint(tensors, checkpoint, state)

    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resumable"] is False
    assert set(manifest["files"]) == {"sae.safetensors", "config.json"}
    restored_model = BatchTopKSAE(activation_dim=4, dict_size=8, k=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="not resumable"):
        load_training_checkpoint(
            checkpoint,
            restored_model,
            restored_optimizer,
        )
