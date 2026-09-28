from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from chunk_saes.metrics import ReconstructionMetricAccumulator


def load_train_module():
    path = Path(__file__).parents[1] / "src" / "train_chunk_saes.py"
    spec = importlib.util.spec_from_file_location("train_chunk_saes_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    train_mean: torch.Tensor,
    *,
    scale: float = 1.0,
) -> dict[str, float | int | bool]:
    accumulator = ReconstructionMetricAccumulator()
    accumulator.update(
        reconstruction * scale,
        target,
        activation_scale=scale,
        mean_predictor_raw=train_mean,
    )
    return accumulator.summary().as_dict()


def test_reconstruction_metrics_perfect_baseline_and_negative_fve() -> None:
    target = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    train_mean = torch.tensor([1.0, 1.0])

    perfect = summarize(target, target, train_mean)
    assert perfect["raw_mse"] == pytest.approx(0.0)
    assert perfect["normalized_mse"] == pytest.approx(0.0)
    assert perfect["fve"] == pytest.approx(1.0)

    baseline = summarize(train_mean.expand_as(target), target, train_mean)
    assert baseline["raw_mse"] == pytest.approx(1.0)
    assert baseline["mean_predictor_mse"] == pytest.approx(1.0)
    assert baseline["normalized_mse"] == pytest.approx(1.0)
    assert baseline["fve"] == pytest.approx(0.0)

    worse = summarize(torch.full_like(target, 4.0), target, train_mean)
    assert float(worse["normalized_mse"]) > 1.0
    assert float(worse["fve"]) < 0.0


def test_metric_accumulator_is_batch_partition_invariant_and_scale_correct() -> None:
    target = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    )
    reconstruction = target + torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    train_mean = torch.tensor([3.0, 4.0])

    whole = ReconstructionMetricAccumulator()
    whole.update(
        reconstruction * 3.0,
        target,
        activation_scale=3.0,
        mean_predictor_raw=train_mean,
    )

    partitioned = ReconstructionMetricAccumulator()
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        partitioned.update(
            reconstruction[start:stop] * 3.0,
            target[start:stop],
            activation_scale=3.0,
            mean_predictor_raw=train_mean,
        )

    partitioned_metrics = partitioned.summary().as_dict()
    metrics = whole.summary().as_dict()
    for key in metrics:
        if isinstance(metrics[key], float) and math.isnan(metrics[key]):
            assert math.isnan(float(partitioned_metrics[key]))
        else:
            assert partitioned_metrics[key] == pytest.approx(metrics[key])
    assert metrics["scaled_objective"] == pytest.approx(
        float(metrics["raw_mse"]) * target.shape[1] * 3.0**2
    )


def test_metrics_use_frozen_train_mean_not_validation_mean() -> None:
    target = torch.tensor([[10.0], [12.0]])
    reconstruction = torch.full_like(target, 11.0)
    frozen_train_mean = torch.tensor([0.0])
    metrics = summarize(reconstruction, target, frozen_train_mean)

    assert metrics["raw_mse"] == pytest.approx(1.0)
    assert metrics["mean_predictor_mse"] == pytest.approx(122.0)
    assert metrics["normalized_mse"] == pytest.approx(1.0 / 122.0)


def test_zero_variance_baseline_is_explicitly_degenerate() -> None:
    target = torch.ones(3, 2)
    metrics = summarize(target, target, torch.ones(2))
    assert metrics["baseline_degenerate"] is True
    assert math.isnan(float(metrics["normalized_mse"]))
    assert math.isnan(float(metrics["fve"]))


def test_occurrence_matched_four_views_and_directions() -> None:
    train = load_train_module()
    token = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    chunk_mean = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
    )
    partner_mean = chunk_mean + 10.0
    side = torch.tensor([False, True, False])
    batch = {
        "token_hidden": token,
        "chunk_mean": chunk_mean,
        "partner_mean": partner_mean,
        "side": side,
    }

    token_input, token_target, token_side = train.select_occurrence_view(
        "token", batch
    )
    temporal_input, temporal_target, temporal_side = train.select_occurrence_view(
        "temporal", batch
    )
    mean_input, mean_target, mean_side = train.select_occurrence_view("mean", batch)
    cross_input, cross_target, cross_side = train.select_occurrence_view(
        "cross", batch
    )

    assert torch.equal(token_input, token)
    assert torch.equal(token_target, token)
    assert torch.equal(temporal_input, token)
    assert torch.equal(temporal_target, token)
    assert torch.equal(mean_input, chunk_mean)
    assert torch.equal(mean_target, chunk_mean)
    assert torch.equal(cross_input, chunk_mean)
    assert torch.equal(cross_target, partner_mean)
    assert torch.equal(token_side, side)
    assert torch.equal(temporal_side, side)
    assert torch.equal(mean_side, side)
    assert torch.equal(cross_side, side)


def test_packed_v2_expansion_preserves_occurrence_ids_and_partner_means() -> None:
    train = load_train_module()
    tensors = {
        "token_hidden": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "length_a": torch.tensor([2, 1]),
        "length_b": torch.tensor([1, 1]),
        "mean_a": torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
        "mean_b": torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        "pair_id": torch.tensor([7, 8]),
        "occurrence_start": torch.tensor([100, 103]),
    }
    rows = train.PackedV2OccurrenceLoader._expand(tensors)

    assert rows["occurrence_id"].tolist() == [100, 101, 102, 103, 104]
    assert rows["side"].tolist() == [False, False, True, False, True]
    assert rows["pair_id"].tolist() == [7, 7, 7, 8, 8]
    assert torch.equal(
        rows["chunk_mean"],
        torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        ),
    )
    assert torch.equal(
        rows["partner_mean"],
        torch.tensor(
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [4.0, 4.0], [3.0, 3.0]]
        ),
    )
    assert rows["temporal_pair_mask"].tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert torch.equal(
        rows["previous_token_hidden"],
        torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
        ),
    )


def test_high_level_transfer_summary_is_chance_normalized() -> None:
    train = load_train_module()
    # Import the probe helper without running its CLI.
    probe_path = Path(__file__).parents[1] / "src" / "run_linear_probes.py"
    spec = importlib.util.spec_from_file_location("run_linear_probes_for_tests", probe_path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    def row(full: float, ood: float, low: tuple[float, float, float]) -> dict:
        return {
            "full": {
                "accuracy": full,
                "worst_class_accuracy": full - 0.1,
                "class_accuracy_std": 0.05,
            },
            "ood": {
                "accuracy": ood,
                "worst_class_accuracy": ood - 0.1,
                "class_accuracy_std": 0.06,
            },
            "low_label": {
                str(budget): {"mean_accuracy": value}
                for budget, value in zip((16, 64, 256), low, strict=True)
            },
        }

    results = {
            "representations": {
                "token_sae_mean": row(0.80, 0.70, (0.50, 0.60, 0.70)),
                "temporal_sae_mean": row(0.805, 0.71, (0.51, 0.61, 0.71)),
                "mean_chunk_sae": row(0.81, 0.72, (0.52, 0.62, 0.72)),
            "cross_chunk_sae": row(0.82, 0.75, (0.55, 0.65, 0.75)),
        }
    }
    summary = probe.high_level_summary(
        results,
        "mean",
        [16, 64, 256],
    )
    assert summary["chance_accuracy"] == pytest.approx(0.125)
    assert (
        summary["methods"]["cross"]["high_level_transfer_score"]
        > summary["methods"]["mean"]["high_level_transfer_score"]
        > summary["methods"]["token"]["high_level_transfer_score"]
    )
    assert (
        summary["comparisons"]["cross_minus_mean"][
            "high_level_transfer_score"
        ]
        > 0
    )


def test_paired_bootstrap_comparison_uses_matched_examples() -> None:
    probe_path = Path(__file__).parents[1] / "src" / "run_linear_probes.py"
    spec = importlib.util.spec_from_file_location(
        "run_linear_probes_bootstrap_for_tests", probe_path
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    result = probe._bootstrap_mean_difference(
        torch.tensor([1.0, 0.0, -1.0, 1.0]).numpy(),
        samples=1000,
        seed=7,
    )
    assert result["point"] == pytest.approx(0.25)
    assert result["95ci"][0] <= result["point"] <= result["95ci"][1]
