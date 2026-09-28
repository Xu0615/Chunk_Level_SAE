from __future__ import annotations

import json

import pytest

from evals.rfve.analyze_training_fidelity import (
    _joint_fidelity_values,
    _joint_validation_trajectory,
)


def test_joint_fidelity_is_ratio_of_weighted_explained_variances():
    values = _joint_fidelity_values(
        mean_fve=0.8,
        cross_fve=0.4,
        alpha=0.25,
        mean_reference_fve=1.0,
        cross_reference_fve=0.5,
    )
    assert values["mean_rfve"] == pytest.approx(0.8)
    assert values["cross_rfve"] == pytest.approx(0.8)
    assert values["joint_fve"] == pytest.approx(0.72)
    assert values["joint_reference_fve"] == pytest.approx(0.9)
    assert values["rfve"] == pytest.approx(0.8)


def test_joint_validation_trajectory_retains_component_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"split": "train", "step": 10},
        {
            "split": "validation",
            "step": 250,
            "validation/joint_mean_fve": 0.8,
            "validation/joint_cross_fve": 0.4,
            "validation/joint_cross/a_to_b/fve": 0.3,
            "validation/joint_cross/b_to_a/fve": 0.45,
            "validation/joint_mean/samples": 1024,
            "validation/joint_k_prefix": 55.0,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    trajectory = _joint_validation_trajectory(
        path,
        alpha=0.25,
        mean_reference_fve=1.0,
        cross_reference_fve=0.5,
        cross_direction_references={"a_to_b": 0.5, "b_to_a": 0.5},
    )
    assert len(trajectory) == 1
    point = trajectory[0]
    assert point["step"] == 250
    assert point["samples_seen"] == 8_000_000
    assert point["validation_samples"] == 1024
    assert point["k_prefix"] == pytest.approx(55.0)
    assert point["mean_rfve"] == pytest.approx(0.8)
    assert point["cross_rfve"] == pytest.approx(0.8)
    assert point["rfve"] == pytest.approx(0.8)
    assert point["a_to_b_cross_rfve"] == pytest.approx(0.6)
    assert point["b_to_a_cross_rfve"] == pytest.approx(0.9)
