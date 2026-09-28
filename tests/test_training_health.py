from __future__ import annotations

import pytest

from evals.rfve.plot_training_health import (
    _health_statuses,
    _task_rfve,
)


def test_standard_task_rfve_uses_method_reference():
    method = {"reference": {"reference_fve": 0.5}}
    assert _task_rfve(
        method,
        {"validation/fve": 0.4},
        split="validation",
    ) == pytest.approx(0.8)


def test_joint_task_rfve_weights_fve_before_dividing_by_reference():
    method = {
        "alpha": 0.25,
        "components": {
            "mean": {"reference_fve": 1.0},
            "cross": {"reference_fve": 0.5},
        },
    }
    row = {
        "train/joint_mean_fve": 0.8,
        "train/joint_cross_fve": 0.4,
    }
    assert _task_rfve(method, row, split="train") == pytest.approx(0.8)


def test_health_statuses_distinguish_exact_and_small_nonzero_counts():
    statuses, annotations = _health_statuses(
        effective_l0=128.0,
        k=128,
        zero_code_fraction=0.0,
        never_fired=0,
        recent_dead=12,
        dictionary_width=65_536,
    )
    assert statuses == [2, 2, 2, 1]
    assert annotations == ["128.0 / 128", "0.000%", "0", "12"]
