from __future__ import annotations

import pytest
import torch

from evals.rfve.evaluate_reconstruction_cosine import CosineAccumulator


def test_cosine_accumulator_converts_scaled_reconstruction_back_to_raw():
    accumulator = CosineAccumulator(torch.device("cpu"))
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed_scaled = 3.0 * target
    accumulator.update(
        reconstructed_scaled,
        target,
        activation_scale=3.0,
    )
    result = accumulator.result()
    assert result["samples"] == 2
    assert result["mean_cosine_similarity"] == pytest.approx(1.0)
    assert result["std_cosine_similarity"] == pytest.approx(0.0)
