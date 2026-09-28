from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motivation.run_motivation_experiment import (
    _bootstrap_median,
    _center_token_id,
    _highlighted_surface,
    _mean_pairwise_jaccard,
    _trigger_category,
    analyze_token_triggers,
)


def test_center_token_and_surface_are_exact() -> None:
    example = {
        "token_ids": [10, 11, 12, 13, 14],
        "text": "left <<Declare>> right",
    }
    assert _center_token_id(example) == 12
    assert _highlighted_surface(example) == "declare"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (",", "punctuation_or_markup"),
        ("", "blank_or_whitespace"),
        ("0", "number"),
        ("the", "function_word"),
        (".jpg", "subword_code_or_mixed"),
        ("volcano", "content_word"),
    ],
)
def test_trigger_category(value: str, expected: str) -> None:
    assert _trigger_category(value) == expected


def test_pairwise_jaccard() -> None:
    value = _mean_pairwise_jaccard(
        [
            frozenset({"volcano", "eruption"}),
            frozenset({"volcano", "lava"}),
            frozenset({"city"}),
        ]
    )
    assert value == pytest.approx((1 / 3 + 0 + 0) / 3)


def test_bootstrap_median_is_deterministic() -> None:
    values = np.arange(1, 11, dtype=np.float64)
    assert _bootstrap_median(values, samples=1_000, seed=7) == _bootstrap_median(
        values,
        samples=1_000,
        seed=7,
    )


def test_trigger_analysis_uses_token_ids_not_surface_forms(tmp_path: Path) -> None:
    path = tmp_path / "prepared.jsonl"
    rows = []
    for feature_id in range(1_000):
        examples = []
        for index in range(10):
            examples.append(
                {
                    "kind": "top",
                    "is_active": True,
                    "token_ids": [1, 99, 2],
                    "text": f"context {index} <<X>> unrelated words",
                    "center_activation": 10.0 - index / 10,
                }
            )
        rows.append(
            {
                "method": "token",
                "status": "prepared",
                "feature_id": feature_id,
                "generation_examples": examples,
            }
        )
    # The production illustration IDs must exist in the fixed sample.
    for source, target in zip((0, 1, 2), (21417, 26062, 53225), strict=True):
        rows[source]["feature_id"] = target
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = analyze_token_triggers(path)
    assert result["same_token_all_10"] == 1_000
    assert result["dominant_token_at_least_8_of_10"] == 1_000
