from __future__ import annotations

import pytest
import numpy as np

from evals.rfve.evaluate_dictionary_health import _document_prevalence_metrics


def test_document_prevalence_groups_repeated_pairs_from_the_same_document():
    codes = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    result = _document_prevalence_metrics(
        codes[:2],
        codes[2:],
        np.asarray([b"doc-a", b"doc-b"], dtype=object),
        ubiquitous_support_fraction=0.50,
    )
    assert result["heldout_pairs"] == 2
    assert result["heldout_documents"] == 2
    assert result["ubiquitous_min_support_documents"] == 1
    assert result["ubiquitous_features"] == 3
    assert result["ubiquitous_fraction"] == pytest.approx(1.0)
