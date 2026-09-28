#!/usr/bin/env python
"""Render the five-method sequence-activation figure for the eval summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.feature_dynamics.plot_cross_feature_semantic_manifold import (
    WHITE,
    build_sequence_activation_figure,
    sequence_feature_annotations,
)


SUMMARY_METHODS = (
    "token",
    "temporal",
    "mean",
    "joint_alpha0p25",
    "cross",
)

DISPLAY_ANNOTATION_REPLACEMENTS = {
    "Measurement and time units after numbers": (
        "Numbers with measurement/time units"
    ),
    "Common English words linking phrases together": (
        "Common English phrase-linking words"
    ),
    "Punctuation formatting mathematical expressions": (
        "Mathematical expression punctuation"
    ),
    "Plural nouns: questions topics policies problems": (
        "Plural topic/policy/problem nouns"
    ),
    "Periods ending sentences in ordinary prose": (
        "Sentence-ending periods in prose"
    ),
    "Measurement units and times after numbers": (
        "Units and times after numbers"
    ),
    "Common short English words in prose": (
        "Common short English prose words"
    ),
    "Punctuation after LaTeX formulas and notation": (
        "Punctuation after LaTeX formulas"
    ),
    "Terms for mathematical constructions bounds expressions": (
        "Math construction and bound terms"
    ),
    "English function words linking phrases together": (
        "English phrase-linking function words"
    ),
    "Mixed technical and narrative prose fragments": (
        "Mixed technical and narrative prose"
    ),
    "Advanced mathematics across algebra analysis physics": (
        "Advanced algebra and physics"
    ),
    "Technical prose explaining methods problems products": (
        "Technical prose on methods/products"
    ),
    "Informal posts mixing reflection and opinion": (
        "Informal reflective opinion posts"
    ),
    "First-person stories sharing emotions and anecdotes": (
        "First-person emotion and anecdotes"
    ),
    "Shared programming and news text patterns": (
        "Shared programming/news patterns"
    ),
    "Calibrated geometry and smooth manifold mathematics": (
        "Calibrated smooth-manifold geometry"
    ),
    "News reports on viral media reactions": (
        "News reports on viral-media reactions"
    ),
    "Shared news and medical text patterns": (
        "Shared news/medical text patterns"
    ),
    "Programming code with CSV leading zeros": (
        "CSV programming with leading zeros"
    ),
    "Advanced mathematics on definitions conjectures geometry": (
        "Math definitions/conjectures/geometry"
    ),
    "Political news covering international current events": (
        "International politics/current events"
    ),
    "C code build scripts configuration macros": (
        "C build/config scripts and macros"
    ),
    "Indian affairs spanning politics business culture": (
        "Indian politics, business, and culture"
    ),
    "Personal blogs sharing anecdotes and opinions": (
        "Personal blog anecdotes and opinions"
    ),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eval-root", required=True)
    result.add_argument(
        "--output-name",
        default="13_sequence_activation_traces",
    )
    result.add_argument("--dpi", type=int, default=300)
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _save_atomic(fig: plt.Figure, path: Path, *, dpi: int | None = None) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    kwargs: dict[str, object] = {
        "facecolor": WHITE,
        "format": path.suffix.removeprefix("."),
    }
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(temporary, **kwargs)
    temporary.replace(path)


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    feature_dir = eval_root / "feature_dynamics"
    metadata = _read_json(feature_dir / "sequence_activation_traces.json")
    with np.load(
        feature_dir / "sequence_activation_traces.npz",
        allow_pickle=True,
    ) as handle:
        arrays = {key: handle[key] for key in handle.files}

    annotations, annotation_path = sequence_feature_annotations(
        eval_root,
        metadata,
        methods=SUMMARY_METHODS,
    )
    batchtopk_third_feature = int(arrays["token_feature_ids"][2])
    annotations["token"][batchtopk_third_feature] = (
        "Punctuation formatting mathematical expressions"
    )
    for method_annotations in annotations.values():
        for feature_id, annotation in list(method_annotations.items()):
            method_annotations[feature_id] = (
                DISPLAY_ANNOTATION_REPLACEMENTS.get(annotation, annotation)
            )
    fig = build_sequence_activation_figure(
        arrays=arrays,
        metadata=metadata,
        feature_annotations=annotations,
        methods=SUMMARY_METHODS,
        figsize=(20.0, 12.5),
        left=0.075,
        right=0.975,
        bottom=0.12,
        top=0.865,
        panel_title="Own top-5 features on identical text",
        x_axis_label="Texts from different domains",
        line_width_scale=1.50,
        feature_annotation_fontsize=18.9,
        sequence_width_ratios=(0.58, 0.42),
        panel_title_fontsize=22.0,
        panel_title_y=1.38,
        x_axis_label_fontsize=20.0,
        y_axis_label_fontsize=19.0,
        x_tick_fontsize=17.0,
        feature_panel_title_fontsize=18.5,
        rank_legend_fontsize=15.0,
        rank_legend_x=0.50,
        rank_legend_y=1.18,
        rank_legend_loc="center",
        method_label_y=0.075,
        method_label_vertical_alignment="bottom",
    )

    output_base = (
        eval_root / "evaluation_summary" / "figures" / args.output_name
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    _save_atomic(fig, png_path, dpi=args.dpi)
    _save_atomic(fig, pdf_path)
    plt.close(fig)

    print(
        json.dumps(
            {
                "methods": list(SUMMARY_METHODS),
                "annotation_source": str(annotation_path),
                "files": [str(png_path), str(pdf_path)],
                "aspect_ratio": "1.6:1",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
