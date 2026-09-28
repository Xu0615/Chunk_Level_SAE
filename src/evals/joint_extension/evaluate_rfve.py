#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunk_saes.artifacts import resolve_sae_artifact_set
from evals.joint_extension.common import (
    JointSpec,
    add_joint_root_arguments,
    joint_specs_from_args,
    merge_sidecar,
)
from evals.rfve import analyze_training_fidelity as rfve


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate an alpha sweep of nested Joint-Chunk SAEs with RFVE."
    )
    p.add_argument("--eval-root", required=True)
    p.add_argument("--baseline-sae-root", required=True)
    add_joint_root_arguments(p)
    return p


def _specs(args: argparse.Namespace) -> list[JointSpec]:
    return joint_specs_from_args(args)


def main() -> None:
    args = parser().parse_args()
    eval_root = Path(args.eval_root).resolve()
    base = json.loads(
        (eval_root / "rfve/training_fidelity.json").read_text(encoding="utf-8")
    )
    cross_reference = base["references"]["cross"]
    baseline_set = resolve_sae_artifact_set(
        args.baseline_sae_root,
        selection="best",
    )
    specs = _specs(args)
    methods = {}
    for spec in specs:
        # Historical checkpoints can record equivalent mount aliases.
        # All other provenance fields (cache digests, dimensions, layer,
        # training coverage, seed) remain checked by the canonical helper.
        joint_set = resolve_sae_artifact_set(
            spec.root,
            selection="best",
            modes=("joint_chunk",),
        )
        compatible_base_common = dict(baseline_set["common"])
        compatible_base_common["model"] = joint_set["common"]["model"]
        row, reference, _joint_set = rfve._joint_method_record(
            joint_root=spec.root,
            selection="best",
            base_common=compatible_base_common,
            cross_reference=cross_reference,
        )
        methods[spec.key] = {
            "label": spec.label,
            **row,
            "reference": reference,
        }
    output = eval_root / "rfve/joint_extension.json"
    merge_sidecar(
        output,
        task="rfve",
        methods=methods,
        specs=specs,
        protocol={
            "definition": base["definition"],
            "trajectory_protocol": base["trajectory_protocol"],
            "same_cross_reference": True,
            "cross_reference_artifact_digest": cross_reference[
                "artifact_digest"
            ],
            "checkpoint_selection": "best",
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
