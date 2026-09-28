#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from chunk_saes.artifacts import (
    file_record,
    resolve_sae_artifact_set,
    write_artifact_manifest,
)
from chunk_saes.metrics import parse_fidelity_reference_fves
from chunk_saes.reference_fve import (
    audit_scalar_cross_reference,
    identity_reference,
    load_cross_reference_manifest,
    validate_cross_reference_against_sae,
)
from chunk_saes.utils import atomic_json_dump


RESULT_FORMAT = "chunk-saes-training-fidelity-v2"
METHODS = ("token", "temporal", "mean", "cross")
JOINT_MODE = "joint_chunk"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Publish Reference FVE (RFVE) endpoints and full validation "
            "training trajectories for the SAE methods, optionally including "
            "a separately trained Joint Chunk SAE."
        )
    )
    p.add_argument("--sae-root", required=True)
    p.add_argument(
        "--joint-sae-root",
        help=(
            "Optional checkpoint root containing a completed joint_chunk mode. "
            "Its Mean/Cross objectives are combined into one alpha-weighted "
            "Joint RFVE trajectory while retaining both component RFVEs."
        ),
    )
    p.add_argument(
        "--checkpoint-selection",
        choices=("best", "final"),
        default="best",
    )
    p.add_argument(
        "--attainable-reference-fves",
        default="token=1.0,temporal=1.0,mean=1.0",
        help=(
            "Legacy audit-only mode=value references. A Cross value is rejected "
            "unless --allow-audit-cross-scalar is explicitly set."
        ),
    )
    p.add_argument(
        "--cross-reference-manifest",
        help=(
            "Publication-grade chunk-saes-reference-fve-v1 manifest for the "
            "direction-blind dense Cross-Chunk predictor."
        ),
    )
    p.add_argument("--allow-audit-cross-scalar", action="store_true")
    p.add_argument("--output", required=True)
    return p


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validation_trajectory(path: Path, reference_fve: float) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}"
                ) from error
            if (
                row.get("split") != "validation"
                or row.get("step") is None
                or row.get("validation/fve") is None
            ):
                continue
            step = int(row["step"])
            fve = float(row["validation/fve"])
            if not math.isfinite(fve):
                continue
            point: dict[str, Any] = {
                "step": step,
                "samples_seen": int(row.get("progress/samples_seen", step * 32_000)),
                "rfve": fve / reference_fve,
                "validation_samples": int(row.get("validation/samples", 0)),
            }
            if row.get("validation/a_to_b/fve") is not None:
                point["a_to_b_rfve"] = (
                    float(row["validation/a_to_b/fve"]) / reference_fve
                )
            if row.get("validation/b_to_a/fve") is not None:
                point["b_to_a_rfve"] = (
                    float(row["validation/b_to_a/fve"]) / reference_fve
                )
            by_step[step] = point
    trajectory = [by_step[step] for step in sorted(by_step)]
    if not trajectory:
        raise ValueError(f"{path} contains no periodic validation FVE rows")
    return trajectory


def _endpoint_metrics(
    *,
    sae_root: Path,
    mode: str,
    selection: str,
    complete: dict[str, Any],
) -> tuple[dict[str, Any], str, int]:
    if selection == "final":
        metrics = complete.get("final_full_validation_metrics") or {}
        return metrics, "full held-out validation cache", int(complete["steps"])
    best_step = int(complete["best_step"])
    selected = None
    with (sae_root / mode / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("split") == "validation"
                and int(row.get("step", -1)) == best_step
                and row.get("validation/fve") is not None
            ):
                selected = row
    if selected is None:
        raise ValueError(
            f"{mode} lacks periodic validation metrics for best step {best_step}"
        )
    return selected, "fixed periodic held-out validation subset", best_step


def _joint_fidelity_values(
    *,
    mean_fve: float,
    cross_fve: float,
    alpha: float,
    mean_reference_fve: float,
    cross_reference_fve: float,
) -> dict[str, float]:
    """Return component and task-weighted Joint RFVE values.

    Joint training minimizes ``(NMSE_mean + alpha * NMSE_cross)/(1+alpha)``.
    Therefore the corresponding explained-variance numerator is the same
    weighted average of the two FVEs. Its same-information reference uses the
    identity oracle for Mean and the verified dense predictor for Cross.
    """

    joint_fve = (mean_fve + alpha * cross_fve) / (1.0 + alpha)
    joint_reference_fve = (
        mean_reference_fve + alpha * cross_reference_fve
    ) / (1.0 + alpha)
    return {
        "mean_fve": mean_fve,
        "mean_rfve": mean_fve / mean_reference_fve,
        "cross_fve": cross_fve,
        "cross_rfve": cross_fve / cross_reference_fve,
        "joint_fve": joint_fve,
        "joint_reference_fve": joint_reference_fve,
        "rfve": joint_fve / joint_reference_fve,
    }


def _joint_validation_trajectory(
    path: Path,
    *,
    alpha: float,
    mean_reference_fve: float,
    cross_reference_fve: float,
    cross_direction_references: dict[str, float],
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}"
                ) from error
            mean_fve = row.get("validation/joint_mean_fve")
            cross_fve = row.get("validation/joint_cross_fve")
            if (
                row.get("split") != "validation"
                or row.get("step") is None
                or mean_fve is None
                or cross_fve is None
            ):
                continue
            mean_fve = float(mean_fve)
            cross_fve = float(cross_fve)
            if not math.isfinite(mean_fve) or not math.isfinite(cross_fve):
                continue
            step = int(row["step"])
            point: dict[str, Any] = {
                "step": step,
                "samples_seen": int(
                    row.get("progress/samples_seen", step * 32_000)
                ),
                "validation_samples": int(
                    row.get("validation/joint_mean/samples", 0)
                ),
                "k_prefix": float(
                    row.get("validation/joint_k_prefix", float("nan"))
                ),
                **_joint_fidelity_values(
                    mean_fve=mean_fve,
                    cross_fve=cross_fve,
                    alpha=alpha,
                    mean_reference_fve=mean_reference_fve,
                    cross_reference_fve=cross_reference_fve,
                ),
            }
            for direction in ("a_to_b", "b_to_a"):
                value = row.get(f"validation/joint_cross/{direction}/fve")
                if value is not None:
                    point[f"{direction}_cross_rfve"] = (
                        float(value) / cross_direction_references[direction]
                    )
            by_step[step] = point
    trajectory = [by_step[step] for step in sorted(by_step)]
    if not trajectory:
        raise ValueError(f"{path} contains no periodic Joint validation rows")
    return trajectory


def _joint_endpoint_metrics(
    *,
    joint_root: Path,
    selection: str,
    complete: dict[str, Any],
) -> tuple[dict[str, Any], str, int]:
    if selection == "final":
        metrics = complete.get("final_full_validation_metrics") or {}
        return metrics, "full held-out validation cache", int(complete["steps"])
    best_step = int(complete["best_step"])
    selected = None
    with (joint_root / JOINT_MODE / "metrics.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("split") == "validation"
                and int(row.get("step", -1)) == best_step
                and row.get("validation/joint_mean_fve") is not None
                and row.get("validation/joint_cross_fve") is not None
            ):
                selected = row
    if selected is None:
        raise ValueError(
            "joint_chunk lacks periodic validation metrics for "
            f"best step {best_step}"
        )
    return selected, "fixed periodic held-out validation subset", best_step


def _validate_joint_common(
    *,
    base_common: dict[str, Any],
    joint_common: dict[str, Any],
) -> None:
    for key in (
        "project",
        "training_format",
        "model",
        "layer",
        "activation_dim",
        "dict_size",
        "k",
        "global_batch_size",
        "steps",
        "exact_coverage_required",
        "seed",
        "train_cache_digest",
        "validation_cache_digest",
    ):
        if joint_common.get(key) != base_common.get(key):
            raise ValueError(
                f"Joint/base SAE mismatch for {key}: "
                f"{joint_common.get(key)!r} != {base_common.get(key)!r}"
            )


def _joint_method_record(
    *,
    joint_root: Path,
    selection: str,
    base_common: dict[str, Any],
    cross_reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    joint_set = resolve_sae_artifact_set(
        joint_root,
        selection=selection,
        modes=(JOINT_MODE,),
    )
    _validate_joint_common(
        base_common=base_common,
        joint_common=joint_set["common"],
    )
    complete = _read_json(joint_root / JOINT_MODE / "complete.json")
    config = _read_json(joint_root / JOINT_MODE / "config.json")
    if config.get("joint_chunk_layout") != "nested_prefix":
        raise ValueError(
            f"{joint_root} is not a nested-prefix Joint Chunk SAE"
        )
    cross_prefix = int(config.get("joint_cross_prefix", 0))
    if not 0 < cross_prefix < int(config["dict_size"]):
        raise ValueError(
            f"invalid Joint Cross prefix {cross_prefix} in {joint_root}"
        )
    alpha = float(config.get("joint_chunk_alpha", 0.0))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"invalid Joint alpha {alpha} in {joint_root}")

    mean_reference_fve = 1.0
    cross_reference_fve = float(cross_reference["reference_fve"])
    direction_references = cross_reference.get("directions") or {
        "a_to_b": cross_reference_fve,
        "b_to_a": cross_reference_fve,
    }
    metrics, evaluation_split, selected_step = _joint_endpoint_metrics(
        joint_root=joint_root,
        selection=selection,
        complete=complete,
    )
    mean_fve = float(metrics["validation/joint_mean_fve"])
    cross_fve = float(metrics["validation/joint_cross_fve"])
    fidelity = _joint_fidelity_values(
        mean_fve=mean_fve,
        cross_fve=cross_fve,
        alpha=alpha,
        mean_reference_fve=mean_reference_fve,
        cross_reference_fve=cross_reference_fve,
    )
    trajectory = _joint_validation_trajectory(
        joint_root / JOINT_MODE / "metrics.jsonl",
        alpha=alpha,
        mean_reference_fve=mean_reference_fve,
        cross_reference_fve=cross_reference_fve,
        cross_direction_references={
            direction: float(direction_references[direction])
            for direction in ("a_to_b", "b_to_a")
        },
    )
    reference = {
        "mode": "joint",
        "kind": "alpha_weighted_mean_cross_reference",
        "reference_fve": fidelity["joint_reference_fve"],
        "provenance_status": cross_reference["provenance_status"],
        "alpha": alpha,
        "composition": (
            "(mean_reference_fve + alpha * cross_reference_fve) / "
            "(1 + alpha)"
        ),
        "mean": identity_reference("mean"),
        "cross": cross_reference,
    }
    directions = {
        direction: {
            "fve": float(
                metrics[f"validation/joint_cross/{direction}/fve"]
            ),
            "rfve": (
                float(metrics[f"validation/joint_cross/{direction}/fve"])
                / float(direction_references[direction])
            ),
            "reference_fve": float(direction_references[direction]),
        }
        for direction in ("a_to_b", "b_to_a")
    }
    row = {
        **fidelity,
        "remaining_reference_regret": 1.0 - fidelity["rfve"],
        "reference": reference,
        "evaluation_split": evaluation_split,
        "evaluation_samples": int(
            metrics["validation/joint_mean/samples"]
        ),
        "effective_l0": float(
            metrics["validation/joint_mean_effective_l0"]
        ),
        "k_prefix": float(metrics["validation/joint_k_prefix"]),
        "cross_prefix": cross_prefix,
        "alpha": alpha,
        "best_step": int(complete["best_step"]),
        "selected_step": selected_step,
        "selection_metric": str(
            complete.get("best_metric", "joint_cross_nmse")
        ),
        "alive_features": int(complete["alive_features"]),
        "dead_features": int(complete.get("dead_features", 0)),
        "coverage_fraction": float(complete["coverage_fraction"]),
        "components": {
            "mean": {
                "fve": fidelity["mean_fve"],
                "rfve": fidelity["mean_rfve"],
                "reference_fve": mean_reference_fve,
            },
            "cross": {
                "fve": fidelity["cross_fve"],
                "rfve": fidelity["cross_rfve"],
                "reference_fve": cross_reference_fve,
                "directions": directions,
            },
        },
        "trajectory": trajectory,
    }
    return row, reference, joint_set


def _reference_records(
    *,
    args: argparse.Namespace,
    sae_set: dict[str, Any],
    legacy_references: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    references = {
        mode: identity_reference(mode)
        for mode in sae_set["modes"]
        if mode in {"token", "temporal", "mean"}
    }
    cross_manifest = None
    if "cross" not in sae_set["modes"]:
        return references, cross_manifest
    if args.cross_reference_manifest:
        cross_manifest, cross_results = load_cross_reference_manifest(
            args.cross_reference_manifest,
            verify_files=True,
        )
        validate_cross_reference_against_sae(
            manifest=cross_manifest,
            results=cross_results,
            sae_common=sae_set["common"],
        )
        test = cross_results["test"]
        validation_monitor = cross_results.get("validation_monitor")
        if not isinstance(validation_monitor, dict):
            raise ValueError(
                "Cross RFVE artifact lacks validation_monitor metrics required "
                "for the training trajectory"
            )
        references["cross"] = {
            "mode": "cross",
            "kind": "train_fitted_dense_predictor",
            "reference_fve": float(validation_monitor["pooled_fve"]),
            "final_test_reference_fve": float(test["pooled_fve"]),
            "provenance_status": "publication",
            "artifact_digest": cross_manifest["artifact_digest"],
            "manifest_path": str(
                Path(args.cross_reference_manifest).expanduser().resolve()
            ),
            "directions": {
                "a_to_b": float(validation_monitor["a_to_b_fve"]),
                "b_to_a": float(validation_monitor["b_to_a_fve"]),
            },
            "final_test_directions": {
                "a_to_b": float(test["a_to_b_fve"]),
                "b_to_a": float(test["b_to_a_fve"]),
            },
            "selected_model": cross_results["selected_model"],
            "capacity_sweep": cross_results["capacity_sweep"],
            "test_cache_digest": cross_manifest["identity"]["test_cache_digest"],
        }
        return references, cross_manifest
    value = legacy_references.get("cross")
    if value is None:
        raise ValueError(
            "Cross-Chunk RFVE requires --cross-reference-manifest. "
            "Use --allow-audit-cross-scalar only to reproduce a legacy audit."
        )
    if not args.allow_audit_cross_scalar:
        raise ValueError(
            "A scalar Cross reference has no auditable provenance. Provide "
            "--cross-reference-manifest or explicitly opt into audit-only output "
            "with --allow-audit-cross-scalar."
        )
    references["cross"] = audit_scalar_cross_reference(float(value))
    return references, cross_manifest


def main() -> None:
    args = parser().parse_args()
    sae_root = Path(args.sae_root).resolve()
    sae_set = resolve_sae_artifact_set(
        sae_root,
        selection=args.checkpoint_selection,
    )
    legacy_references = parse_fidelity_reference_fves(
        args.attainable_reference_fves
    )
    references, cross_manifest = _reference_records(
        args=args,
        sae_set=sae_set,
        legacy_references=legacy_references,
    )
    modes = tuple(sae_set["modes"])
    methods = {}
    for mode in modes:
        complete = _read_json(sae_root / mode / "complete.json")
        metrics, evaluation_split, selected_step = _endpoint_metrics(
            sae_root=sae_root,
            mode=mode,
            selection=args.checkpoint_selection,
            complete=complete,
        )
        sparse_fve = metrics.get("validation/fve")
        reference = references.get(mode)
        if sparse_fve is None:
            raise ValueError(f"{mode} lacks full-validation FVE")
        if reference is None:
            raise ValueError(
                f"RFVE reference is missing for mode={mode}"
            )
        reference_fve = float(reference["reference_fve"])
        trajectory = _validation_trajectory(
            sae_root / mode / "metrics.jsonl",
            reference_fve,
        )
        row = {
            "rfve": float(sparse_fve) / reference_fve,
            "remaining_reference_regret": (
                1.0 - float(sparse_fve) / reference_fve
            ),
            "reference": reference,
            "evaluation_split": evaluation_split,
            "evaluation_samples": int(metrics["validation/samples"]),
            "effective_l0": float(metrics["validation/effective_l0"]),
            "best_step": int(complete["best_step"]),
            "selected_step": selected_step,
            "alive_features": int(complete["alive_features"]),
            "coverage_fraction": float(complete["coverage_fraction"]),
            "trajectory": trajectory,
        }
        if mode == "cross":
            reference_directions = reference.get("directions") or {
                "a_to_b": reference_fve,
                "b_to_a": reference_fve,
            }
            row["directions"] = {
                direction: {
                    "rfve": (
                        float(metrics[f"validation/{direction}/fve"])
                        / float(reference_directions[direction])
                    ),
                    "reference_fve": float(reference_directions[direction]),
                }
                for direction in ("a_to_b", "b_to_a")
            }
        methods[mode] = row

    joint_set = None
    if args.joint_sae_root:
        joint_root = Path(args.joint_sae_root).expanduser().resolve()
        cross_reference = references.get("cross")
        if cross_reference is None:
            raise ValueError(
            "Joint RFVE requires the same verified Cross reference used "
                "by Cross-Chunk SAE"
            )
        joint_row, joint_reference, joint_set = _joint_method_record(
            joint_root=joint_root,
            selection=args.checkpoint_selection,
            base_common=sae_set["common"],
            cross_reference=cross_reference,
        )
        methods["joint"] = joint_row
        references["joint"] = joint_reference

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "sae_set_digest": sae_set["artifact_digest"],
        "checkpoint_selection": args.checkpoint_selection,
        "reference_fves": {
            mode: float(reference["reference_fve"])
            for mode, reference in references.items()
        },
        "cross_reference_artifact_digest": (
            cross_manifest["artifact_digest"]
            if cross_manifest is not None
            else None
        ),
    }
    if joint_set is not None:
        identity["joint_sae_set_digest"] = joint_set["artifact_digest"]
    payload = {
        "format": RESULT_FORMAT,
        "complete": True,
        "identity": identity,
        "definition": (
            "RFVE = sparse SAE FVE / same-information task reference FVE"
        ),
        "metric_name": "Reference FVE (RFVE)",
        "interpretation": (
            "RFVE is the fraction of the variance explained by a "
            "same-information, task-specific reference predictor that is "
            "captured by the K-sparse SAE. It is not raw reconstruction FVE."
        ),
        "trajectory_protocol": (
            "All curves use the same fixed periodic held-out validation subset. "
            "The Cross dense reference is selected on full validation and "
            "evaluated once on this monitor subset to define the curve denominator. "
            "When present, Joint-Chunk is one alpha-weighted composite curve; "
            "its Mean and Cross component RFVEs are retained in every point."
        ),
        "joint_rfve_definition": (
            "For Joint-Chunk, the plotted scalar is the ratio of the "
            "alpha-weighted Mean/Cross explained variance to the corresponding "
            "alpha-weighted same-information reference. Component RFVEs are "
            "also retained in methods.joint.components."
            if joint_set is not None
            else None
        ),
        "publication_ready": all(
            reference["provenance_status"] in {"exact", "publication"}
            for reference in references.values()
        ),
        "references": references,
        "methods": methods,
    }
    atomic_json_dump(payload, output)
    write_artifact_manifest(
        {
            "format": RESULT_FORMAT,
            "complete": True,
            "identity": identity,
            "files": {
                "results": file_record(output, relative_to=output.parent),
                **(
                    {
                        "cross_reference_manifest": file_record(
                            Path(args.cross_reference_manifest).resolve(),
                            relative_to=output.parent,
                        )
                    }
                    if args.cross_reference_manifest
                    else {}
                ),
                **(
                    {
                        "joint_complete": file_record(
                            Path(args.joint_sae_root).resolve()
                            / JOINT_MODE
                            / "complete.json",
                            relative_to=output.parent,
                        ),
                        "joint_config": file_record(
                            Path(args.joint_sae_root).resolve()
                            / JOINT_MODE
                            / "config.json",
                            relative_to=output.parent,
                        ),
                        "joint_metrics": file_record(
                            Path(args.joint_sae_root).resolve()
                            / JOINT_MODE
                            / "metrics.jsonl",
                            relative_to=output.parent,
                        ),
                    }
                    if args.joint_sae_root
                    else {}
                ),
            },
        },
        output.with_name("training_fidelity_manifest.json"),
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
