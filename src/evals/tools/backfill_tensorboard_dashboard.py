#!/usr/bin/env python
"""Build and optionally follow the complete SAE TensorBoard dashboard."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from chunk_saes.metrics import (
    DASHBOARD_SCALAR_TAGS,
    TensorBoardLogger,
    parse_fidelity_reference_fves,
)


MODES = ("token", "temporal", "mean", "cross")
SPLIT_ORDER = {"train": 0, "validation": 1, "validation_full": 2}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill/follow complete SAE TensorBoard scalars."
    )
    parser.add_argument("--checkpoint-run-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--attainable-reference-fves",
        "--fidelity-reference-fves",
        dest="fidelity_reference_fves",
        default="token=1.0,temporal=1.0,mean=1.0",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["split"]), int(row["step"])


def _row_fingerprint(row: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_canonical_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[_row_key(row)] = row
    return rows


def _sorted_rows(rows: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows.values(),
        key=lambda row: (
            int(row["step"]),
            SPLIT_ORDER.get(str(row["split"]), 99),
        ),
    )


def _dashboard_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("split") != "validation_full":
        return row
    normalized = {
        key: value for key, value in row.items() if not key.startswith("validation/")
    }
    normalized.update(
        {
            key.replace("validation/", "validation_full/", 1): value
            for key, value in row.items()
            if key.startswith("validation/")
        }
    )
    return normalized


def _writer(
    output_dir: Path,
    *,
    mode: str,
    references: dict[str, float],
    gradient_clip: float,
) -> TensorBoardLogger:
    return TensorBoardLogger(
        output_dir,
        rank=0,
        flush_secs=2,
        max_queue=100,
        scalar_allowlist=DASHBOARD_SCALAR_TAGS,
        mode=mode,
        fidelity_reference_fve=references.get(mode),
        gradient_clip=gradient_clip,
    )


def _write_rows(
    writer: TensorBoardLogger,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        writer.add_scalars(_dashboard_row(row), int(row["step"]))


def _output_dir(tensorboard_dir: Path, run_id: str, mode: str) -> Path:
    # Keep derived events in their own run so resume purges cannot affect the
    # original training scalars written by the trainer.
    return tensorboard_dir / run_id / "dashboard" / mode


def _rebuild_mode(
    *,
    mode: str,
    metrics_path: Path,
    output_dir: Path,
    references: dict[str, float],
    gradient_clip: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = _read_canonical_rows(metrics_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    writer = _writer(
        output_dir,
        mode=mode,
        references=references,
        gradient_clip=gradient_clip,
    )
    try:
        _write_rows(writer, _sorted_rows(rows))
        writer.flush()
    finally:
        writer.close()
    return rows


def _follow_mode(
    *,
    mode: str,
    metrics_path: Path,
    output_dir: Path,
    references: dict[str, float],
    gradient_clip: float,
    poll_seconds: float,
) -> None:
    rows = _read_canonical_rows(metrics_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    fingerprints = {_key: _row_fingerprint(row) for _key, row in rows.items()}
    writer = _writer(
        output_dir,
        mode=mode,
        references=references,
        gradient_clip=gradient_clip,
    )
    # Replaying into the new writer keeps its train/validation state aligned
    # with the historical rows before incremental writes begin.
    try:
        _write_rows(writer, _sorted_rows(rows))
        writer.flush()
        while True:
            time.sleep(max(0.25, poll_seconds))
            current = _read_canonical_rows(metrics_path)
            current_fingerprints = {
                key: _row_fingerprint(row) for key, row in current.items()
            }
            existing_changed = any(
                current_fingerprints[key] != fingerprints[key]
                for key in set(current_fingerprints) & set(fingerprints)
            )
            removed = set(fingerprints) - set(current_fingerprints)
            if existing_changed or removed:
                writer.close()
                rows = _read_canonical_rows(metrics_path)
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                fingerprints = {
                    key: _row_fingerprint(row) for key, row in rows.items()
                }
                writer = _writer(
                    output_dir,
                    mode=mode,
                    references=references,
                    gradient_clip=gradient_clip,
                )
                _write_rows(writer, _sorted_rows(rows))
                writer.flush()
                continue
            new_keys = set(current) - set(rows)
            for key in sorted(
                new_keys,
                key=lambda item: (item[1], SPLIT_ORDER.get(item[0], 99)),
            ):
                row = current[key]
                writer.add_scalars(_dashboard_row(row), int(row["step"]))
            if new_keys:
                writer.flush()
            rows = current
            fingerprints = current_fingerprints
    finally:
        writer.close()


def main() -> None:
    args = _parser().parse_args()
    references = parse_fidelity_reference_fves(args.fidelity_reference_fves)
    checkpoint_run_dir = args.checkpoint_run_dir.resolve()
    run_id = args.run_id or checkpoint_run_dir.name
    if not checkpoint_run_dir.is_dir():
        raise FileNotFoundError(checkpoint_run_dir)
    if args.poll_seconds <= 0.0:
        raise ValueError("poll-seconds must be positive")
    if args.overwrite:
        dashboard_root = args.tensorboard_dir / run_id / "dashboard"
        if dashboard_root.exists():
            shutil.rmtree(dashboard_root)

    if args.follow:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODES)) as pool:
            futures = [
                pool.submit(
                    _follow_mode,
                    mode=mode,
                    metrics_path=checkpoint_run_dir / mode / "metrics.jsonl",
                    output_dir=_output_dir(args.tensorboard_dir, run_id, mode),
                    references=references,
                    gradient_clip=args.gradient_clip,
                    poll_seconds=args.poll_seconds,
                )
                for mode in MODES
            ]
            for future in futures:
                future.result()
    else:
        for mode in MODES:
            metrics_path = checkpoint_run_dir / mode / "metrics.jsonl"
            output_dir = _output_dir(args.tensorboard_dir, run_id, mode)
            _rebuild_mode(
                mode=mode,
                metrics_path=metrics_path,
                output_dir=output_dir,
                references=references,
                gradient_clip=args.gradient_clip,
            )


if __name__ == "__main__":
    main()
