from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ReconstructionSummary:
    """Globally aggregatable reconstruction statistics.

    ``scaled_objective`` is the exact objective optimized by the trainer:
    squared error summed over the activation dimension and averaged over
    samples. ``raw_mse`` and ``mean_predictor_mse`` are ordinary element-wise
    MSE values in the unscaled activation space.
    """

    samples: int
    elements: int
    scaled_sse: float
    raw_sse: float
    mean_predictor_sse: float
    active_entries: float
    encoded_samples: int
    zero_code_samples: int

    def as_dict(self) -> dict[str, float | int | bool]:
        scaled_objective = (
            self.scaled_sse / self.samples if self.samples else float("nan")
        )
        scaled_mse = self.scaled_sse / self.elements if self.elements else float("nan")
        raw_mse = self.raw_sse / self.elements if self.elements else float("nan")
        baseline_mse = (
            self.mean_predictor_sse / self.elements if self.elements else float("nan")
        )
        baseline_degenerate = not math.isfinite(self.mean_predictor_sse) or (
            self.mean_predictor_sse <= 0.0
        )
        if baseline_degenerate:
            normalized_mse = float("nan")
            fve = float("nan")
        else:
            normalized_mse = self.raw_sse / self.mean_predictor_sse
            fve = 1.0 - normalized_mse
        effective_l0 = (
            self.active_entries / self.encoded_samples
            if self.encoded_samples
            else float("nan")
        )
        zero_code_fraction = (
            self.zero_code_samples / self.encoded_samples
            if self.encoded_samples
            else float("nan")
        )
        return {
            "samples": self.samples,
            "elements": self.elements,
            "scaled_sse": self.scaled_sse,
            "scaled_objective": scaled_objective,
            "scaled_mse": scaled_mse,
            "raw_sse": self.raw_sse,
            "raw_mse": raw_mse,
            "mean_predictor_sse": self.mean_predictor_sse,
            "mean_predictor_mse": baseline_mse,
            "normalized_mse": normalized_mse,
            "nmse": normalized_mse,
            "fve": fve,
            "improvement_over_mean_predictor": fve,
            "baseline_degenerate": baseline_degenerate,
            "effective_l0": effective_l0,
            "zero_code_fraction": zero_code_fraction,
        }


class ReconstructionMetricAccumulator:
    """Accumulate sufficient statistics without averaging per-rank ratios."""

    _SCALED_SSE = 0
    _RAW_SSE = 1
    _BASELINE_SSE = 2
    _SAMPLES = 3
    _ELEMENTS = 4
    _ACTIVE_ENTRIES = 5
    _ENCODED_SAMPLES = 6
    _ZERO_CODE_SAMPLES = 7
    _SIZE = 8

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.totals = torch.zeros(self._SIZE, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(
        self,
        reconstructed_scaled: torch.Tensor,
        target_raw: torch.Tensor,
        *,
        activation_scale: float | torch.Tensor,
        mean_predictor_raw: torch.Tensor,
        features: torch.Tensor | None = None,
        active_counts: torch.Tensor | None = None,
    ) -> None:
        if reconstructed_scaled.shape != target_raw.shape:
            raise ValueError(
                "reconstruction and target shapes differ: "
                f"{tuple(reconstructed_scaled.shape)} != {tuple(target_raw.shape)}"
            )
        if target_raw.ndim != 2:
            raise ValueError(
                f"reconstruction metrics require [samples, hidden], got {target_raw.shape}"
            )
        if target_raw.shape[0] == 0:
            return
        scale_tensor = torch.as_tensor(
            activation_scale,
            dtype=torch.float64,
            device=reconstructed_scaled.device,
        )
        if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor).item():
            raise ValueError("activation_scale must be one finite scalar")
        scale = float(scale_tensor.item())
        if scale <= 0.0:
            raise ValueError(f"activation_scale must be positive, got {scale}")

        reconstructed = reconstructed_scaled.detach().to(dtype=torch.float64)
        target = target_raw.detach().to(
            device=reconstructed_scaled.device, dtype=torch.float64
        )
        residual_scaled = reconstructed - target * scale
        scaled_sse = residual_scaled.square().sum()
        raw_sse = scaled_sse / (scale * scale)

        baseline = mean_predictor_raw.detach().to(
            device=reconstructed_scaled.device, dtype=torch.float64
        )
        if baseline.ndim == 1:
            if baseline.shape[0] != target.shape[1]:
                raise ValueError(
                    f"baseline hidden size {baseline.shape[0]} != {target.shape[1]}"
                )
        elif baseline.shape != target.shape:
            raise ValueError(
                "mean predictor must be [hidden] or [samples, hidden], got "
                f"{tuple(baseline.shape)}"
            )
        baseline_sse = (target - baseline).square().sum()

        samples = target.shape[0]
        elements = target.numel()
        update = torch.zeros_like(self.totals)
        update[self._SCALED_SSE] = scaled_sse.to(self.device)
        update[self._RAW_SSE] = raw_sse.to(self.device)
        update[self._BASELINE_SSE] = baseline_sse.to(self.device)
        update[self._SAMPLES] = samples
        update[self._ELEMENTS] = elements
        if active_counts is not None:
            if active_counts.ndim != 1 or active_counts.shape[0] != samples:
                raise ValueError(
                    "active_counts must have shape [samples], got "
                    f"{tuple(active_counts.shape)}"
                )
            active_counts = active_counts.detach()
            update[self._ACTIVE_ENTRIES] = active_counts.sum().to(
                device=self.device, dtype=torch.float64
            )
            update[self._ENCODED_SAMPLES] = samples
            update[self._ZERO_CODE_SAMPLES] = (active_counts == 0).sum().to(
                device=self.device, dtype=torch.float64
            )
        elif features is not None:
            if features.ndim != 2 or features.shape[0] != samples:
                raise ValueError(
                    "features must have shape [samples, dictionary], got "
                    f"{tuple(features.shape)}"
                )
            active_per_sample = (features.detach() != 0).sum(dim=-1)
            update[self._ACTIVE_ENTRIES] = active_per_sample.sum().to(
                device=self.device, dtype=torch.float64
            )
            update[self._ENCODED_SAMPLES] = samples
            update[self._ZERO_CODE_SAMPLES] = (active_per_sample == 0).sum().to(
                device=self.device, dtype=torch.float64
            )
        self.totals.add_(update)

    @torch.no_grad()
    def update_sufficient_statistics(
        self,
        *,
        scaled_sse: torch.Tensor | float,
        raw_sse: torch.Tensor | float,
        mean_predictor_sse: torch.Tensor | float,
        samples: int,
        elements: int,
        active_counts: torch.Tensor | None = None,
    ) -> None:
        """Accumulate already-computed training statistics.

        This avoids converting complete reconstruction tensors to FP64 and
        recomputing the same residual used by the optimization objective.
        Validation continues to use :meth:`update` for the full-precision
        reporting path.
        """

        if samples <= 0 or elements <= 0:
            return
        update = torch.zeros_like(self.totals)
        update[self._SCALED_SSE] = torch.as_tensor(
            scaled_sse, device=self.device, dtype=torch.float64
        )
        update[self._RAW_SSE] = torch.as_tensor(
            raw_sse, device=self.device, dtype=torch.float64
        )
        update[self._BASELINE_SSE] = torch.as_tensor(
            mean_predictor_sse, device=self.device, dtype=torch.float64
        )
        update[self._SAMPLES] = samples
        update[self._ELEMENTS] = elements
        if active_counts is not None:
            if active_counts.ndim != 1 or active_counts.shape[0] != samples:
                raise ValueError(
                    "active_counts must have shape [samples], got "
                    f"{tuple(active_counts.shape)}"
                )
            update[self._ACTIVE_ENTRIES] = active_counts.sum().to(
                device=self.device, dtype=torch.float64
            )
            update[self._ENCODED_SAMPLES] = samples
            update[self._ZERO_CODE_SAMPLES] = (active_counts == 0).sum().to(
                device=self.device, dtype=torch.float64
            )
        self.totals.add_(update)

    def merge_(self, other: "ReconstructionMetricAccumulator") -> None:
        self.totals.add_(other.totals.to(self.device))

    def all_reduce_(self, group=None) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.totals, op=dist.ReduceOp.SUM, group=group)

    def reset(self) -> None:
        self.totals.zero_()

    def summary(self) -> ReconstructionSummary:
        values = self.totals.detach().cpu().tolist()
        return ReconstructionSummary(
            samples=int(round(values[self._SAMPLES])),
            elements=int(round(values[self._ELEMENTS])),
            scaled_sse=float(values[self._SCALED_SSE]),
            raw_sse=float(values[self._RAW_SSE]),
            mean_predictor_sse=float(values[self._BASELINE_SSE]),
            active_entries=float(values[self._ACTIVE_ENTRIES]),
            encoded_samples=int(round(values[self._ENCODED_SAMPLES])),
            zero_code_samples=int(round(values[self._ZERO_CODE_SAMPLES])),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"totals": self.totals.detach().cpu()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        totals = state["totals"]
        if totals.numel() != self._SIZE:
            raise ValueError(f"invalid metric state size: {totals.numel()}")
        self.totals.copy_(totals.to(device=self.device, dtype=torch.float64))


def prefix_metrics(
    prefix: str, metrics: Mapping[str, float | int | bool]
) -> dict[str, float | int | bool]:
    clean = prefix.strip("/")
    return {f"{clean}/{key}": value for key, value in metrics.items()}


DASHBOARD_DERIVED_SCALAR_TAGS = frozenset(
    {
        "comparison/train_attainable_fidelity",
        "comparison/validation_attainable_fidelity",
        "comparison/generalization_gap",
        "comparison/remaining_regret",
        "comparison/final_validation_attainable_fidelity",
        "optimization/grad_norm_to_clip",
        "sparsity/effective_l0",
        "cross/validation_fve_volatility_20",
    }
)

# Joint Chunk SAE has a separate pair of decoder-head training scalars.  Keep
# the names in the dashboard registry, but let the training entry point add
# them only for the joint mode so legacy mode event surfaces stay stable.
JOINT_CHUNK_SCALAR_TAGS = frozenset(
    {
        "train/joint_mean_scaled_objective",
        "train/joint_cross_scaled_objective",
        "train/joint_mean_normalized_mse",
        "train/joint_cross_normalized_mse",
        "train/joint_mean_fve",
        "train/joint_cross_fve",
        "train/joint_task_loss",
        "train/joint_alpha",
        "train/joint_k_prefix",
        "validation/joint_mean_scaled_objective",
        "validation/joint_cross_scaled_objective",
        "validation/joint_mean_normalized_mse",
        "validation/joint_cross_normalized_mse",
        "validation/joint_mean_nmse",
        "validation/joint_cross_nmse",
        "validation/joint_mean_fve",
        "validation/joint_cross_fve",
        "validation/joint_mean_effective_l0",
        "validation/joint_cross_effective_l0",
        "validation/joint_mean_zero_code_fraction",
        "validation/joint_cross_zero_code_fraction",
        "validation/joint_k_prefix",
        "validation_full/joint_mean_scaled_objective",
        "validation_full/joint_cross_scaled_objective",
        "validation_full/joint_mean_normalized_mse",
        "validation_full/joint_cross_normalized_mse",
        "validation_full/joint_mean_nmse",
        "validation_full/joint_cross_nmse",
        "validation_full/joint_mean_fve",
        "validation_full/joint_cross_fve",
        "validation_full/joint_mean_effective_l0",
        "validation_full/joint_cross_effective_l0",
        "validation_full/joint_mean_zero_code_fraction",
        "validation_full/joint_cross_zero_code_fraction",
        "validation_full/joint_k_prefix",
    }
)

# Keep the dashboard broad enough to diagnose training, while excluding
# duplicate aliases and large cumulative sufficient statistics such as SSE and
# element counts. These tags can all be reconstructed from metrics.jsonl.
_TRAIN_RECONSTRUCTION_SCALAR_TAGS = frozenset(
    {
        "train/scaled_objective",
        "train/raw_mse",
        "train/mean_predictor_mse",
        "train/normalized_mse",
        "train/fve",
        "train/effective_l0",
        "train/zero_code_fraction",
        "train/auxiliary_loss",
        "train/temporal_contrastive_loss",
        "train/temporal_contrastive_accuracy",
        "train/temporal_pairs",
        *JOINT_CHUNK_SCALAR_TAGS,
    }
)

_VALIDATION_RECONSTRUCTION_METRICS = (
    "scaled_objective",
    "raw_mse",
    "mean_predictor_mse",
    "nmse",
    "fve",
    "effective_l0",
    "zero_code_fraction",
)

_VALIDATION_RECONSTRUCTION_PREFIXES = (
    "validation",
    "validation/a_to_b",
    "validation/b_to_a",
    "validation_full",
    "validation_full/a_to_b",
    "validation_full/b_to_a",
)

DASHBOARD_SOURCE_SCALAR_TAGS = frozenset(
    {
        *(
            f"{prefix}/{metric}"
            for prefix in _VALIDATION_RECONSTRUCTION_PREFIXES
            for metric in _VALIDATION_RECONSTRUCTION_METRICS
        ),
        *_TRAIN_RECONSTRUCTION_SCALAR_TAGS,
        "sparsity/dead_features",
        "sparsity/dead_fraction",
        "sparsity/threshold",
        "sparsity/auxk_candidates",
        "optimizer/grad_norm",
        "optimizer/lr",
        "progress/accepted_training_rows",
        "progress/unique_occurrence_coverage",
        "filter/rejected_rows",
        "filter/balance_dropped_rows",
        "filter/input_norm_threshold",
        "filter/target_norm_threshold",
        "system/elapsed_seconds",
        "system/samples_per_second",
        "system/batch_wait_seconds",
        "system/batch_wait_fraction",
    }
)

DASHBOARD_SCALAR_TAGS = (
    DASHBOARD_SOURCE_SCALAR_TAGS | DASHBOARD_DERIVED_SCALAR_TAGS
)


def parse_fidelity_reference_fves(spec: str | None) -> dict[str, float]:
    """Parse ``mode=value`` fidelity references used for RNF dashboards."""

    references: dict[str, float] = {}
    if spec is None or not spec.strip():
        return references
    for item in spec.split(","):
        mode, separator, raw_value = item.strip().partition("=")
        if not separator or not mode or not raw_value:
            raise ValueError(
                "fidelity reference FVE entries must use mode=value syntax"
            )
        if mode in references:
            raise ValueError(f"duplicate fidelity reference for mode={mode}")
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"fidelity reference FVE for mode={mode} must be finite and positive"
            )
        references[mode] = value
    return references


class FidelityDashboard:
    """Derive task-normalized fidelity metrics for live comparison.

    Raw FVE is measured against a constant train-mean predictor.  That gives
    self-reconstruction objectives a ceiling of one, but Cross has irreducible
    uncertainty because one independently-forwarded chunk cannot determine all
    details of its partner.  ``reference_fve`` is therefore the validation FVE
    of a high-capacity predictor fitted on the same input/target task.  The
    ratio ``observed_fve / reference_fve`` answers the comparable question:
    what fraction of the task's empirically attainable explained variance is
    captured by this sparse dictionary?
    """

    def __init__(
        self,
        *,
        mode: str,
        reference_fve: float | None,
        gradient_clip: float,
        validation_window: int = 20,
    ) -> None:
        self.mode = mode
        self.reference_fve = reference_fve
        self.gradient_clip = float(gradient_clip)
        self.validation_fves: deque[float] = deque(
            maxlen=max(1, int(validation_window))
        )
        self.last_train_attainable_fidelity: float | None = None

    def derive(
        self,
        metrics: Mapping[str, float | int | bool],
    ) -> dict[str, float]:
        derived: dict[str, float] = {}

        grad_norm = metrics.get("optimizer/grad_norm")
        if (
            isinstance(grad_norm, (int, float))
            and math.isfinite(float(grad_norm))
            and self.gradient_clip > 0.0
        ):
            derived["optimization/grad_norm_to_clip"] = (
                float(grad_norm) / self.gradient_clip
            )

        effective_l0 = metrics.get("train/effective_l0")
        if isinstance(effective_l0, (int, float)) and math.isfinite(
            float(effective_l0)
        ):
            derived["sparsity/effective_l0"] = float(effective_l0)

        validation_fve = metrics.get("validation/fve")
        if isinstance(validation_fve, (int, float)) and math.isfinite(
            float(validation_fve)
        ):
            value = float(validation_fve)
            self.validation_fves.append(value)
            if self.mode == "cross":
                mean = sum(self.validation_fves) / len(self.validation_fves)
                variance = sum(
                    (sample - mean) ** 2 for sample in self.validation_fves
                ) / len(self.validation_fves)
                derived["cross/validation_fve_volatility_20"] = math.sqrt(
                    variance
                )

            if self.reference_fve is not None:
                attainable_fidelity = value / self.reference_fve
                derived[
                    "comparison/validation_attainable_fidelity"
                ] = attainable_fidelity
                derived["comparison/remaining_regret"] = (
                    1.0 - attainable_fidelity
                )
                if self.last_train_attainable_fidelity is not None:
                    derived["comparison/generalization_gap"] = (
                        self.last_train_attainable_fidelity
                        - attainable_fidelity
                    )

        if self.reference_fve is None:
            return derived

        train_fve = metrics.get("train/fve")
        if isinstance(train_fve, (int, float)) and math.isfinite(float(train_fve)):
            self.last_train_attainable_fidelity = (
                float(train_fve) / self.reference_fve
            )
            derived[
                "comparison/train_attainable_fidelity"
            ] = self.last_train_attainable_fidelity

        final_fve = metrics.get("validation_full/fve")
        if isinstance(final_fve, (int, float)) and math.isfinite(float(final_fve)):
            derived[
                "comparison/final_validation_attainable_fidelity"
            ] = (
                float(final_fve) / self.reference_fve
            )

        return derived


class TensorBoardLogger:
    """Small rank-aware wrapper around ``SummaryWriter``.

    Importing TensorBoard is intentionally lazy so metric-only jobs do not pay
    its import cost. Non-zero ranks are strict no-ops and never create a log
    directory or event file.
    """

    def __init__(
        self,
        log_dir: str | Path | None,
        *,
        rank: int,
        flush_secs: int = 30,
        max_queue: int = 100,
        purge_step: int | None = None,
        scalar_allowlist: Collection[str] | None = None,
        mode: str | None = None,
        fidelity_reference_fve: float | None = None,
        gradient_clip: float = 0.0,
    ) -> None:
        self.writer = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.scalar_allowlist = (
            frozenset(scalar_allowlist) if scalar_allowlist is not None else None
        )
        self.dashboard = (
            FidelityDashboard(
                mode=mode,
                reference_fve=fidelity_reference_fve,
                gradient_clip=gradient_clip,
            )
            if mode is not None
            else None
        )
        if rank != 0 or self.log_dir is None:
            return
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            max_queue=max_queue,
            flush_secs=flush_secs,
            purge_step=purge_step,
        )

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalars(
        self,
        metrics: Mapping[str, float | int | bool],
        step: int,
    ) -> None:
        if self.writer is None:
            return
        expanded_metrics = dict(metrics)
        if self.dashboard is not None:
            expanded_metrics.update(self.dashboard.derive(metrics))
        for tag, value in expanded_metrics.items():
            if (
                self.scalar_allowlist is not None
                and tag not in self.scalar_allowlist
            ):
                continue
            if isinstance(value, bool):
                scalar = float(value)
            elif isinstance(value, (int, float)):
                scalar = float(value)
            else:
                continue
            self.writer.add_scalar(tag, scalar, global_step=step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
            finally:
                self.writer.close()
                self.writer = None

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
