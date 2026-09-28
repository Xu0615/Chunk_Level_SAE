from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .utils import distributed_info


def initialize_distributed(
    *,
    backend: str = "nccl",
    timeout: timedelta = timedelta(hours=24),
) -> tuple[int, int, int]:
    """Initialize the process group and return ``(rank, world_size, local_rank)``."""

    rank, world_size, local_rank = distributed_info()
    if world_size > 1 and not dist.is_initialized():
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL distributed execution requires CUDA")
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend,
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timeout,
            )
        else:
            dist.init_process_group(backend, timeout=timeout)
    return rank, world_size, local_rank


def broadcast_object(value: Any, *, rank: int, source: int = 0) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def all_gather_objects(value: Any, *, world_size: int) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    output: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(output, value)
    return output


def _parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(part))
    return result


def _gpu_inventory() -> list[dict[str, Any]]:
    """Read the driver inventory without confusing CUDA index and device minor."""

    system_root = Path(os.environ.get("SYSTEM_FS_ROOT", os.sep))
    root = system_root / "proc" / "driver" / "nvidia" / "gpus"
    inventory: list[dict[str, Any]] = []
    if not root.is_dir():
        return inventory
    for information_path in sorted(root.glob("*/information")):
        fields: dict[str, str] = {}
        for line in information_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        bus = fields.get("Bus Location")
        uuid = fields.get("GPU UUID")
        if not bus or not uuid:
            continue
        numa_path = system_root / "sys" / "bus" / "pci" / "devices" / bus / "numa_node"
        node = (
            int(numa_path.read_text(encoding="utf-8").strip())
            if numa_path.is_file()
            else -1
        )
        inventory.append(
            {
                "bus": bus.lower(),
                "uuid": uuid.removeprefix("GPU-").lower(),
                "minor": int(fields.get("Device Minor", -1)),
                "numa_node": node if node >= 0 else None,
            }
        )
    return inventory


def _cuda_uuid(local_rank: int) -> str | None:
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        return None
    value = str(torch.cuda.get_device_properties(local_rank).uuid).lower()
    return value.removeprefix("gpu-")


def cuda_device_numa_node(local_rank: int) -> int | None:
    uuid = _cuda_uuid(local_rank)
    if uuid is None:
        return None
    for item in _gpu_inventory():
        if item["uuid"] == uuid:
            return item["numa_node"]
    return None


def bind_local_rank_cpu_affinity(
    *,
    local_rank: int,
    local_world_size: int,
) -> list[int] | None:
    """Bind a rank near its actual CUDA device, identified by GPU UUID.

    ``CUDA_VISIBLE_DEVICES`` numeric entries are NVIDIA indices, not Linux
    device-minor numbers. Matching them to ``Device Minor`` silently binds the
    rank to the wrong NUMA node on systems whose index/minor orders differ.
    """

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    allowed = set(os.sched_getaffinity(0))
    nodes = [cuda_device_numa_node(index) for index in range(local_world_size)]
    node = nodes[local_rank] if local_rank < len(nodes) else None
    if node is not None:
        system_root = Path(os.environ.get("SYSTEM_FS_ROOT", os.sep))
        cpulist_path = (
            system_root / "sys" / "devices" / "system" / "node"
            / f"node{node}" / "cpulist"
        )
        if cpulist_path.is_file():
            node_cpus = [
                cpu
                for cpu in _parse_cpu_list(
                    cpulist_path.read_text(encoding="utf-8")
                )
                if cpu in allowed
            ]
            peers = [
                index for index, peer_node in enumerate(nodes) if peer_node == node
            ]
            peer_index = peers.index(local_rank)
            assigned = node_cpus[peer_index:: max(1, len(peers))]
            if assigned:
                os.sched_setaffinity(0, assigned)
                return assigned
    ordered = sorted(allowed)
    assigned = ordered[local_rank:: max(1, local_world_size)]
    if assigned:
        os.sched_setaffinity(0, assigned)
        return assigned
    return None


def configure_cpu_threads(default: int = 4) -> int:
    configured = int(
        os.environ.get(
            "SAE_CPU_THREADS",
            os.environ.get("OMP_NUM_THREADS", str(default)),
        )
        or default
    )
    configured = max(1, configured)
    torch.set_num_threads(configured)
    try:
        torch.set_num_interop_threads(max(1, min(2, configured)))
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before parallel work.
        pass
    return configured


@dataclass
class PerformanceCounters:
    """Small process-local timing accumulator suitable for manifests/logging."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)

    def add(self, name: str, seconds: float, count: int = 1) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + int(count)

    def timer(self, name: str, count: int = 1) -> "_CounterTimer":
        return _CounterTimer(self, name, count)

    def summary(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "totals": dict(sorted(self.totals.items())),
            "counts": dict(sorted(self.counts.items())),
            "averages": {
                name: self.totals[name] / max(1, self.counts.get(name, 0))
                for name in sorted(self.totals)
            },
        }

    def log_json(self, prefix: str) -> None:
        print(f"{prefix} {json.dumps(self.summary(), sort_keys=True)}", flush=True)


class _CounterTimer:
    def __init__(
        self,
        counters: PerformanceCounters,
        name: str,
        count: int,
    ) -> None:
        self.counters = counters
        self.name = name
        self.count = count
        self.started = 0.0

    def __enter__(self) -> "_CounterTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.counters.add(
            self.name,
            time.perf_counter() - self.started,
            self.count,
        )
