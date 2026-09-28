"""Helpers used by the backwards-compatible top-level module shims."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def expose(module_name: str, namespace: dict[str, Any]) -> ModuleType:
    """Expose every implementation symbol in a compatibility module.

    Updating ``namespace`` instead of using ``import *`` also preserves the
    private helpers used by the repository's tests and by older scripts.
    """

    target = import_module(module_name)
    for name, value in vars(target).items():
        if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
            namespace[name] = value
    namespace["__wrapped_module__"] = target
    return target
