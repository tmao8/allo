# Copyright Zero-Cosim demo. SPDX-License-Identifier: Apache-2.0
"""Stable contract for LLM-generated or library zero-cosim models (Allo / tooling)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol, TypedDict, runtime_checkable


class KernelCycleInfo(TypedDict):
    """Per-kernel cycle metrics for one analytical schedule."""

    start_cycle: int
    latency_cycles: int
    end_cycle: int


class CycleReport(TypedDict):
    """Aggregated cycle report from a zero-cosim model."""

    makespan_cycles: int
    kernels: Dict[str, KernelCycleInfo]


@runtime_checkable
class ZeroCosimModel(Protocol):
    """
    Implementations parse CSynth + ADB (or equivalent) and expose cycle metrics + Perfetto JSON.

    Allo can ``importlib``-load a generated module that defines a concrete class
    satisfying this protocol (see ``zero_cosim.loader.load_zero_cosim_class``).
    """

    def report_cycle(self) -> CycleReport:
        """Return per-kernel and global makespan in cycles."""

    def dump_trace(self, path: str | Path | None = None) -> str:
        """Write Chrome Trace JSON (``traceEvents`` + ``displayTimeUnit``) and return the path."""
