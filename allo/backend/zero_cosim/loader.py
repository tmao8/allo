# Copyright Zero-Cosim demo. SPDX-License-Identifier: Apache-2.0
"""Load agent-generated zero-cosim modules by file path (Allo integration)."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Type


def load_module_from_path(
    module_path: str | Path,
    *,
    module_name: Optional[str] = None,
) -> ModuleType:
    """Load a Python file as a module (caller should restrict ``module_path`` to trusted dirs)."""
    path = Path(module_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    if module_name is None:
        module_name = f"zero_cosim_dynamic_{uuid.uuid4().hex[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_zero_cosim_class(
    module_path: str | Path,
    class_name: str = "ZeroCosimModel",
    *,
    module_name: Optional[str] = None,
) -> Type[Any]:
    """
    Import ``module_path`` and return the named model class.

    Instantiate with your ``solution_dir`` and ``clock_period_ns``, then call
    ``report_cycle()`` / ``dump_trace()``.
    """
    mod = load_module_from_path(module_path, module_name=module_name)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Module {module_path!s} has no class {class_name!r}")
    cls = getattr(mod, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"{class_name!r} is not a class")
    return cls
