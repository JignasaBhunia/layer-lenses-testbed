"""Backward-compatible import aliases for pickles made before the package rename.

Old pickle files may reference modules such as ``first_experiment.relu_mlp``.
Keep those module names importable while the maintained package remains
``layer_lenses``.
"""

from __future__ import annotations

import importlib
import sys

import layer_lenses as _layer_lenses

_SUBMODULES = (
    "analysis",
    "dlgn",
    "experiment_config",
    "experiment_runner",
    "io",
    "multiseed",
    "odt",
    "relu_analysis",
    "relu_mlp",
    "relu_training",
    "training",
)

for _name in _SUBMODULES:
    _module = importlib.import_module(f"layer_lenses.{_name}")
    sys.modules[f"first_experiment.{_name}"] = _module
    globals()[_name] = _module

__all__ = list(getattr(_layer_lenses, "__all__", ())) + list(_SUBMODULES)
