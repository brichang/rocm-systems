# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Public API for inject_roctx.

install_global_wraps accepts a single backend name, a comma-separated
string, or any iterable of backend names. Each requested backend is installed
independently; failures are isolated by _backends.install_many. The reserved
name "api" expands to every known backend ("torch", "triton").
"""

from collections.abc import Iterable
from typing import Union

from ._backends._torch import (
    dump_recordfn_stats,
    install_function_apply_wrappers,
    using_c_tier,
)
from ._core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)

__all__ = [
    "install_global_wraps",
    "using_c_tier",
    "dump_recordfn_stats",
    "install_function_apply_wrappers",
    "_pop_scope",
    "_push_scope",
    "resolve_user_caller_location",
]

# Meta-alias: ROCPROFCOMPUTE_ROCTX_FRAMEWORKS=api installs every backend.
_API_ALIAS = "api"
_API_EXPANSION: tuple[str, ...] = ("torch", "triton")


def install_global_wraps(backends: Union[str, Iterable[str]] = "") -> None:
    """Install ROCTX instrumentation for each backend in backends.

    "api" expands to every known backend. Empty input is a no-op.
    """
    from ._backends import install_many

    if isinstance(backends, str):
        names = [n.strip() for n in backends.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in backends if str(n).strip()]

    expanded: list[str] = []
    for n in names:
        if n == _API_ALIAS:
            expanded.extend(_API_EXPANSION)
        else:
            expanded.append(n)

    if not expanded:
        return
    install_many(expanded)
