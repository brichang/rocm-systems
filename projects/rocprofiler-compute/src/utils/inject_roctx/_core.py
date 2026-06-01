# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-agnostic ROCTX scope core.

Owns the per-thread marker/context/tier stacks, the Python-tier rangePush
/rangePop hookup, and the optional native-tier hook used by the C++
RecordFunction backend. Backends call _push_scope/_pop_scope and the
helpers defined here; they must not touch the module-level state directly.
"""

import inspect
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


class NativeTierHook(Protocol):
    def active(self) -> bool: ...
    def push(self, marker: str, context: str) -> bool: ...
    def pop(self) -> None: ...


def _missing_range_push(_label: str) -> None:
    raise RuntimeError(
        "inject_roctx._core: Python tier rangePush is not configured.",
    )


def _missing_range_pop() -> None:
    raise RuntimeError(
        "inject_roctx._core: Python tier rangePop is not configured.",
    )


_range_push: Callable[[str], None] = _missing_range_push
_range_pop: Callable[[], None] = _missing_range_pop
_native_tier_hook: Optional[NativeTierHook] = None
_framework_roots: list[str] = []


def set_python_tier_io(
    push: Callable[[str], None],
    pop: Callable[[], None],
) -> None:
    global _range_push, _range_pop
    _range_push = push
    _range_pop = pop


def set_native_tier_hook(hook: Optional[NativeTierHook]) -> None:
    global _native_tier_hook
    _native_tier_hook = hook


def get_native_tier_hook() -> Optional[NativeTierHook]:
    return _native_tier_hook


def add_framework_root(path: str) -> None:
    if path and path not in _framework_roots:
        _framework_roots.append(path)


# Per-thread stacks. Source of truth on Python tier; mirror on C++ tier.
_thread_local = threading.local()


def get_marker_stack() -> list[str]:
    if not hasattr(_thread_local, "marker_stack"):
        _thread_local.marker_stack = []
    return _thread_local.marker_stack


def get_context_stack() -> list[str]:
    if not hasattr(_thread_local, "context_stack"):
        _thread_local.context_stack = []
    return _thread_local.context_stack


def get_tier_stack() -> list[bool]:
    # Each frame records the tier that handled its push (True for the C++
    # RecordFunction tier, False for the Python tier) so _pop_scope can
    # route the matching pop even when tiers fall through mid-stack.
    if not hasattr(_thread_local, "tier_stack"):
        _thread_local.tier_stack = []
    return _thread_local.tier_stack


def resolve_user_caller_location() -> str:
    """'file:line' for the nearest user frame, or 'python.dispatch:0'."""
    this_file = __file__
    frame = inspect.currentframe()
    while frame is not None:
        fn_path = frame.f_code.co_filename
        if fn_path != this_file and not any(
            fn_path.startswith(root) for root in _framework_roots
        ):
            return f"{Path(fn_path).name}:{frame.f_lineno}"
        frame = frame.f_back
    return "python.dispatch:0"


# Wire format: "<op_path>:#N@file:line/..." split on ":#" by utils_analysis.py.


def _push_scope(marker: str, context: str) -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    used_native = False
    hook = _native_tier_hook
    if hook is not None and hook.active():
        try:
            used_native = bool(hook.push(marker, context))
        except Exception:
            used_native = False

    if not used_native:
        # Build full marker before mutating stacks so this frame appears once.
        full = (
            "/".join([*marker_stack, marker])
            + ":"
            + "/".join([*context_stack, context])
        )
        _range_push(full)

    # Bookkeeping after the push. If a stack append raises mid-sequence,
    # undo any partial state and issue the matching pop on the tier so
    # subsequent _pop_scope calls stay balanced.
    try:
        tier_stack.append(used_native)
        marker_stack.append(marker)
        context_stack.append(context)
    except Exception:
        if len(tier_stack) > len(marker_stack):
            tier_stack.pop()
        try:
            if used_native and _native_tier_hook is not None:
                _native_tier_hook.pop()
            else:
                _range_pop()
        except Exception:
            pass
        raise


def _pop_scope() -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    # Unmatched _pop_scope: no-op to avoid over-popping or masking caller exceptions.
    if not tier_stack:
        return

    used_native = tier_stack.pop()
    try:
        if used_native and _native_tier_hook is not None:
            _native_tier_hook.pop()
        else:
            _range_pop()
    finally:
        if marker_stack:
            marker_stack.pop()
        if context_stack:
            context_stack.pop()


# Structural wrapper primitives: entry points bracketed via these helpers
# bypass the ATen dispatcher, so neither tier records them automatically.


def roctx_wrapper(
    func: Callable[..., Any],
    name: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap func with a ROCTX range. Idempotent via _roctx_wrapped."""
    if getattr(func, "_roctx_wrapped", False):
        return func
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> object:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(func_name, f"#{call_counter['count']}@{location}")
        try:
            return func(*args, **kwargs)
        finally:
            _pop_scope()

    wrapper._roctx_wrapped = True
    return wrapper


def _marker_only_init_wrapper(name: str) -> Callable[..., Any]:
    """__init__ that emits a ROCTX range, then calls object.__init__(self).

    For classes whose real construction lives in __new__ (cuda.Event/Stream),
    where __init__ inherits from object and would reject forwarded kwargs.
    """
    call_counter = {"count": 0}

    def marker_only_init(self: object, *args: Any, **kwargs: Any) -> None:
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(name, f"#{call_counter['count']}@{location}")
        try:
            return object.__init__(self)
        finally:
            _pop_scope()

    marker_only_init._roctx_wrapped = True
    return marker_only_init


def _walk_subclasses(cls: type, fn: Callable[[type], None]) -> None:
    """Apply `fn` to every (transitive) subclass of `cls`."""
    for sub in cls.__subclasses__():
        fn(sub)
        _walk_subclasses(sub, fn)
