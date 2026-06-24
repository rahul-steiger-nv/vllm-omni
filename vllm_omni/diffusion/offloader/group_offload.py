# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-forward, component-level CPU offload for a single ``nn.Module``.

Some models swap mutually-exclusive *components* inside their own ``forward``
rather than at pipeline-component boundaries -- e.g. Cosmos3's understanding
(reasoner) component runs once per generation while the generation (generator)
component runs every denoising step. The generic ``SequentialOffloadHook``
cannot drive this because it is triggered by a module's ``forward()`` call, and
these components are phases inside a single transformer ``forward`` (the
generator is a bare ``ModuleList`` with no single forward).

This module is general purpose: a model declares which of its submodules form
each mutually-exclusive component and the machinery here does the swapping, so
future models (e.g. a Cosmos4) only declare components and wrap their forward
phases.

- :class:`GroupOffloadManager` -- keeps exactly one named group GPU-resident at a
  time, moving the others to CPU via the existing model-level ``.to()`` movers
  (:meth:`SequentialOffloadHook._to_cpu` / ``_to_gpu``), reusing their
  pin_memory / DTensor / XPU / ``empty_cache`` handling verbatim. Every weight
  of the owning module that is *not* claimed by a group stays GPU-resident; the
  manager infers this set, so models never enumerate resident submodules.
- :class:`ModelCPUOffloadMixin` -- a declarative mixin: a model lists its groups
  as class metadata and wraps each phase with ``with self._offload_context(name):``;
  the mixin builds and drives the manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class GroupOffloadManager:
    """Mutually-exclusive ``.to()`` group swap driven at explicit phase boundaries.

    Keeps a single named group on the device at a time; all other groups are
    moved to CPU. Group moves reuse the existing
    :class:`~vllm_omni.diffusion.offloader.sequential_backend.SequentialOffloadHook`
    ``_to_cpu`` / ``_to_gpu`` movers (no hook is registered on any forward -- we
    only borrow the movers and call them at the boundaries the model marks).

    Any parameter or buffer reachable from ``owner`` that is *not* owned by one
    of the group modules is treated as resident: it is moved to the device once
    and never offloaded. Residents are inferred from ``owner`` minus the groups,
    so callers only declare the mutually-exclusive groups.
    """

    def __init__(
        self,
        owner: nn.Module,
        group_specs: dict[str, list[nn.Module]],
        *,
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> None:
        if not group_specs:
            raise ValueError("GroupOffloadManager requires at least one group spec")
        # Local import avoids import-order coupling between offloader submodules.
        from .sequential_backend import SequentialOffloadHook

        self.device = torch.device(device)
        self.owner = owner
        self.groups = {name: list(modules) for name, modules in group_specs.items()}
        self._mover = SequentialOffloadHook(
            offload_targets=[], device=self.device, pin_memory=pin_memory, use_hsdp=use_hsdp
        )
        self.enabled = False
        self.active_group: str | None = None

    def _grouped_tensor_ids(self) -> set[int]:
        """ids() of every parameter/buffer claimed by some group module."""
        grouped: set[int] = set()
        for modules in self.groups.values():
            for module in modules:
                grouped.update(id(p) for p in module.parameters())
                grouped.update(id(b) for b in module.buffers())
        return grouped

    def _move_residents(self) -> None:
        """Move everything in ``owner`` that no group claims onto the device."""
        grouped = self._grouped_tensor_ids()
        with torch.no_grad():
            for param in self.owner.parameters():
                if id(param) not in grouped and param.data.device != self.device:
                    param.data = param.data.to(self.device, non_blocking=False)
            for buffer in self.owner.buffers():
                if id(buffer) not in grouped and buffer.device != self.device:
                    buffer.data = buffer.data.to(self.device, non_blocking=False)

    def _offload_group(self, name: str) -> None:
        for module in self.groups[name]:
            self._mover._to_cpu(module)

    def _load_group(self, name: str) -> None:
        for module in self.groups[name]:
            self._mover._to_gpu(module)

    def enable(self) -> None:
        if self.enabled:
            return
        self._move_residents()
        for name in self.groups:
            self._offload_group(name)
        self.enabled = True
        self.active_group = None
        logger.info(
            "Component-level CPU offload enabled: components %s swap on %s",
            list(self.groups),
            self.device,
        )

    def disable(self) -> None:
        if not self.enabled:
            return
        for name in self.groups:
            self._load_group(name)
        if self.device.type != "cpu":
            current_omni_platform.synchronize()
        self.enabled = False
        self.active_group = None

    def activate(self, name: str) -> None:
        """Make ``name`` the GPU-resident group, offloading all others to CPU."""
        if not self.enabled:
            return
        if name not in self.groups:
            raise ValueError(f"Unknown offload group: {name!r} (known: {list(self.groups)})")
        if self.active_group == name:
            return
        for other in self.groups:
            if other != name:
                self._offload_group(other)
        self._load_group(name)
        self.active_group = name

    @contextmanager
    def context(self, name: str) -> Iterator[None]:
        """Activate ``name`` for the enclosed forward phase."""
        self.activate(name)
        yield


class ModelCPUOffloadMixin:
    """Declarative in-forward, component-level CPU offload for an ``nn.Module``.

    A model mixes this in, declares its mutually-exclusive components as class
    metadata, and wraps each phase of its ``forward`` with
    ``with self._offload_context(name):``. The mixin builds and drives a
    :class:`GroupOffloadManager` from those declarations.

    Subclasses declare ``_offload_group_specs``: maps a component name to the
    dotted attribute paths (relative to ``self``) of the submodules in that
    component. Components are mutually exclusive: only one is GPU-resident at a
    time. Every other weight of the model stays GPU-resident automatically -- the
    manager infers residents as ``self`` minus the declared groups, so models
    never enumerate resident submodules.

    Must be combined with :class:`torch.nn.Module` (it relies on ``parameters()``).
    """

    _offload_group_specs: ClassVar[dict[str, list[str]]] = {}

    # Set on enable(); the class default lets ``device`` work before __init__ runs
    # and for instances that never enable offload.
    _offload_manager: GroupOffloadManager | None = None

    def _resolve_offload_module(self, path: str) -> nn.Module | None:
        obj: Any = self
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if isinstance(obj, nn.Module) else None

    def _build_offload_group_specs(self) -> dict[str, list[nn.Module]]:
        specs: dict[str, list[nn.Module]] = {}
        for name, paths in self._offload_group_specs.items():
            modules: list[nn.Module] = []
            for path in paths:
                module = self._resolve_offload_module(path)
                if module is None:
                    raise ValueError(
                        f"{type(self).__name__} offload group {name!r} references "
                        f"missing or non-module attribute {path!r}"
                    )
                modules.append(module)
            specs[name] = modules
        return specs

    @property
    def device(self) -> torch.device:
        manager = self._offload_manager
        if manager is not None and manager.enabled:
            return manager.device
        return next(self.parameters()).device

    def enable_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> None:
        """Build the offload manager from the declared components and enable it."""
        manager = self._create_model_cpu_offload_manager(
            device=torch.device(device),
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        manager.enable()
        self._offload_manager = manager

    def _create_model_cpu_offload_manager(
        self,
        *,
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> GroupOffloadManager:
        """Create the manager used by ``enable_model_cpu_offload``.

        Kept as a single construction hook so alternate storage implementations
        can be selected later without changing model declarations.
        """
        return GroupOffloadManager(
            self,
            self._build_offload_group_specs(),
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )

    def disable_model_cpu_offload(self) -> None:
        if self._offload_manager is None:
            return
        self._offload_manager.disable()
        self._offload_manager = None

    def _offload_context(self, group: str) -> AbstractContextManager[None]:
        """Activate ``group`` for the enclosed forward phase (no-op if disabled)."""
        if self._offload_manager is None:
            return nullcontext()
        return self._offload_manager.context(group)
