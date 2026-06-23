# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_PACKED_TENSOR_ALIGNMENT = 16


def _align_offset(offset: int, alignment: int = _PACKED_TENSOR_ALIGNMENT) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


@dataclass
class _FlatTensorEntry:
    name: str
    target: nn.Parameter | torch.Tensor
    dtype: torch.dtype
    shape: torch.Size
    stride: tuple[int, ...]
    offset: int
    nbytes: int


class FlatModuleOffloadGroup:
    """Byte-packed CPU backing storage for immutable module weights.

    This mirrors the TensorRT-LLM visual-gen offload layout: all tensors in a
    group are packed into one aligned uint8 CPU buffer, and activation copies
    that buffer into one reusable uint8 device arena. The original module
    tensors are rebound to typed views of either CPU storage or the active arena.
    """

    def __init__(
        self,
        name: str,
        *,
        parameters: dict[str, nn.Parameter],
        buffers: dict[str, torch.Tensor],
        pin_memory: bool = True,
    ) -> None:
        self.name = name
        self.cpu_storage: torch.Tensor
        self.gpu_storage: torch.Tensor | None = None
        self.entries: list[_FlatTensorEntry] = []
        self.cpu_views: tuple[torch.Tensor, ...] = ()
        self.gpu_views: tuple[torch.Tensor, ...] = ()
        self._gpu_view_key: tuple[int, int, int, torch.device] | None = None
        self._build_cpu_storage(parameters, buffers, pin_memory=pin_memory)
        self.offload()

    @property
    def cpu_weights(self) -> torch.Tensor:
        return self.cpu_storage

    @property
    def gpu_weights(self) -> torch.Tensor | None:
        return self.gpu_storage

    @staticmethod
    def _set_tensor_storage(target: nn.Parameter | torch.Tensor, value: torch.Tensor) -> None:
        target.data = value

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[int, int] | None:
        if tensor.numel() == 0:
            return None
        storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
        return tensor.untyped_storage().data_ptr(), storage_offset_bytes

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _iter_tensors(
        parameters: dict[str, nn.Parameter],
        buffers: dict[str, torch.Tensor],
    ) -> Iterable[tuple[str, nn.Parameter | torch.Tensor, torch.Tensor]]:
        for name, param in parameters.items():
            yield name, param, param
        for name, buffer in buffers.items():
            yield name, buffer, buffer

    def _build_cpu_storage(
        self,
        parameters: dict[str, nn.Parameter],
        buffers: dict[str, torch.Tensor],
        *,
        pin_memory: bool,
    ) -> None:
        offset = 0
        seen_tensors: dict[tuple[int, int], _FlatTensorEntry] = {}
        pending: list[tuple[_FlatTensorEntry, torch.Tensor, bool]] = []

        for name, target, local in self._iter_tensors(parameters, buffers):
            if not local.is_contiguous():
                raise ValueError(
                    f"Cannot byte-pack non-contiguous offload tensor '{self.name}.{name}' "
                    f"with stride {tuple(local.stride())}"
                )

            key = self._storage_key(local)
            alias = seen_tensors.get(key) if key is not None else None
            if alias is None:
                offset = _align_offset(offset)
                entry_offset = offset
                nbytes = self._tensor_nbytes(local)
                offset += nbytes
            else:
                nbytes = self._tensor_nbytes(local)
                if nbytes != alias.nbytes or local.dtype != alias.dtype:
                    raise ValueError(
                        "Shared parameters or buffers with different sizes or dtypes "
                        f"are not supported by FlatModuleOffloadGroup: '{self.name}.{name}'"
                    )
                entry_offset = alias.offset

            entry = _FlatTensorEntry(
                name=name,
                target=target,
                dtype=local.dtype,
                shape=local.shape,
                stride=tuple(local.stride()),
                offset=entry_offset,
                nbytes=nbytes,
            )
            self.entries.append(entry)
            pending.append((entry, local.detach(), alias is None))
            if key is not None and alias is None:
                seen_tensors[key] = entry

        if not self.entries:
            raise ValueError(f"Offload group '{self.name}' has no parameters or buffers")

        self.cpu_storage = torch.empty(
            _align_offset(offset),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        with torch.no_grad():
            for entry, local, should_copy in pending:
                if not should_copy or entry.nbytes == 0:
                    continue
                tensor_bytes = local.reshape(-1).view(torch.uint8).cpu()
                self.cpu_storage.narrow(0, entry.offset, entry.nbytes).copy_(tensor_bytes)

    @staticmethod
    def _storage_view_key(storage: torch.Tensor) -> tuple[int, int, int, torch.device]:
        return (
            storage.untyped_storage().data_ptr(),
            storage.storage_offset(),
            storage.numel(),
            storage.device,
        )

    def _make_view(self, storage: torch.Tensor, entry: _FlatTensorEntry) -> torch.Tensor:
        view = storage.narrow(0, entry.offset, entry.nbytes).view(entry.dtype)
        return view.as_strided(entry.shape, entry.stride)

    def _make_views(self, storage: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self._make_view(storage, entry) for entry in self.entries)

    def _bind_views(self, views: tuple[torch.Tensor, ...]) -> None:
        for entry, view in zip(self.entries, views, strict=True):
            self._set_tensor_storage(entry.target, view)

    def _ensure_gpu_views(self, storage: torch.Tensor) -> None:
        view_key = self._storage_view_key(storage)
        if self._gpu_view_key != view_key:
            self.gpu_views = self._make_views(storage)
            self._gpu_view_key = view_key

    def required_nbytes(self) -> int:
        return self.cpu_storage.numel()

    def materialize(self, device: torch.device, *, non_blocking: bool = False) -> None:
        gpu_storage = torch.empty(
            self.cpu_storage.shape,
            dtype=torch.uint8,
            device=device,
        )
        copy_non_blocking = non_blocking and self.cpu_storage.is_pinned()
        with torch.no_grad():
            gpu_storage.copy_(self.cpu_storage, non_blocking=copy_non_blocking)
        self.gpu_storage = gpu_storage
        self._ensure_gpu_views(gpu_storage)
        self._bind_views(self.gpu_views)

    def materialize_into(
        self,
        arena_storage: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        gpu_storage = arena_storage.narrow(0, 0, self.required_nbytes())
        copy_non_blocking = non_blocking and self.cpu_storage.is_pinned()
        with torch.no_grad():
            gpu_storage.copy_(self.cpu_storage, non_blocking=copy_non_blocking)
        self.gpu_storage = gpu_storage
        self._ensure_gpu_views(gpu_storage)
        self._bind_views(self.gpu_views)

    def offload(self) -> None:
        if not self.cpu_views:
            self.cpu_views = self._make_views(self.cpu_storage)
        self._bind_views(self.cpu_views)
        self.gpu_storage = None

    @classmethod
    def from_modules(
        cls,
        name: str,
        modules: list[nn.Module],
        *,
        extra_parameters: dict[str, nn.Parameter] | None = None,
        extra_buffers: dict[str, torch.Tensor] | None = None,
        pin_memory: bool = True,
    ) -> "FlatModuleOffloadGroup":
        parameters: dict[str, nn.Parameter] = {}
        buffers: dict[str, torch.Tensor] = {}

        for module_idx, module in enumerate(modules):
            prefix = f"module_{module_idx}"
            for param_name, param in module.named_parameters(recurse=True, remove_duplicate=False):
                parameters[f"{prefix}.{param_name}"] = param
            for buffer_name, buffer in module.named_buffers(recurse=True, remove_duplicate=False):
                buffers[f"{prefix}.{buffer_name}"] = buffer

        if extra_parameters:
            parameters.update(extra_parameters)
        if extra_buffers:
            buffers.update(extra_buffers)

        return cls(name, parameters=parameters, buffers=buffers, pin_memory=pin_memory)


class FlatModuleOffloadArena:
    """Reusable uint8 device staging area for byte-packed offload groups."""

    def __init__(self) -> None:
        self.weight: torch.Tensor | None = None

    def _ensure_capacity(self, group: FlatModuleOffloadGroup, device: torch.device) -> None:
        required_nbytes = group.required_nbytes()
        if self.weight is None or self.weight.device != device or self.weight.numel() < required_nbytes:
            self.weight = torch.empty(required_nbytes, dtype=torch.uint8, device=device)

    def materialize(
        self,
        group: FlatModuleOffloadGroup,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> None:
        self._ensure_capacity(group, device)
        assert self.weight is not None
        group.materialize_into(self.weight, non_blocking=non_blocking)

    def clear(self) -> None:
        self.weight = None


class FlatGroupOffloadManager:
    """Model-agnostic swap driver over byte-packed offload groups.

    Given a set of named module groups, the manager keeps exactly one group
    materialized on the GPU at a time (sharing a single reusable
    :class:`FlatModuleOffloadArena`), while every inactive group is rebound to
    its pinned CPU storage.  ``resident`` modules / tensors are moved to the GPU
    once and stay there.

    This generalizes the mutual-exclusion offload pattern (e.g. Cosmos3's
    reasoner/generator pathways, or a pipeline's text-encoder <-> DiT swap) and
    mirrors TensorRT-LLM's ``OffloadPipeline`` staging layer on top of the
    ``ModuleOffloadManager`` packing primitives.

    The active group is the only one bound to arena (GPU) storage; inactive
    groups are always CPU-backed.  This preserves the invariant that no stale
    pointer into the reused GPU arena can be observed outside the active scope.

    HSDP / DTensor sharded weights are not supported because flat byte-packing
    assumes contiguous local storage; ``use_hsdp=True`` raises
    :class:`NotImplementedError`.
    """

    def __init__(
        self,
        group_specs: dict[str, list[nn.Module]],
        *,
        device: torch.device,
        resident_modules: Iterable[nn.Module] | None = None,
        resident_parameters: Iterable[nn.Parameter] | None = None,
        resident_buffers: Iterable[torch.Tensor] | None = None,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> None:
        if use_hsdp:
            raise NotImplementedError("Flat-storage model-level CPU offload does not support HSDP/DTensor weights.")
        if not group_specs:
            raise ValueError("FlatGroupOffloadManager requires at least one group spec")

        self._group_specs = group_specs
        self.device = torch.device(device)
        self._resident_modules = list(resident_modules) if resident_modules is not None else []
        self._resident_parameters = list(resident_parameters) if resident_parameters is not None else []
        self._resident_buffers = list(resident_buffers) if resident_buffers is not None else []
        self.pin_memory = pin_memory
        self.non_blocking = not current_omni_platform.is_xpu()

        self.enabled = False
        self.active_group: str | None = None
        self.groups: dict[str, FlatModuleOffloadGroup] = {}
        self.arena: FlatModuleOffloadArena | None = None

    def _move_residents(self) -> None:
        for module in self._resident_modules:
            module.to(self.device, non_blocking=True)
        with torch.no_grad():
            for param in self._resident_parameters:
                if param is not None:
                    param.data = param.data.to(self.device, non_blocking=True)
            for buffer in self._resident_buffers:
                if buffer is not None:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

    def enable(self) -> None:
        """Build packed CPU storage for each group and offload them all."""
        if self.enabled:
            return
        self.groups = {
            name: FlatModuleOffloadGroup.from_modules(name, modules, pin_memory=self.pin_memory)
            for name, modules in self._group_specs.items()
        }
        self._move_residents()
        self.arena = FlatModuleOffloadArena()
        self.enabled = True

        for group in self.groups.values():
            group.offload()
        self.active_group = None
        if self.device.type != "cpu":
            current_omni_platform.empty_cache()
        logger.info(
            "Flat-storage model-level CPU offload enabled: groups %s swap on %s",
            list(self.groups),
            self.device,
        )

    def disable(self) -> None:
        """Materialize every group back to its own GPU storage and tear down."""
        if not self.enabled:
            return
        for group in self.groups.values():
            group.materialize(self.device)
        if self.device.type != "cpu":
            current_omni_platform.synchronize()
        self.enabled = False
        self.active_group = None
        self.groups.clear()
        if self.arena is not None:
            self.arena.clear()
        self.arena = None

    def activate(self, name: str) -> None:
        """Stage ``name`` on the GPU arena, offloading every other group to CPU."""
        if not self.enabled or self.arena is None:
            return
        if name not in self.groups:
            raise ValueError(f"Unknown offload group: {name!r} (known: {list(self.groups)})")
        if self.active_group == name:
            return

        for other, group in self.groups.items():
            if other != name:
                group.offload()
        self.arena.materialize(self.groups[name], self.device, non_blocking=self.non_blocking)
        self.active_group = name

    @contextmanager
    def context(self, name: str) -> Iterator[None]:
        """Activate ``name`` for the enclosed phase."""
        self.activate(name)
        yield


class NaiveGroupOffloadManager:
    """Baseline ``.to()`` group swap that reuses ``SequentialOffloadHook``.

    Drop-in alternative to :class:`FlatGroupOffloadManager` with the same
    interface (``enable``/``disable``/``activate``/``context``/``device``/
    ``enabled``).  Instead of packing groups into a pinned-CPU + reusable-arena
    layout, it moves each group's modules between CPU and the device using the
    existing model-level ``.to()`` movers
    (:meth:`SequentialOffloadHook._to_cpu` / :meth:`._to_gpu`) -- i.e. the
    pre-flat-storage path, with its pin_memory / DTensor / XPU / ``empty_cache``
    handling reused verbatim.  This exists so callers can A/B the packed-arena
    path against the naive ``.to()`` path through the same offload-context call
    sites (e.g. via ``use_flat_storage``), keeping the comparison apples-to-apples.
    """

    def __init__(
        self,
        group_specs: dict[str, list[nn.Module]],
        *,
        device: torch.device,
        resident_modules: Iterable[nn.Module] | None = None,
        resident_parameters: Iterable[nn.Parameter] | None = None,
        resident_buffers: Iterable[torch.Tensor] | None = None,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> None:
        if not group_specs:
            raise ValueError("NaiveGroupOffloadManager requires at least one group spec")
        # Local import avoids import-order coupling between offloader submodules.
        from .sequential_backend import SequentialOffloadHook

        self.device = torch.device(device)
        self.groups = {name: list(modules) for name, modules in group_specs.items()}
        self._resident_modules = list(resident_modules) if resident_modules is not None else []
        self._resident_parameters = list(resident_parameters) if resident_parameters is not None else []
        self._resident_buffers = list(resident_buffers) if resident_buffers is not None else []
        # Reuse the existing model-level .to swap logic. We don't register the
        # hook on any forward (no offload_targets); we only borrow its
        # _to_cpu/_to_gpu movers and drive them at Cosmos3's phase boundaries.
        self._mover = SequentialOffloadHook(
            offload_targets=[], device=self.device, pin_memory=pin_memory, use_hsdp=use_hsdp
        )
        self.enabled = False
        self.active_group: str | None = None

    def _move_residents(self) -> None:
        for module in self._resident_modules:
            self._mover._to_gpu(module)
        with torch.no_grad():
            for param in self._resident_parameters:
                if param is not None and param.data.device != self.device:
                    param.data = param.data.to(self.device, non_blocking=True)
            for buffer in self._resident_buffers:
                if buffer is not None and buffer.device != self.device:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

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
            "Naive (.to) model-level CPU offload enabled: groups %s swap on %s",
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
        self.activate(name)
        yield


class FlatModelCPUOffloadMixin:
    """Declarative in-forward flat-storage CPU offload for a single ``nn.Module``.

    Some models swap mutually-exclusive *pathways* inside their own ``forward``
    (rather than at pipeline-component boundaries) -- e.g. an understanding
    pathway that runs once and a generation pathway that runs every denoising
    step.  Such a module mixes this in, declares its groups and resident
    submodules as class metadata, and wraps each phase with
    ``with self._offload_context(name):``.  All the heavy lifting (packing, the
    reusable GPU arena, and the mutual-exclusion swaps) is delegated to
    :class:`FlatGroupOffloadManager`.

    Subclasses declare:

    - ``_offload_group_specs``: maps a group name to the dotted attribute paths
      (relative to ``self``) of the submodules packed into that group.  Groups
      are mutually exclusive: only one is GPU-resident at a time.
    - ``_offload_resident_modules``: dotted attribute paths of submodules that
      stay GPU-resident across all phases.  Missing/``None`` paths are skipped,
      so optional submodules can be listed unconditionally.

    The module's own direct parameters and buffers are treated as resident too.

    This mixin must be combined with :class:`torch.nn.Module` (it relies on
    ``parameters()``, ``_parameters`` and ``_buffers``).
    """

    _offload_group_specs: ClassVar[dict[str, list[str]]] = {}
    _offload_resident_modules: ClassVar[list[str]] = []

    # Set on enable(); the class default lets ``device`` work before __init__ runs
    # and for instances that never enable offload.
    _offload_manager: FlatGroupOffloadManager | NaiveGroupOffloadManager | None = None

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

    def _offload_resident_module_list(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for path in self._offload_resident_modules:
            module = self._resolve_offload_module(path)
            if module is not None:
                modules.append(module)
        return modules

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
        use_flat_storage: bool = True,
    ) -> None:
        """Build the offload manager from the declared groups and enable it.

        ``use_flat_storage=True`` (default) uses the packed pinned-CPU +
        reusable-arena :class:`FlatGroupOffloadManager`; ``False`` uses
        :class:`NaiveGroupOffloadManager`, the ``.to()`` baseline that reuses the
        existing ``SequentialOffloadHook`` movers -- so the two can be A/B
        compared through the same offload-context call sites.
        """
        manager_cls: type[FlatGroupOffloadManager] | type[NaiveGroupOffloadManager] = (
            FlatGroupOffloadManager if use_flat_storage else NaiveGroupOffloadManager
        )
        manager = manager_cls(
            self._build_offload_group_specs(),
            device=torch.device(device),
            resident_modules=self._offload_resident_module_list(),
            resident_parameters=list(self._parameters.values()),
            resident_buffers=list(self._buffers.values()),
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        manager.enable()
        self._offload_manager = manager

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
