# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

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

    def required_numel_by_dtype(self) -> dict[torch.dtype, int]:
        return {torch.uint8: self.required_nbytes()}

    @property
    def is_context_bound(self) -> bool:
        return False

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

    def bind_context(self, arena_storage: torch.Tensor) -> None:
        self.gpu_storage = arena_storage.narrow(0, 0, self.required_nbytes())
        self._ensure_gpu_views(self.gpu_storage)
        self._bind_views(self.gpu_views)

    def materialize_context(self, *, non_blocking: bool = False) -> None:
        if self.gpu_storage is None:
            raise RuntimeError(f"Offload group {self.name!r} is not bound to a device context")
        copy_non_blocking = non_blocking and self.cpu_storage.is_pinned()
        with torch.no_grad():
            self.gpu_storage.copy_(self.cpu_storage, non_blocking=copy_non_blocking)

    def deactivate_context(self) -> None:
        self.offload()

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

    @property
    def weights(self) -> dict[torch.dtype, torch.Tensor]:
        return {torch.uint8: self.weight} if self.weight is not None else {}

    def _ensure_capacity(self, group: FlatModuleOffloadGroup, device: torch.device) -> None:
        required_nbytes = group.required_nbytes()
        if self.weight is None or self.weight.device != device or self.weight.numel() < required_nbytes:
            self.weight = torch.empty(required_nbytes, dtype=torch.uint8, device=device)

    def bind_context(
        self,
        groups: Iterable[FlatModuleOffloadGroup],
        device: torch.device,
    ) -> None:
        groups = list(groups)
        required_nbytes = max(group.required_nbytes() for group in groups)
        if self.weight is None or self.weight.device != device or self.weight.numel() < required_nbytes:
            self.weight = torch.empty(required_nbytes, dtype=torch.uint8, device=device)
        for group in groups:
            group.bind_context(self.weight)

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
