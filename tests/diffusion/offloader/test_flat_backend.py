# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the flat-storage model-level offload manager and backend."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.flat_backend import FlatGroupOffloadHook, FlatModelLevelOffloadBackend
from vllm_omni.diffusion.offloader.flat_storage import (
    FlatGroupOffloadManager,
    FlatModelCPUOffloadMixin,
    NaiveGroupOffloadManager,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_manager_activate_swaps_groups_and_keeps_inactive_on_cpu() -> None:
    group_a = nn.Linear(2, 2, bias=False)
    group_b = nn.Linear(2, 2, bias=False)
    group_a.weight.data.fill_(1.0)
    group_b.weight.data.fill_(2.0)

    manager = FlatGroupOffloadManager(
        {"a": [group_a], "b": [group_b]},
        device=torch.device("cpu"),
        pin_memory=False,
    )
    manager.enable()

    # After enable both groups are packed into their own pinned CPU storage and offloaded.
    a_cpu_ptr = group_a.weight.untyped_storage().data_ptr()
    b_cpu_ptr = group_b.weight.untyped_storage().data_ptr()
    assert manager.enabled
    assert manager.active_group is None
    assert manager.groups["a"].gpu_weights is None
    assert manager.groups["b"].gpu_weights is None
    torch.testing.assert_close(group_a.weight, torch.ones_like(group_a.weight))
    torch.testing.assert_close(group_b.weight, torch.full_like(group_b.weight, 2.0))

    manager.activate("a")
    assert manager.active_group == "a"
    assert manager.groups["a"].gpu_weights is not None
    assert manager.groups["b"].gpu_weights is None
    assert manager.arena is not None and manager.arena.weight is not None
    # Active group is arena-backed; inactive group stays CPU-backed.
    assert group_a.weight.untyped_storage().data_ptr() == manager.arena.weight.untyped_storage().data_ptr()
    assert group_b.weight.untyped_storage().data_ptr() == b_cpu_ptr
    torch.testing.assert_close(group_a.weight, torch.ones_like(group_a.weight))

    manager.activate("b")
    assert manager.active_group == "b"
    assert manager.groups["a"].gpu_weights is None
    assert manager.groups["b"].gpu_weights is not None
    # Group a rebound back to its own CPU storage; group b now arena-backed.
    assert group_a.weight.untyped_storage().data_ptr() == a_cpu_ptr
    assert group_b.weight.untyped_storage().data_ptr() == manager.arena.weight.untyped_storage().data_ptr()
    torch.testing.assert_close(group_b.weight, torch.full_like(group_b.weight, 2.0))

    # Re-activating the current group is a no-op.
    manager.activate("b")
    assert manager.active_group == "b"

    manager.disable()
    assert not manager.enabled
    assert manager.arena is None
    torch.testing.assert_close(group_a.weight, torch.ones_like(group_a.weight))
    torch.testing.assert_close(group_b.weight, torch.full_like(group_b.weight, 2.0))


def test_manager_rejects_unknown_group_and_hsdp() -> None:
    with pytest.raises(NotImplementedError, match="does not support HSDP"):
        FlatGroupOffloadManager({"a": [nn.Linear(2, 2)]}, device=torch.device("cpu"), use_hsdp=True)

    manager = FlatGroupOffloadManager({"a": [nn.Linear(2, 2)]}, device=torch.device("cpu"), pin_memory=False)
    manager.enable()
    with pytest.raises(ValueError, match="Unknown offload group"):
        manager.activate("missing")


def test_manager_moves_resident_modules_and_tensors() -> None:
    swappable = nn.Linear(2, 2, bias=False)
    resident = nn.Linear(2, 2, bias=False)
    resident_param = nn.Parameter(torch.zeros(2))

    manager = FlatGroupOffloadManager(
        {"a": [swappable]},
        device=torch.device("cpu"),
        resident_modules=[resident],
        resident_parameters=[resident_param],
        pin_memory=False,
    )
    manager.enable()
    # Resident module weights are untouched by group packing.
    assert resident.weight.device == torch.device("cpu")
    assert resident_param.device == torch.device("cpu")


class _ToyMixinModule(FlatModelCPUOffloadMixin, nn.Module):
    """Toy module declaring two in-forward offload pathways and a resident head."""

    _offload_group_specs = {
        "reasoner": ["reasoner_layers"],
        "generator": ["generator_layers"],
    }
    # "missing" is listed unconditionally and should be skipped when absent.
    _offload_resident_modules = ["head", "missing"]

    def __init__(self) -> None:
        super().__init__()
        self.reasoner_layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.generator_layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.head = nn.Linear(2, 2, bias=False)
        self.bias_param = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._offload_context("reasoner"):
            x = self.reasoner_layers[0](x)
        with self._offload_context("generator"):
            x = self.generator_layers[0](x)
        return self.head(x) + self.bias_param


def test_offload_mixin_declarative_groups_and_residents() -> None:
    model = _ToyMixinModule()

    # Disabled by default: context() is a no-op and device falls back to params.
    assert model._offload_manager is None
    assert model.device == torch.device("cpu")

    model.enable_model_cpu_offload(device=torch.device("cpu"), pin_memory=False)
    manager = model._offload_manager
    assert manager is not None
    assert set(manager.groups) == {"reasoner", "generator"}
    # Own direct parameter (bias_param) is treated as resident, head module resolved,
    # and the "missing" resident path is silently skipped.
    assert model.device == torch.device("cpu")

    out = model(torch.zeros(1, 2))
    assert tuple(out.shape) == (1, 2)
    # After a full forward, the generator pathway is the last activated group.
    assert manager.active_group == "generator"
    assert manager.groups["reasoner"].gpu_weights is None
    assert manager.groups["generator"].gpu_weights is not None

    model.disable_model_cpu_offload()
    assert model._offload_manager is None


def test_offload_mixin_uses_naive_to_manager_when_flat_disabled() -> None:
    model = _ToyMixinModule()

    # use_flat_storage=False selects the .to() baseline (reuses SequentialOffloadHook).
    model.enable_model_cpu_offload(device=torch.device("cpu"), pin_memory=False, use_flat_storage=False)
    manager = model._offload_manager
    assert isinstance(manager, NaiveGroupOffloadManager)
    assert not isinstance(manager, FlatGroupOffloadManager)
    assert set(manager.groups) == {"reasoner", "generator"}
    assert model.device == torch.device("cpu")

    out = model(torch.zeros(1, 2))
    assert tuple(out.shape) == (1, 2)
    # The context-driven swap tracks the active pathway through the same call sites.
    assert manager.active_group == "generator"

    model.disable_model_cpu_offload()
    assert model._offload_manager is None


def test_offload_mixin_rejects_missing_group_module() -> None:
    class BadModule(FlatModelCPUOffloadMixin, nn.Module):
        _offload_group_specs = {"a": ["does_not_exist"]}

        def __init__(self) -> None:
            super().__init__()
            self.real = nn.Linear(2, 2)

    model = BadModule()
    with pytest.raises(ValueError, match="references missing or non-module attribute"):
        model.enable_model_cpu_offload(device=torch.device("cpu"), pin_memory=False)


class _StubPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.text_encoder = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)


def test_flat_model_level_backend_swaps_dit_and_encoder() -> None:
    pipeline = _StubPipeline()
    backend = FlatModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False, use_flat_storage=True),
        torch.device("cpu"),
    )

    backend.enable(pipeline)
    assert backend.enabled
    manager = backend._manager
    assert manager is not None
    assert set(manager.groups) == {"dit_0", "encoder_0"}
    assert manager.active_group is None

    # Running the encoder activates its group and offloads the DiT.
    pipeline.text_encoder(torch.zeros(1, 2))
    assert manager.active_group == "encoder_0"
    assert manager.groups["dit_0"].gpu_weights is None

    # Running the transformer swaps residency (mutual exclusion).
    pipeline.transformer(torch.zeros(1, 2))
    assert manager.active_group == "dit_0"
    assert manager.groups["encoder_0"].gpu_weights is None

    backend.disable()
    assert not backend.enabled
    assert backend._manager is None
    # Hooks removed after disable.
    registry = getattr(pipeline.transformer, "_hook_registry", None)
    assert registry is None or registry.get_hook(FlatGroupOffloadHook._HOOK_NAME) is None


def test_flat_model_level_backend_delegates_to_custom_pipeline_offload() -> None:
    class CustomPipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enable_args = None
            self.disable_called = False

        def enable_omni_model_cpu_offload(self, **kwargs) -> None:
            self.enable_args = kwargs

        def disable_omni_model_cpu_offload(self) -> None:
            self.disable_called = True

    pipeline = CustomPipeline()
    backend = FlatModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False, use_flat_storage=True),
        torch.device("cpu"),
    )

    backend.enable(pipeline)
    assert backend.enabled is True
    assert pipeline.enable_args == {
        "device": torch.device("cpu"),
        "pin_memory": False,
        "use_hsdp": False,
        "use_flat_storage": True,
    }

    backend.disable()
    assert backend.enabled is False
    assert pipeline.disable_called is True
