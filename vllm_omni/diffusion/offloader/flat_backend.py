# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat-storage model-level CPU offload backend.

This is the fast, packed-pinned-memory variant of model-level offloading.  It
mirrors the TensorRT-LLM visual-gen offload design: immutable module weights are
byte-packed into pinned CPU storage (CPU is the source of truth) and staged into
a single reusable GPU arena, so each swap is one large bandwidth-friendly H2D
copy with no device-to-host leg.

Compared to :class:`ModelLevelOffloadBackend` (which moves each parameter with
``.to(device)`` / ``.to("cpu")``), this backend trades extra setup time and code
complexity for fewer, larger transfers and a persistent pinned CPU master copy.
It is opt-in via ``offload_use_flat_storage`` so it can be A/B compared against
the default backend.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

from .base import OffloadBackend, OffloadConfig
from .flat_storage import FlatGroupOffloadManager
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class FlatGroupOffloadHook(ModelHook):
    """Activate a manager group before the hooked module's forward.

    Registered on each swappable module (DiT / encoder).  When the module runs,
    its group is materialized on the GPU arena and every other group is offloaded
    to CPU, giving mutual-exclusion GPU residency.
    """

    _HOOK_NAME = "flat_group_offload"

    def __init__(self, manager: FlatGroupOffloadManager, group_name: str):
        self.manager = manager
        self.group_name = group_name

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        self.manager.activate(self.group_name)
        return args, kwargs


def remove_flat_group_offload(modules: list[nn.Module]) -> None:
    """Remove flat-group offload hooks from modules."""
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(FlatGroupOffloadHook._HOOK_NAME)
            logger.debug("Removed flat offload hook from %s", module.__class__.__name__)


class FlatModelLevelOffloadBackend(OffloadBackend):
    """Model-level offloading backend backed by :class:`FlatGroupOffloadManager`.

    Each DiT and each encoder becomes its own offload group; VAE(s) and declared
    resident modules stay on the GPU.  Exactly one group is GPU-resident at a
    time, so a DiT running offloads the encoders (and any other DiT), and an
    encoder running offloads the DiTs.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # modules carrying hooks
        self._manager: FlatGroupOffloadManager | None = None
        self._custom_pipeline: nn.Module | None = None

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("FlatModelLevelOffloadBackend already enabled")
            return

        # Pipelines with a nested transformer (e.g. Cosmos3's reasoner/generator
        # pathways) own their own mutual-exclusion swaps and cannot be offloaded
        # with the generic encoder<->DiT hooks. Delegate to the custom hook.
        custom_enable = getattr(pipeline, "enable_omni_model_cpu_offload", None)
        if callable(custom_enable):
            custom_enable(
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=self.config.use_hsdp,
                use_flat_storage=self.config.use_flat_storage,
            )
            self._custom_pipeline = pipeline
            self.enabled = True
            logger.info(
                "Flat-storage model-level offloading enabled through %s.enable_omni_model_cpu_offload",
                pipeline.__class__.__name__,
            )
            return

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping flat model-level offloading")
            return

        if not modules.encoders:
            # Nothing to swap against — move DiTs to GPU and skip offloading.
            for dit in modules.dits:
                dit.to(self.device)
            logger.warning("No encoder modules found, skipping flat model-level offloading")
            return

        # Each swappable module is its own group; only one is GPU-resident at a
        # time. Modules that must be co-resident within a single forward pass
        # would need to share a group (not the case for component swaps today).
        group_specs: dict[str, list[nn.Module]] = {}
        group_modules: list[tuple[nn.Module, str]] = []
        for i, dit in enumerate(modules.dits):
            name = f"dit_{i}"
            group_specs[name] = [dit]
            group_modules.append((dit, name))
        for j, enc in enumerate(modules.encoders):
            name = f"encoder_{j}"
            group_specs[name] = [enc]
            group_modules.append((enc, name))

        self._manager = FlatGroupOffloadManager(
            group_specs,
            device=self.device,
            resident_modules=[*modules.vaes, *modules.resident_modules],
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
        )
        self._manager.enable()

        for module, group_name in group_modules:
            registry = HookRegistry.get_or_create(module)
            registry.register_hook(
                FlatGroupOffloadHook._HOOK_NAME,
                FlatGroupOffloadHook(self._manager, group_name),
            )
            self._offload_modules.append(module)

        self.enabled = True
        logger.info(
            "Flat-storage model-level offloading enabled: %s <-> %s (mutual exclusion)%s",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
            f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        if self._custom_pipeline is not None:
            custom_disable = getattr(self._custom_pipeline, "disable_omni_model_cpu_offload", None)
            if callable(custom_disable):
                custom_disable()
            self._custom_pipeline = None
            self.enabled = False
            logger.info("Flat-storage model-level offloading disabled")
            return

        remove_flat_group_offload(self._offload_modules)
        self._offload_modules.clear()
        if self._manager is not None:
            self._manager.disable()
            self._manager = None
        self.enabled = False
        logger.info("Flat-storage model-level offloading disabled")
