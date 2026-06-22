# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.diffusion.offloader.flat_storage import FlatModuleOffloadArena, FlatModuleOffloadGroup


def test_flat_offload_arena_switches_byte_packed_cpu_and_arena_views() -> None:
    reasoner = nn.Linear(2, 2, bias=False)
    generator = nn.Linear(2, 2, bias=False)
    reasoner.weight.data.fill_(1.0)
    generator.weight.data.fill_(2.0)

    reasoner_group = FlatModuleOffloadGroup.from_modules(
        "reasoner",
        [reasoner],
        pin_memory=False,
    )
    generator_group = FlatModuleOffloadGroup.from_modules(
        "generator",
        [generator],
        pin_memory=False,
    )
    arena = FlatModuleOffloadArena()
    reasoner_cpu_ptr = reasoner.weight.untyped_storage().data_ptr()
    generator_cpu_ptr = generator.weight.untyped_storage().data_ptr()

    arena.materialize(reasoner_group, torch.device("cpu"))
    assert arena.weight is not None
    assert arena.weight.dtype == torch.uint8
    assert reasoner_group.gpu_weights is not None
    assert reasoner.weight.untyped_storage().data_ptr() == arena.weight.untyped_storage().data_ptr()
    torch.testing.assert_close(reasoner.weight, torch.ones_like(reasoner.weight))
    reasoner_group.offload()
    assert reasoner_group.gpu_weights is None
    assert reasoner.weight.untyped_storage().data_ptr() == reasoner_cpu_ptr

    arena.materialize(generator_group, torch.device("cpu"))
    assert generator_group.gpu_weights is not None
    assert generator.weight.untyped_storage().data_ptr() == arena.weight.untyped_storage().data_ptr()
    torch.testing.assert_close(generator.weight, torch.full_like(generator.weight, 2.0))
    generator_group.offload()
    assert generator_group.gpu_weights is None
    assert generator.weight.untyped_storage().data_ptr() == generator_cpu_ptr

    arena.materialize(reasoner_group, torch.device("cpu"))
    torch.testing.assert_close(reasoner.weight, torch.ones_like(reasoner.weight))
    assert reasoner.weight.untyped_storage().data_ptr() == arena.weight.untyped_storage().data_ptr()
