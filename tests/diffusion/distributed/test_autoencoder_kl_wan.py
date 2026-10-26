# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import TileTask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    sentinel = object()
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = sentinel
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    assert model._execution_context() is sentinel
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


@pytest.mark.parametrize("fail_second_chunk", [False, True])
def test_tile_decode_releases_history_before_assembly_and_on_failure(monkeypatch, fail_second_chunk):
    model = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    torch.nn.Module.__init__(model)
    model._cached_conv_counts = {"decoder": 1, "encoder": 0}
    model.post_quant_conv = torch.nn.Identity()
    cache_refs = []

    class Decoder(torch.nn.Module):
        def forward(self, tile, *, feat_cache, feat_idx, first_chunk):
            assert feat_idx == [0]
            if first_chunk:
                assert feat_cache == [None]
            else:
                # A premature per-chunk clear would lose causal history.
                torch.testing.assert_close(feat_cache[0], torch.ones_like(tile))
            feat_cache[0] = tile.clone()
            cache_refs.append(weakref.ref(feat_cache[0]))
            if fail_second_chunk and not first_chunk:
                raise RuntimeError("injected tile decode failure")
            return tile * 2

    model.decoder = Decoder()
    task = TileTask(0, (0, 0), [torch.ones(1, 3, 1, 2, 2), torch.full((1, 3, 1, 2, 2), 2.0)])
    original_cat = torch.cat
    assembled = []

    def checked_cat(tensors, dim):
        assert model._feat_map == [None]
        assert all(ref() is None for ref in cache_refs)
        assembled.append(True)
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(wan_vae_module.torch, "cat", checked_cat)
    if fail_second_chunk:
        with pytest.raises(RuntimeError, match="injected tile decode failure"):
            model.tile_exec(task)
        assert not assembled
    else:
        output = model.tile_exec(task)
        torch.testing.assert_close(output[:, :, :1], task.tensor[0] * 2)
        torch.testing.assert_close(output[:, :, 1:], task.tensor[1] * 2)
        assert assembled == [True]
    assert model._feat_map == [None]
