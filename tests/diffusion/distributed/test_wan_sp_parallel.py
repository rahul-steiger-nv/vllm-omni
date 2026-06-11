from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import wan_sp_parallel
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_split_for_parallel_decode_pads_uneven_height():
    x = torch.arange(1 * 1 * 1 * 5 * 2, dtype=torch.float32).reshape(1, 1, 1, 5, 2)

    local, expected_height = wan_sp_parallel.split_for_parallel_decode(
        x,
        upsample_count=2,
        rank=2,
        world_size=3,
    )

    assert expected_height == 20
    assert local.shape == (1, 1, 1, 2, 2)
    assert torch.equal(local[..., 0, :], x[..., 4, :])
    assert torch.equal(local[..., 1, :], torch.zeros_like(local[..., 1, :]))


def test_split_for_parallel_decode_pads_uneven_width():
    x = torch.arange(1 * 1 * 1 * 2 * 5, dtype=torch.float32).reshape(1, 1, 1, 2, 5)

    local, expected_width = wan_sp_parallel.split_for_parallel_decode(
        x,
        upsample_count=2,
        split_dim="width",
        rank=2,
        world_size=3,
    )

    assert expected_width == 20
    assert local.shape == (1, 1, 1, 2, 2)
    assert torch.equal(local[..., :, 0], x[..., :, 4])
    assert torch.equal(local[..., :, 1], torch.zeros_like(local[..., :, 1]))


def test_gather_and_trim_height(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (0, 3))

    def fake_all_gather(gathered, x, group=None):
        for idx, output in enumerate(gathered):
            output.copy_(x + idx)

    monkeypatch.setattr(wan_sp_parallel.dist, "all_gather", fake_all_gather)

    x = torch.zeros((1, 1, 1, 2, 1), dtype=torch.float32)
    out = wan_sp_parallel.gather_and_trim_extent(x, expected_extent=5, split_dim="height", group=object())

    assert out.shape == (1, 1, 1, 5, 1)
    assert torch.equal(out.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0]))


def test_gather_and_trim_width(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (0, 3))

    def fake_all_gather(gathered, x, group=None):
        for idx, output in enumerate(gathered):
            output.copy_(x + idx)

    monkeypatch.setattr(wan_sp_parallel.dist, "all_gather", fake_all_gather)

    x = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32)
    out = wan_sp_parallel.gather_and_trim_extent(x, expected_extent=5, split_dim="width", group=object())

    assert out.shape == (1, 1, 1, 1, 5)
    assert torch.equal(out.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0]))


def test_reshard_from_trimmed_height_pads_invalid_rows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (2, 3))

    x = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5, 1)
    token = wan_sp_parallel._SPATIAL_SHARD_CONTEXT.set(
        wan_sp_parallel.SpatialShardContext(
            input_extent=5,
            local_input_extent=2,
            split_dim="height",
            rank=2,
            world_size=3,
        )
    )
    try:
        out = wan_sp_parallel.reshard_from_trimmed_extent(
            x,
            local_extent=2,
            split_dim="height",
            group=object(),
        )
    finally:
        wan_sp_parallel._SPATIAL_SHARD_CONTEXT.reset(token)

    assert out.shape == (1, 1, 1, 2, 1)
    assert torch.equal(out.flatten(), torch.tensor([4.0, 0.0]))


def test_reshard_from_trimmed_width_pads_invalid_columns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (2, 3))

    x = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 1, 5)
    token = wan_sp_parallel._SPATIAL_SHARD_CONTEXT.set(
        wan_sp_parallel.SpatialShardContext(
            input_extent=5,
            local_input_extent=2,
            split_dim="width",
            rank=2,
            world_size=3,
        )
    )
    try:
        out = wan_sp_parallel.reshard_from_trimmed_extent(
            x,
            local_extent=2,
            split_dim="width",
            group=object(),
        )
    finally:
        wan_sp_parallel._SPATIAL_SHARD_CONTEXT.reset(token)

    assert out.shape == (1, 1, 1, 1, 2)
    assert torch.equal(out.flatten(), torch.tensor([4.0, 0.0]))


def test_halo_exchange_single_rank_noop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (0, 1))

    x = torch.randn((1, 1, 1, 4, 2))
    out, recv_top, recv_bottom = wan_sp_parallel.halo_exchange(
        x,
        group=object(),
        halo_size=1,
    )

    assert out is x
    assert recv_top is None
    assert recv_bottom is None


def test_dist_zero_pad_only_applies_global_height_edges(monkeypatch: pytest.MonkeyPatch):
    x = torch.ones((1, 1, 2, 2))

    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (1, 3))
    mid_rank_pad = wan_sp_parallel.WanDistZeroPad2d((0, 1, 1, 1), group=object())
    mid = mid_rank_pad(x)
    assert mid.shape == (1, 1, 2, 3)

    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (2, 3))
    last_rank_pad = wan_sp_parallel.WanDistZeroPad2d((0, 1, 1, 1), group=object())
    last = last_rank_pad(x)
    assert last.shape == (1, 1, 3, 3)


def test_dist_zero_pad_only_applies_global_width_edges(monkeypatch: pytest.MonkeyPatch):
    x = torch.ones((1, 1, 2, 2))

    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (1, 3))
    mid_rank_pad = wan_sp_parallel.WanDistZeroPad2d((1, 1, 0, 0), group=object(), split_dim="width")
    mid = mid_rank_pad(x)
    assert mid.shape == (1, 1, 2, 2)

    monkeypatch.setattr(wan_sp_parallel, "_rank_world", lambda group: (2, 3))
    last_rank_pad = wan_sp_parallel.WanDistZeroPad2d((1, 1, 0, 0), group=object(), split_dim="width")
    last = last_rank_pad(x)
    assert last.shape == (1, 1, 2, 3)


def test_sp_height_gate_falls_back_for_partial_group(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_OMNI_WAN_VAE_PARALLEL_MODE", "sp_height")
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.dist.get_world_size",
        lambda group=None: 4,
    )

    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    vae.use_tiling = True
    vae.distributed_executor = SimpleNamespace(group=object(), parallel_size=2)
    vae.is_distributed_enabled = lambda: True

    z = torch.zeros((1, 16, 1, 8, 8))

    assert vae._sp_height_decode_enabled(z) is False


def test_sp_width_gate_selects_width(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_OMNI_WAN_VAE_PARALLEL_MODE", "sp_width")
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.dist.get_world_size",
        lambda group=None: 2,
    )

    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    vae.distributed_executor = SimpleNamespace(group=object(), parallel_size=2)
    vae.is_distributed_enabled = lambda: True

    z = torch.zeros((1, 16, 1, 8, 8))

    assert vae._sp_decode_split_dim() == "width"
    assert vae._sp_decode_enabled(z) is True
