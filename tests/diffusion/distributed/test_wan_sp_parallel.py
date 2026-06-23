import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.autoencoders import wan_sp_parallel
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.platforms import current_omni_platform

# Module-level markers apply to the CPU unit tests below. The multi-GPU
# correctness test at the end of the file adds the nightly/distributed markers
# and a skip guard so it only runs where >= 2 CUDA devices are available.
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


def test_split_for_parallel_decode_rejects_invalid_split_dim():
    x = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="split_dim"):
        wan_sp_parallel.split_for_parallel_decode(
            x,
            upsample_count=1,
            split_dim="depth",
            rank=0,
            world_size=2,
        )


def test_split_for_parallel_decode_rejects_zero_world_size():
    x = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="world_size"):
        wan_sp_parallel.split_for_parallel_decode(
            x,
            upsample_count=1,
            rank=0,
            world_size=0,
        )


def test_split_for_parallel_decode_rejects_rank_out_of_range():
    x = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="rank"):
        wan_sp_parallel.split_for_parallel_decode(
            x,
            upsample_count=1,
            rank=3,
            world_size=3,
        )


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
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.dist.get_world_size",
        lambda group=None: 4,
    )

    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    vae.use_tiling = True
    vae.distributed_executor = SimpleNamespace(group=object(), parallel_size=2, parallel_mode="sp_height")
    vae.is_distributed_enabled = lambda: True

    z = torch.zeros((1, 16, 1, 8, 8))

    assert vae._sp_decode_split_dim() == "height"
    assert vae._sp_decode_enabled(z) is False


def test_sp_width_gate_selects_width(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan.dist.get_world_size",
        lambda group=None: 2,
    )

    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    vae.distributed_executor = SimpleNamespace(group=object(), parallel_size=2, parallel_mode="sp_width")
    vae.is_distributed_enabled = lambda: True

    z = torch.zeros((1, 16, 1, 8, 8))

    assert vae._sp_decode_split_dim() == "width"
    assert vae._sp_decode_enabled(z) is True


def test_tile_mode_disables_sp_decode():
    vae = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    vae.distributed_executor = SimpleNamespace(group=object(), parallel_size=2, parallel_mode="tile")
    vae.is_distributed_enabled = lambda: True

    z = torch.zeros((1, 16, 1, 8, 8))

    assert vae._sp_decode_split_dim() is None
    assert vae._sp_decode_enabled(z) is False


# =============================================================================
# Multi-GPU numerical-correctness test (nightly / full_model, distributed CUDA)
#
# Spawns a small process group and verifies that sp_height/sp_width decode match
# a single-process (non-distributed) reference decode of the same latent within
# tolerance. Requires >= 2 accelerator devices and downloads a Wan VAE.
# =============================================================================

_SP_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_SP_SUBFOLDER = "vae"
_SP_WORLD_SIZE = 2
_SP_LATENT_FRAMES = 5
_SP_LATENT_HEIGHT = 60
_SP_LATENT_WIDTH = 104
_SP_TOLERANCE = 3e-2


def _sp_decode_worker(rank: int, split_dim: str, return_dict, master_port: str) -> None:
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    dtype = torch.float32

    backend = current_omni_platform.dist_backend
    init_distributed_environment(world_size=_SP_WORLD_SIZE, rank=rank, local_rank=rank, backend=backend)
    initialize_model_parallel(sequence_parallel_size=_SP_WORLD_SIZE, ulysses_degree=_SP_WORLD_SIZE, backend=backend)

    try:
        vae = DistributedAutoencoderKLWan.from_pretrained(_SP_MODEL, subfolder=_SP_SUBFOLDER, torch_dtype=dtype)
        vae.to(device=device, dtype=dtype)
        vae.eval()

        generator = torch.Generator(device=device).manual_seed(0)
        latents = torch.randn(
            (1, vae.config.z_dim, _SP_LATENT_FRAMES, _SP_LATENT_HEIGHT, _SP_LATENT_WIDTH),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        with torch.inference_mode():
            # Ground-truth reference: standard non-parallel, untiled decode (computed identically on
            # every rank). Tiling must be OFF so neither the tile-parallel nor the single-GPU tiled
            # path is exercised; otherwise we would be comparing SP against tiled decode.
            vae.use_tiling = False
            vae.set_parallel_size(1, mode="tile")
            reference = vae.decode(latents, return_dict=False)[0].float()

            # Spatially-sharded decode across the full group (requires tiling to enter tiled_decode).
            vae.use_tiling = True
            vae.set_parallel_size(_SP_WORLD_SIZE, mode=f"sp_{split_dim}")
            sharded = vae.decode(latents, return_dict=False)[0].float()

        diff = (sharded - reference).abs()
        if rank == 0:
            return_dict["max_abs_diff"] = diff.max().item()
            return_dict["mean_abs_diff"] = diff.mean().item()
            return_dict["shape"] = tuple(sharded.shape)
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.distributed_cuda
@pytest.mark.skipif(
    current_omni_platform.get_device_count() < _SP_WORLD_SIZE,
    reason="Requires >= 2 accelerator devices",
)
@pytest.mark.parametrize("split_dim", ["height", "width"])
def test_sp_decode_matches_reference(split_dim: str):
    manager = mp.get_context("spawn").Manager()
    return_dict = manager.dict()
    # Use a per-split-dim port to avoid collisions across parametrized runs.
    master_port = str(29500 + (1 if split_dim == "width" else 0))

    mp.spawn(
        _sp_decode_worker,
        args=(split_dim, return_dict, master_port),
        nprocs=_SP_WORLD_SIZE,
        join=True,
    )

    assert "max_abs_diff" in return_dict, "rank 0 did not report a result"
    max_abs_diff = return_dict["max_abs_diff"]
    mean_abs_diff = return_dict["mean_abs_diff"]
    print(
        f"sp_{split_dim} vs reference: max_abs_diff={max_abs_diff:.6e} "
        f"mean_abs_diff={mean_abs_diff:.6e} shape={return_dict.get('shape')}"
    )
    assert max_abs_diff <= _SP_TOLERANCE, f"sp_{split_dim} max_abs_diff {max_abs_diff} exceeds {_SP_TOLERANCE}"
    assert mean_abs_diff <= _SP_TOLERANCE, f"sp_{split_dim} mean_abs_diff {mean_abs_diff} exceeds {_SP_TOLERANCE}"
