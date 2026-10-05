# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the real Cosmos3 guardrail adapter.

These live apart from ``test_cosmos3_pipeline.py`` because that module installs
an autouse fixture replacing ``guardrails`` in ``sys.modules`` with a stub, so a
test there can never reach the code below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import guardrails

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("chunk_bytes", [1, 64 << 20])
@pytest.mark.parametrize("modify_frames", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_display_conversion_preserves_guardrail_input_and_output_bytes(
    monkeypatch: pytest.MonkeyPatch, chunk_bytes: int, modify_frames: bool, dtype: torch.dtype, device: str
) -> None:
    from diffusers.video_processor import VideoProcessor

    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(pipeline_cosmos3, "_DISPLAY_CHUNK_BYTES", chunk_bytes)
    # Include bf16 rounding boundaries, saturation, and both ends of the range.
    values = torch.tensor([-1.4, -1, -0.50390625, 0, 0.50390625, 1, 1.4], dtype=dtype)
    video = values.view(1, 1, 7, 1, 1).expand(1, 3, 7, 2, 2).clone().to(device)
    original = video.clone()
    captured: list[np.ndarray] = []

    def check(frames: np.ndarray) -> np.ndarray:
        captured.append(frames.copy())
        if modify_frames:
            # Exercise the adapter's handling of modified guardrail output,
            # including every possible byte through the old float round trip.
            return np.arange(256 * 3, dtype=np.int64).astype(np.uint8).reshape(1, 16, 16, 3)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", check)
    old_checked = guardrails.check_video_safety(video)
    reference = VideoProcessor(vae_scale_factor=16).postprocess_video(old_checked, output_type="np")
    expected = np.round(np.clip(reference, 0, 1) * 255).astype(np.uint8)

    narrowed = pipeline_cosmos3.to_display_uint8(video, guardrails_enabled=True)
    checked = guardrails.check_video_safety(narrowed)

    assert np.array_equal(captured[0], captured[1])
    assert np.array_equal(checked.cpu().numpy(), expected)
    assert torch.equal(video, original)
    if dtype == torch.bfloat16:
        assert not torch.equal(narrowed, pipeline_cosmos3.to_display_uint8(video))


def test_check_video_safety_is_a_no_op_when_no_guardrail_is_loaded() -> None:
    frames = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)

    assert guardrails.check_video_safety(frames) is frames


def test_check_video_safety_skips_conversions_for_display_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """uint8 channel-last frames are already the guardrail's own format.

    The float path has to denormalize, scale, round and permute on the way in and
    undo all of it on the way out. Frames that arrive display-ready skip both.
    """
    seen: list[np.ndarray] = []

    def _guardrail(frames: np.ndarray) -> np.ndarray:
        seen.append(frames)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", _guardrail)
    frames = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(1, 2, 4, 4, 3)

    checked = guardrails.check_video_safety(frames)

    assert len(seen) == 1
    assert seen[0].dtype == np.uint8
    assert seen[0].shape == (2, 4, 4, 3)
    assert checked.dtype == torch.uint8
    assert torch.equal(checked, frames)


def test_check_video_safety_accepts_unbatched_display_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guardrails, "_video_guardrail", lambda frames: frames)
    frames = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 4, 3)

    checked = guardrails.check_video_safety(frames)

    assert checked.shape == frames.shape
    assert torch.equal(checked, frames)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (torch.zeros(1, 2, 4, 4, 4, dtype=torch.uint8), "display-frame guardrails expect"),
        (torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8), "one video per request"),
    ],
)
def test_check_video_safety_validates_display_frame_contract(
    monkeypatch: pytest.MonkeyPatch,
    frames: torch.Tensor,
    message: str,
) -> None:
    monkeypatch.setattr(guardrails, "_video_guardrail", lambda value: value)

    with pytest.raises(ValueError, match=message):
        guardrails.check_video_safety(frames)


def test_check_video_safety_still_round_trips_the_vae_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that pass float [-1, 1] must get float [-1, 1] back."""
    captured: list[np.ndarray] = []

    def _guardrail(frames: np.ndarray) -> np.ndarray:
        captured.append(frames)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", _guardrail)
    video = torch.zeros(1, 3, 2, 4, 4)

    checked = guardrails.check_video_safety(video)

    # The guardrail sees uint8 channel-last either way; only the wrapping differs.
    assert captured[0].dtype == np.uint8
    assert captured[0].shape == (2, 4, 4, 3)
    assert checked.shape == video.shape
    assert checked.dtype == torch.float32
    torch.testing.assert_close(checked, video, atol=1 / 127.5, rtol=0)
