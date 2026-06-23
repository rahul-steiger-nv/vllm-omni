# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare diffusion inference across CPU-offload strategies.

Benchmarks end-to-end generation latency and peak GPU memory for the same model
and generation config under each offload mode:

  * ``none``      - no offload (baseline; needs the whole model on GPU)
  * ``layerwise`` - block-wise offload (``--enable-layerwise-offload``)
  * ``model``     - model-level offload, default ``.to()`` swap (``--enable-cpu-offload``)
  * ``flat``      - model-level offload, packed pinned CPU + reusable GPU arena
                    (``--enable-cpu-offload --offload-use-flat-storage``)

Each mode runs in its own subprocess so peak-memory and one-time setup costs are
measured cleanly and the inference engine's global state never leaks between runs.

Examples:
    # Cosmos3-nano on an RTX 5090
    python benchmarks/diffusion/bench_offload_modes.py \
        --model <cosmos3-nano-id> --label "cosmos3-nano @ RTX5090" \
        --height 704 --width 1280 --num-frames 121 --num-inference-steps 35 \
        --modes layerwise,model,flat

    # Cosmos3-super on an RTX 6000
    python benchmarks/diffusion/bench_offload_modes.py \
        --model <cosmos3-super-id> --label "cosmos3-super @ RTX6000" \
        --height 704 --width 1280 --num-frames 121 --num-inference-steps 35 \
        --modes none,layerwise,model,flat --output-json super_rtx6000.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Mode -> Omni kwargs that select the offload strategy.
MODE_TO_OMNI_KWARGS: dict[str, dict[str, bool]] = {
    "none": {},
    "layerwise": {"enable_layerwise_offload": True},
    "model": {"enable_cpu_offload": True},
    "flat": {"enable_cpu_offload": True, "offload_use_flat_storage": True},
}

MODE_LABELS: dict[str, str] = {
    "none": "No offload (baseline)",
    "layerwise": "Layerwise (blockwise)",
    "model": "Model-level (.to swap)",
    "flat": "Model-level (flat arena)",
}


# ---------------------------------------------------------------------------
# Worker: runs a single mode in its own process and writes a JSON result.
# ---------------------------------------------------------------------------
def _is_oom_error(exc: BaseException, message: str) -> bool:
    """Heuristically detect a GPU out-of-memory failure."""
    type_name = type(exc).__name__.lower()
    text = message.lower()
    return (
        "outofmemory" in type_name
        or "out of memory" in text
        or "cuda error: out of memory" in text
        or "hip out of memory" in text
    )


def _extract_peak_memory_mb(result: Any) -> float:
    """Pull worker-reported peak VRAM (MiB) from a generation result.

    Mirrors examples/offline_inference/text_to_video/text_to_video.py.
    """
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return 0.0
    val = getattr(result, "peak_memory_mb", 0.0)
    if not val:
        inner = getattr(result, "request_output", None)
        if isinstance(inner, list):
            inner = inner[0] if inner else None
        val = getattr(inner, "peak_memory_mb", 0.0)
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


class _DeviceMemorySampler:
    """Background poller for true device-level GPU memory (process-agnostic).

    Cross-checks the in-process ``peak_memory_mb`` (reserved pool) against the
    actual device "used" memory reported by NVML (or ``nvidia-smi``), which sees
    every process and every phase. Prefers ``pynvml``; falls back to
    ``nvidia-smi``; degrades to disabled (``available is False``) otherwise.

    Reentrant: use one instance across iterations; ``max_mib`` accumulates the
    peak observed across all sampled windows. Targets the device implied by an
    integer ``CUDA_VISIBLE_DEVICES`` (else index 0); assumes a single-GPU run.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.max_mib = 0.0
        self.backend = "none"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._probe = self._make_probe()

    @property
    def available(self) -> bool:
        return self._probe is not None

    @staticmethod
    def _device_index() -> int:
        first = (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").split(",")[0].strip()
        return int(first) if first.isdigit() else 0

    def _make_probe(self):  # returns callable -> used MiB, or None
        idx = self._device_index()
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            def probe() -> float:
                return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)

            probe()
            self.backend = "pynvml"
            return probe
        except Exception:
            pass

        import shutil

        if shutil.which("nvidia-smi") is not None:

            def probe() -> float:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return float(out.stdout.strip().splitlines()[0])

            try:
                probe()
                self.backend = "nvidia-smi"
                return probe
            except Exception:
                return None
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                used = self._probe()
                if used is not None and used > self.max_mib:
                    self.max_mib = used
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_DeviceMemorySampler":
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def _run_worker(config_path: str) -> None:
    """Run one offload mode and write metrics to ``cfg['result_file']``."""
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    cfg = json.loads(Path(config_path).read_text())
    mode = cfg["mode"]
    result: dict[str, Any] = {"mode": mode, "ok": False}

    try:
        omni_kwargs: dict[str, Any] = dict(
            model=cfg["model"],
            enforce_eager=cfg["enforce_eager"],
            vae_use_slicing=cfg["vae_use_slicing"],
            vae_use_tiling=cfg["vae_use_tiling"],
            **MODE_TO_OMNI_KWARGS[mode],
        )
        for key in ("model_class_name", "quantization"):
            if cfg.get(key):
                omni_kwargs[key] = cfg[key]
        for key in ("boundary_ratio", "flow_shift"):
            if cfg.get(key) is not None:
                omni_kwargs[key] = cfg[key]

        # Guardrail controls (Cosmos3) routed through the HF model_config dict,
        # matching the serve CLI's --no-guardrails (model_config["guardrails"]).
        model_config: dict[str, Any] = {}
        if cfg.get("no_guardrails"):
            model_config["guardrails"] = False
        if cfg.get("offload_guardrails"):
            model_config["offload_guardrail_models"] = True
        if model_config:
            omni_kwargs["model_config"] = model_config

        setup_start = time.perf_counter()
        omni = Omni(**omni_kwargs)
        result["setup_s"] = time.perf_counter() - setup_start
        result["device_name"] = current_omni_platform.get_device_name()

        prompt_dict: dict[str, Any] = {"prompt": cfg["prompt"]}
        if cfg.get("negative_prompt"):
            prompt_dict["negative_prompt"] = cfg["negative_prompt"]

        def _generate() -> Any:
            generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(cfg["seed"])
            sampling_kwargs: dict[str, Any] = dict(
                height=cfg["height"],
                width=cfg["width"],
                num_frames=cfg["num_frames"],
                num_inference_steps=cfg["num_inference_steps"],
                generator=generator,
            )
            if cfg.get("guidance_scale") is not None:
                sampling_kwargs["guidance_scale"] = cfg["guidance_scale"]
            if cfg.get("guidance_scale_high") is not None:
                sampling_kwargs["guidance_scale_2"] = cfg["guidance_scale_high"]
            if cfg.get("extra_body"):
                sampling_kwargs["extra_args"] = dict(cfg["extra_body"])
            return omni.generate(prompt_dict, OmniDiffusionSamplingParams(**sampling_kwargs))

        dims = f"{cfg['width']}x{cfg['height']} x {cfg['num_frames']}f, steps={cfg['num_inference_steps']}"
        warmup = int(cfg["warmup"])
        if warmup <= 0:
            print(f"[worker:{mode}] no warmup; timed runs will include cold-start overhead", flush=True)
        for i in range(warmup):
            print(f"[worker:{mode}] warmup {i + 1}/{warmup} at {dims} (same dimensions as timed runs)", flush=True)
            _generate()

        sampler = _DeviceMemorySampler() if cfg.get("gpu_poll", True) else None

        times_ms: list[float] = []
        peak_mb = 0.0
        for _ in range(cfg["iters"]):
            t0 = time.perf_counter()
            if sampler is not None and sampler.available:
                with sampler:
                    out = _generate()
            else:
                out = _generate()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            peak_mb = max(peak_mb, _extract_peak_memory_mb(out))

        result.update(
            ok=True,
            times_ms=times_ms,
            median_ms=statistics.median(times_ms),
            min_ms=min(times_ms),
            peak_memory_mb=peak_mb,
            peak_device_mb=(sampler.max_mib if sampler is not None else 0.0),
            device_poll_backend=(sampler.backend if sampler is not None else "off"),
        )
    except Exception as exc:  # noqa: BLE001 - record and report, keep other modes going
        import traceback

        message = f"{type(exc).__name__}: {exc}".split("\n", 1)[0]
        result["error"] = message
        result["oom"] = _is_oom_error(exc, str(exc))
        result["traceback"] = traceback.format_exc()

    Path(cfg["result_file"]).write_text(json.dumps(result))


# ---------------------------------------------------------------------------
# Orchestrator: spawns one worker per mode and prints a comparison table.
# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace, mode: str, result_file: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "result_file": result_file,
        "model": args.model,
        "model_class_name": args.model_class_name,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_high": args.guidance_scale_high,
        "boundary_ratio": args.boundary_ratio,
        "flow_shift": args.flow_shift,
        "quantization": args.quantization,
        "extra_body": args.extra_body,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "enforce_eager": args.enforce_eager,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "no_guardrails": args.no_guardrails,
        "offload_guardrails": args.offload_guardrails,
        "gpu_poll": args.gpu_poll,
    }


def _run_mode_subprocess(args: argparse.Namespace, mode: str, workdir: Path) -> dict[str, Any]:
    config_path = workdir / f"cfg_{mode}.json"
    result_path = workdir / f"result_{mode}.json"
    config_path.write_text(json.dumps(_build_config(args, mode, str(result_path))))

    print(f"\n{'=' * 78}\n[bench] Running mode: {mode} ({MODE_LABELS.get(mode, mode)})\n{'=' * 78}", flush=True)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)], check=False)

    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        # No result file usually means the process was hard-killed before it could
        # write — most commonly an OS OOM-kill (negative return code = signal).
        result = {
            "mode": mode,
            "ok": False,
            "oom": None,  # unknown; could not be classified
            "error": f"worker produced no result file (exit code {proc.returncode}; likely OOM-killed)",
        }
    if proc.returncode != 0 and result.get("ok"):
        # Worker wrote a result but the process still failed (e.g. crash on teardown).
        result["ok"] = False
        result.setdefault("error", f"worker exited with code {proc.returncode}")
    if not result.get("ok"):
        tag = "OOM" if result.get("oom") else ("OOM?" if result.get("oom") is None else "FAILED")
        print(f"[bench] mode {mode} {tag}: {result.get('error', 'unknown error')}", flush=True)
    return result


def _print_table(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    device = next((r.get("device_name") for r in results if r.get("device_name")), "unknown")
    print("\n" + "=" * 90)
    print(f"Offload comparison  |  {args.label or args.model}")
    print(f"device={device}  |  {args.width}x{args.height} x {args.num_frames}f  |  steps={args.num_inference_steps}")
    print(f"warmup={args.warmup}  iters={args.iters}  enforce_eager={args.enforce_eager}")
    guardrails = "off" if args.no_guardrails else ("on (offloaded)" if args.offload_guardrails else "on (resident)")
    print(f"guardrails={guardrails}")
    poll_backend = next((r.get("device_poll_backend") for r in results if r.get("device_poll_backend")), None)
    show_dev = args.gpu_poll and poll_backend not in (None, "off", "none")
    print(f"gpu_poll={'on (' + poll_backend + ')' if show_dev else 'off'}")
    print("=" * 100)
    dev_col = f" {'dev peak (GiB)':>14}" if show_dev else ""
    header = f"{'mode':<26} {'median (s)':>11} {'min (s)':>10} {'peak (GiB)':>11}{dev_col} {'setup (s)':>10}  status"
    print(header)
    print("-" * 100)

    baseline_median = next((r["median_ms"] for r in results if r["mode"] == "none" and r.get("ok")), None)
    for r in results:
        label = MODE_LABELS.get(r["mode"], r["mode"])
        if r.get("ok"):
            median_s = r["median_ms"] / 1000.0
            min_s = r["min_ms"] / 1000.0
            peak_gib = r["peak_memory_mb"] / 1024.0
            setup_s = r.get("setup_s", 0.0)
            status = "ok"
            if baseline_median and r["mode"] != "none":
                status = f"ok ({r['median_ms'] / baseline_median:.2f}x baseline)"
            dev_cell = ""
            if show_dev:
                dev_mb = r.get("peak_device_mb", 0.0)
                dev_cell = f" {(dev_mb / 1024.0):>14.2f}" if dev_mb else f" {'n/a':>14}"
            print(f"{label:<26} {median_s:>11.2f} {min_s:>10.2f} {peak_gib:>11.2f}{dev_cell} {setup_s:>10.2f}  {status}")
        else:
            if r.get("oom"):
                status = "OOM (out of GPU memory)"
            elif r.get("oom") is None:
                status = "OOM? (no result; likely OOM-killed)"
            else:
                status = f"FAILED: {r.get('error', '')[:48]}"
            dev_cell = f" {'—':>14}" if show_dev else ""
            print(f"{label:<26} {'—':>11} {'—':>10} {'—':>11}{dev_cell} {'—':>10}  {status}")
    print("-" * 100)
    print("Notes: 'peak (GiB)' = worker-reported reserved VRAM (per-process); latency is wall-clock per generate().")
    if show_dev:
        print(f"       'dev peak (GiB)' = independent device-level peak via {poll_backend} (total device used, all processes).")
    if any((not r.get("ok")) and (r.get("oom") or r.get("oom") is None) for r in results):
        print("       OOM rows are expected for modes that do not fit in GPU memory (e.g. 'none' baseline).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Hidden re-entrant worker entrypoint.
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)

    # Not required at the argparse level: the re-entrant worker invocation only
    # passes ``--worker <config>`` and reads the model from that config file.
    parser.add_argument("--model", default=None, help="Diffusers model ID or local path (e.g. a Cosmos3 model).")
    parser.add_argument("--model-class-name", default=None, help="Override the pipeline class name.")
    parser.add_argument("--label", default=None, help="Free-form label for the report (e.g. 'cosmos3-nano @ RTX5090').")
    parser.add_argument(
        "--modes",
        default="layerwise,model,flat",
        help="Comma-separated offload modes to compare. Choices: none, layerwise, model, flat.",
    )

    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--extra-body",
        type=_json_obj,
        default=None,
        help='JSON dict of model-specific knobs routed to sampling extra_args (e.g. \'{"modalities": ["image"]}\').',
    )

    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale. Default: model-specific.")
    parser.add_argument("--guidance-scale-high", type=float, default=None, help="Separate high-noise CFG (Wan2.2).")
    parser.add_argument("--boundary-ratio", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["fp8", "mxfp8", "mxfp4", "mxfp4_dualscale", "int8", "gguf"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup generations (discarded).")
    parser.add_argument("--iters", type=int, default=2, help="Timed generations per mode.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile (eager only).")
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument(
        "--gpu-poll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cross-check peak VRAM with a device-level NVML/nvidia-smi sampler "
        "(reports an independent 'dev peak' column). Use --no-gpu-poll to disable.",
    )
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="(Cosmos3) Disable safety guardrails (sets model_config['guardrails']=False). "
        "Note: disabling may violate the NVIDIA Open Model License; off by default.",
    )
    parser.add_argument(
        "--offload-guardrails",
        action="store_true",
        help="(Cosmos3) Keep guardrail models on CPU and move them to GPU only per-call "
        "(sets model_config['offload_guardrail_models']=True), so they don't inflate peak GPU memory.",
    )
    parser.add_argument("--output-json", default=None, help="Write full results (config + per-mode metrics) here.")
    return parser.parse_args()


def _json_obj(value: str) -> dict[str, Any]:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return obj


def main() -> None:
    args = parse_args()

    if args.worker:
        _run_worker(args.worker)
        return

    if not args.model:
        raise SystemExit("--model is required (e.g. --model nvidia/Cosmos3-Nano)")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODE_TO_OMNI_KWARGS]
    if unknown:
        raise SystemExit(f"Unknown mode(s): {unknown}. Valid: {list(MODE_TO_OMNI_KWARGS)}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bench_offload_") as tmp:
        workdir = Path(tmp)
        for mode in modes:
            results.append(_run_mode_subprocess(args, mode, workdir))

    _print_table(args, results)

    if args.output_json:
        payload = {
            "label": args.label,
            "model": args.model,
            "config": {
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "warmup": args.warmup,
                "iters": args.iters,
                "enforce_eager": args.enforce_eager,
                "no_guardrails": args.no_guardrails,
                "offload_guardrails": args.offload_guardrails,
                "gpu_poll": args.gpu_poll,
            },
            "results": results,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.output_json}")


if __name__ == "__main__":
    main()
