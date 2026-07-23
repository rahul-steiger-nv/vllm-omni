# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
    CutlassNvFp4LinearKernel,
)

from vllm_omni.diffusion.worker.diffusion_worker import (
    _force_qutlass_nvfp4_linear_kernel,
)
from vllm_omni.quantization.qutlass_nvfp4 import (
    QutlassNvFp4LinearKernel,
    QutlassNvFp4Options,
    get_qutlass_transform,
    parse_qutlass_nvfp4_options,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_qutlass_nvfp4_options_are_default_off() -> None:
    assert parse_qutlass_nvfp4_options(None) is None
    assert parse_qutlass_nvfp4_options({}) is None
    assert parse_qutlass_nvfp4_options({"qutlass_nvfp4": False}) is None


def test_qutlass_nvfp4_options_validate_values() -> None:
    options = parse_qutlass_nvfp4_options(
        {
            "qutlass_nvfp4": True,
            "qutlass_nvfp4_transform": "identity",
            "qutlass_nvfp4_block_size": 32,
            "qutlass_nvfp4_seed": 7,
        }
    )
    assert options == QutlassNvFp4Options(
        transform="identity",
        block_size=32,
        seed=7,
    )

    with pytest.raises(ValueError, match="transform"):
        parse_qutlass_nvfp4_options(
            {
                "qutlass_nvfp4": True,
                "qutlass_nvfp4_transform": "dense",
            }
        )
    with pytest.raises(ValueError, match="block_size"):
        parse_qutlass_nvfp4_options(
            {
                "qutlass_nvfp4": True,
                "qutlass_nvfp4_block_size": 8,
            }
        )


def test_random_hadamard_is_seeded_and_orthonormal() -> None:
    options = QutlassNvFp4Options(
        transform="random_hadamard",
        block_size=16,
        seed=17,
    )
    transform = get_qutlass_transform(
        device=torch.device("cpu"),
        dtype=torch.float32,
        options=options,
    )
    repeated = get_qutlass_transform(
        device=torch.device("cpu"),
        dtype=torch.float32,
        options=options,
    )
    other_seed = get_qutlass_transform(
        device=torch.device("cpu"),
        dtype=torch.float32,
        options=QutlassNvFp4Options(
            transform="random_hadamard",
            block_size=16,
            seed=18,
        ),
    )

    assert repeated.data_ptr() == transform.data_ptr()
    torch.testing.assert_close(transform.T @ transform, torch.eye(16))
    assert not torch.equal(transform, other_seed)


def test_qutlass_initializer_patch_is_scoped(mocker: MockerFixture) -> None:
    import vllm.model_executor.layers.quantization.modelopt as vllm_modelopt

    original_initializer = vllm_modelopt.init_nvfp4_linear_kernel
    mocker.patch.object(QutlassNvFp4LinearKernel, "is_supported", return_value=(True, None))
    quant_config = SimpleNamespace(
        LinearMethodCls=vllm_modelopt.ModelOptNvFp4LinearMethod,
        get_name=lambda: "modelopt_fp4",
    )
    additional_config = {
        "qutlass_nvfp4": True,
        "qutlass_nvfp4_transform": "identity",
    }

    with _force_qutlass_nvfp4_linear_kernel(quant_config, additional_config):
        assert vllm_modelopt.init_nvfp4_linear_kernel is not original_initializer
        kernel = vllm_modelopt.init_nvfp4_linear_kernel()
        assert isinstance(kernel, QutlassNvFp4LinearKernel)
        assert kernel.options.transform == "identity"

    assert vllm_modelopt.init_nvfp4_linear_kernel is original_initializer


def test_qutlass_initializer_patch_restores_after_error(mocker: MockerFixture) -> None:
    import vllm.model_executor.layers.quantization.modelopt as vllm_modelopt

    original_initializer = vllm_modelopt.init_nvfp4_linear_kernel
    mocker.patch.object(QutlassNvFp4LinearKernel, "is_supported", return_value=(True, None))
    quant_config = SimpleNamespace(
        LinearMethodCls=vllm_modelopt.ModelOptNvFp4LinearMethod,
        get_name=lambda: "modelopt_fp4",
    )

    with pytest.raises(RuntimeError, match="test failure"):
        with _force_qutlass_nvfp4_linear_kernel(
            quant_config,
            {"qutlass_nvfp4": True},
        ):
            raise RuntimeError("test failure")

    assert vllm_modelopt.init_nvfp4_linear_kernel is original_initializer


def test_qutlass_prepares_static_inputs_during_loading(mocker: MockerFixture) -> None:
    mocker.patch.object(
        CutlassNvFp4LinearKernel,
        "process_weights_after_loading",
    )
    kernel = object.__new__(QutlassNvFp4LinearKernel)
    kernel.options = QutlassNvFp4Options(
        transform="identity",
        block_size=16,
        seed=0,
    )
    layer = SimpleNamespace(
        input_global_scale_inv=torch.tensor([0.5, 1.0]),
        weight=torch.empty(4, 16, dtype=torch.uint8),
    )

    kernel.process_weights_after_loading(layer)

    assert layer.input_global_scale_inv.shape == (1,)
    assert layer.input_global_scale_inv.item() == 1.0
    assert layer.qutlass_transform.shape == (16, 16)
    assert layer.qutlass_transform.dtype == torch.bfloat16


def test_qutlass_apply_pads_and_restores_shape(mocker: MockerFixture) -> None:
    kernel = object.__new__(QutlassNvFp4LinearKernel)
    kernel.options = QutlassNvFp4Options(
        transform="identity",
        block_size=16,
        seed=0,
    )
    layer = SimpleNamespace(
        output_size_per_partition=4,
        weights_padding_cols=1,
        input_global_scale_inv=torch.ones(1),
        qutlass_transform=torch.eye(16, dtype=torch.bfloat16),
        weight=torch.empty(4, 16, dtype=torch.uint8),
        weight_scale=torch.empty(128, 2),
        alpha=torch.ones(1),
    )
    x = torch.randn(2, 3, 30, dtype=torch.bfloat16)
    bias = torch.randn(4, dtype=torch.bfloat16)
    packed_x = torch.empty(6, 16, dtype=torch.uint8)
    row_major_scales = torch.empty(128, 4)
    fused_quantize = mocker.patch(
        "vllm_omni.quantization.qutlass_nvfp4.fusedQuantizeNv",
        return_value=(packed_x, row_major_scales),
    )
    gemm = mocker.patch(
        "vllm_omni.quantization.qutlass_nvfp4.qutlass_nvfp4_matmul",
        return_value=torch.zeros(6, 4, dtype=torch.bfloat16),
    )

    output = kernel.apply_weights(layer, x, bias)

    assert fused_quantize.call_args.args[0].shape == (6, 32)
    forwarded_global_scale = fused_quantize.call_args.args[2]
    assert forwarded_global_scale.shape == (1,)
    assert forwarded_global_scale.item() == 1.0
    assert gemm.call_args.args[2] is row_major_scales
    assert output.shape == (2, 3, 4)
    torch.testing.assert_close(output, bias.expand_as(output))


@pytest.mark.gpu
def test_qutlass_identity_matches_cutlass_nvfp4() -> None:
    from vllm._custom_ops import fusedQuantizeNv
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )

    supported, reason = QutlassNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    device = torch.device("cuda")
    rows, out_features, in_features = 128, 256, 256
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16)
    identity = torch.eye(16, device=device, dtype=torch.bfloat16)
    global_scale = torch.ones(1, device=device, dtype=torch.float32)
    packed_weight, weight_scale = fusedQuantizeNv(
        weight,
        identity,
        global_scale,
    )

    def make_layer() -> SimpleNamespace:
        return SimpleNamespace(
            output_size_per_partition=out_features,
            input_global_scale_inv=global_scale.clone(),
            weight=packed_weight.clone(),
            weight_scale=weight_scale[:out_features, : in_features // 16].clone(),
            alpha=torch.ones(1, device=device, dtype=torch.float32),
        )

    config = NvFp4LinearLayerConfig()
    cutlass_kernel = CutlassNvFp4LinearKernel(config)
    cutlass_layer = make_layer()
    cutlass_kernel.process_weights_after_loading(cutlass_layer)

    qutlass_kernel = QutlassNvFp4LinearKernel(
        config,
        options=QutlassNvFp4Options(
            transform="identity",
            block_size=16,
            seed=0,
        ),
    )
    qutlass_layer = make_layer()
    qutlass_kernel.process_weights_after_loading(qutlass_layer)

    cutlass_output = cutlass_kernel.apply_weights(cutlass_layer, x)
    qutlass_output = qutlass_kernel.apply_weights(qutlass_layer, x)

    # CUTLASS and QuTLASS use different activation-scale calculations, so
    # individual FP4 values need not match. Identity mode must nevertheless
    # remain within normal quantization noise at the linear output.
    relative_l2 = (
        torch.linalg.vector_norm(qutlass_output.float() - cutlass_output.float())
        / torch.linalg.vector_norm(cutlass_output.float())
    )
    cosine = torch.nn.functional.cosine_similarity(
        qutlass_output.float().flatten(),
        cutlass_output.float().flatten(),
        dim=0,
    )
    assert relative_l2.item() < 0.12, f"relative L2 error was {relative_l2.item():.6f}"
    assert cosine.item() > 0.99, f"cosine similarity was {cosine.item():.6f}"
