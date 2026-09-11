# /**
#   ******************************************************************************
#   * @file        test_reaudit_quantization_cuda.py
#   * @author      Egor Izmaylov
#   * @brief       验证 QuantizeLinear CUDA oracle 的 dtype 精度协议。
#   * @details     2026.09.11  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from operator_test_context import *  # noqa: F401,F403

from tools.numerical.cuda import run_cuda_ground_truth
from tools.numerical.runner_cuda_params import build_cuda_params


def _quantize_params(division_mode, count=1):
    # target=int8, division mode, saturate, rank, axis, scale/zp counts,
    # block size, scale rank, input shape.
    return np.array(
        [1, division_mode, 1, 1, 0, 1, 1, 0, 0, count], dtype=np.int32
    ).tobytes()


def _run_quantize_cuda(x, scale, division_mode):
    x = np.asarray(x, dtype=np.float64)
    result = run_cuda_ground_truth(
        "quantize_linear",
        [x, np.asarray([scale], dtype=np.float64), np.asarray([0], dtype=np.float64)],
        params_binary=_quantize_params(division_mode, x.size),
        output_dtype=np.float64,
        target_shape=x.shape,
    )
    return result.astype(np.int8)


@pytest.mark.parametrize(
    "division_mode,expected",
    [
        (2, 30),  # explicit/default FLOAT16
        (1, 29),  # explicit FLOAT
        (0, 29),  # explicit DOUBLE
    ],
)
def test_quantize_cuda_float16_fixture_distinguishes_precision_modes(division_mode, expected):
    x = np.array([0x6DF1], dtype=np.uint16).view(np.float16).astype(np.float32)
    scale = np.array(0x5A72, dtype=np.uint16).view(np.float16).astype(np.float32).item()
    np.testing.assert_array_equal(
        _run_quantize_cuda(x, scale, division_mode),
        np.array([expected], dtype=np.int8),
    )


def test_quantize_cuda_bfloat16_fixture_rounds_quotient_before_integer_rounding():
    # Exact BFLOAT16 payloads: 49664 / 616 = 80.623..., materialized BF16 quotient is 80.5.
    x = np.array([0x4742], dtype=np.uint16).astype(np.uint32)
    x = (x << np.uint32(16)).view(np.float32)
    scale_bits = np.array([0x441A], dtype=np.uint16).astype(np.uint32)
    scale = (scale_bits << np.uint32(16)).view(np.float32).item()

    np.testing.assert_array_equal(
        _run_quantize_cuda(x, scale, 3), np.array([80], dtype=np.int8)
    )
    np.testing.assert_array_equal(
        _run_quantize_cuda(x, scale, 1), np.array([81], dtype=np.int8)
    )


@pytest.mark.parametrize(
    "scale_dtype,precision,expected_mode",
    [
        ("float16", 0, 2),
        ("bfloat16", 0, 3),
        ("float16", TensorProto.FLOAT, 1),
        ("float32", TensorProto.FLOAT16, 2),
        ("float16", TensorProto.DOUBLE, 0),
    ],
)
def test_quantize_cuda_params_encode_default_and_explicit_precision(
    scale_dtype, precision, expected_mode
):
    params = build_cuda_params(
        "quantize_linear",
        [np.zeros((1,), dtype=np.float32), np.ones((1,), dtype=np.float32), np.zeros((1,), dtype=np.int8)],
        {"axis": 0, "precision": precision},
        [(1,), (1,), (1,)],
        ["float32", scale_dtype, "int8"],
        "int8",
        np.zeros((1,), dtype=np.int8),
    )
    assert np.frombuffer(params, dtype=np.int32)[1] == expected_mode
