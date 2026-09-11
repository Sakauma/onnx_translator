# /**
#   ******************************************************************************
#   * @file        test_reaudit_quantization_precision.py
#   * @author      Egor Izmaylov
#   * @brief       固化量化算子逐阶段浮点精度与舍入边界回归。
#   * @details     2026.09.11  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import DynamicQuantizeLinear, QuantizeLinear


def _require_c_backend():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")


def _tensor(values, dtype):
    data = np.asarray(values, dtype=nn.DTYPE_TO_NUMPY[dtype])
    return Tensor(*data.shape, dtype=dtype, data=data)


# FLOAT DynamicQuantizeLinear 的 ReduceMin/Max、Sub、Div 和量化商均须逐阶段物化为 float32。
def test_dynamic_quantize_linear_float32_stage_rounding_is_bit_exact():
    _require_c_backend()

    x = np.array([-6.7799618e17, 4.5765801e17, 0.0, 7.7123871e19], dtype=np.float32)
    y, scale, zero_point = DynamicQuantizeLinear(
        ["x"], ["y", "scale", "zero_point"]
    ).forward(_tensor(x, "float32"))["tensor"]

    np.testing.assert_array_equal(y.data, np.array([0, 4, 2, 255], dtype=np.uint8))
    assert scale.data.shape == ()
    assert scale.data.view(np.uint32).item() == 0x5C877E76
    np.testing.assert_array_equal(zero_point.data, np.array(2, dtype=np.uint8))


# 常规、极小正规数与既有全零分支覆盖修复前后都应稳定的 DQL 数值边界。
@pytest.mark.parametrize(
    "x,expected_y,expected_scale,expected_zero_point",
    [
        (
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            np.array([0, 127, 254], dtype=np.uint8),
            np.float32(2.0 / 255.0),
            np.uint8(127),
        ),
        (
            np.array([-np.finfo(np.float32).tiny, 0.0, np.finfo(np.float32).tiny], dtype=np.float32),
            np.array([0, 128, 255], dtype=np.uint8),
            np.float32(2.0 * np.finfo(np.float32).tiny / np.float32(255.0)),
            np.uint8(128),
        ),
        (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.uint8),
            np.float32(1.0),
            np.uint8(0),
        ),
    ],
)
def test_dynamic_quantize_linear_float32_normal_and_boundary_paths(
    x, expected_y, expected_scale, expected_zero_point
):
    _require_c_backend()

    y, scale, zero_point = DynamicQuantizeLinear(
        ["x"], ["y", "scale", "zero_point"]
    ).forward(_tensor(x, "float32"))["tensor"]

    np.testing.assert_array_equal(y.data, expected_y)
    np.testing.assert_array_equal(scale.data, np.array(expected_scale, dtype=np.float32))
    np.testing.assert_array_equal(zero_point.data, np.array(expected_zero_point, dtype=np.uint8))


# opset 24 省略 precision 时按 scale dtype 除法；显式属性必须覆盖该默认值。
@pytest.mark.parametrize(
    "precision,expected",
    [
        (0, 30),
        (TensorProto.FLOAT16, 30),
        (TensorProto.FLOAT, 29),
        (TensorProto.DOUBLE, 29),
    ],
)
def test_quantize_linear_float16_default_and_explicit_precision(precision, expected):
    _require_c_backend()

    x = np.array([0x6DF1], dtype=np.uint16).view(np.float16)
    scale = np.array(0x5A72, dtype=np.uint16).view(np.float16)
    zero_point = np.array(0, dtype=np.int8)
    actual = QuantizeLinear(
        ["x", "scale", "zero_point"],
        ["y"],
        dtype="int8",
        precision=precision,
        version="24",
    ).forward(
        _tensor(x, "float16"),
        _tensor(scale, "float16"),
        _tensor(zero_point, "int8"),
    )["tensor"]

    np.testing.assert_array_equal(actual.data, np.array([expected], dtype=np.int8))


# 正负半整数使用 ties-to-even，越界值仍由目标整数 dtype 饱和。
def test_quantize_linear_float16_ties_even_and_saturation_boundaries():
    _require_c_backend()

    x = np.array([-300.0, -2.5, -1.5, 1.5, 2.5, 300.0], dtype=np.float16)
    scale = np.array(1.0, dtype=np.float16)
    zero_point = np.array(0, dtype=np.int8)
    actual = QuantizeLinear(
        ["x", "scale", "zero_point"], ["y"], dtype="int8", version="24"
    ).forward(
        _tensor(x, "float16"),
        _tensor(scale, "float16"),
        _tensor(zero_point, "int8"),
    )["tensor"]

    np.testing.assert_array_equal(
        actual.data, np.array([-128, -2, -2, 2, 2, 127], dtype=np.int8)
    )


# 显式低精度要求先把两个操作数转换到该 dtype，再进行同精度除法。
def test_quantize_linear_explicit_float16_converts_both_operands_before_division():
    _require_c_backend()

    x = np.array([3435.3032], dtype=np.float32)
    scale = np.array(981.6252, dtype=np.float32)
    zero_point = np.array(0, dtype=np.int8)
    actual = QuantizeLinear(
        ["x", "scale", "zero_point"],
        ["y"],
        dtype="int8",
        precision=TensorProto.FLOAT16,
        version="24",
    ).forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]

    float32_quotient = np.divide(x, scale, dtype=np.float32)
    half_quotient = np.divide(x.astype(np.float16), scale.astype(np.float16), dtype=np.float16)
    np.testing.assert_array_equal(np.rint(float32_quotient), np.array([3.0], dtype=np.float32))
    np.testing.assert_array_equal(half_quotient, np.array([3.5], dtype=np.float16))
    np.testing.assert_array_equal(actual.data, np.array([4], dtype=np.int8))


# half min-subnormal midpoint 必须 ties-to-even 到零，越过 midpoint 的相邻 float32 则舍入到 bits 1。
def test_quantize_linear_explicit_float16_preserves_min_subnormal_rounding_boundary():
    _require_c_backend()

    midpoint = np.float32(2.0**-25)
    above = np.nextafter(midpoint, np.float32(np.inf), dtype=np.float32)
    x = np.array([midpoint, above, -midpoint, -above], dtype=np.float32)
    scale = np.array(2.0**-24, dtype=np.float32)
    zero_point = np.array(0, dtype=np.int8)
    actual = QuantizeLinear(
        ["x", "scale", "zero_point"],
        ["y"],
        dtype="int8",
        precision=TensorProto.FLOAT16,
        version="24",
    ).forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]

    converted_bits = x.astype(np.float16).view(np.uint16)
    np.testing.assert_array_equal(
        converted_bits, np.array([0x0000, 0x0001, 0x8000, 0x8001], dtype=np.uint16)
    )
    np.testing.assert_array_equal(actual.data, np.array([0, 1, 0, -1], dtype=np.int8))
