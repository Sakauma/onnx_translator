# /**
#   ******************************************************************************
#   * @file        test_operator_c_backend.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 C 后端数值路径、量化路径和池化卷积类回归。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


# 将 float32 数值转换为 bfloat16 的 uint16 位模式，匹配 Tensor 内部存储。
def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# 将 bfloat16 的 uint16 位模式解码成 float32，便于断言低精度输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def test_c_backend_max_min_propagate_nan_like_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.array([np.nan, 1.0, np.nan, -2.0], dtype=np.float32)
    right = np.array([2.0, np.nan, np.nan, -3.0], dtype=np.float32)

    max_out = Max(["a", "b"], ["y"], dtype="float32").forward(
        Tensor(4, dtype="float32", data=left),
        Tensor(4, dtype="float32", data=right),
    )["tensor"].data
    min_out = Min(["a", "b"], ["y"], dtype="float32").forward(
        Tensor(4, dtype="float32", data=left),
        Tensor(4, dtype="float32", data=right),
    )["tensor"].data

    assert np.all(np.isnan(max_out[:3]))
    assert np.all(np.isnan(min_out[:3]))
    assert max_out[3] == np.float32(-2.0)
    assert min_out[3] == np.float32(-3.0)

    bf16_max = Max(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(4, dtype="bfloat16", data=_bf16_bits(left)),
        Tensor(4, dtype="bfloat16", data=_bf16_bits(right)),
    )["tensor"]
    bf16_min = Min(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(4, dtype="bfloat16", data=_bf16_bits(left)),
        Tensor(4, dtype="bfloat16", data=_bf16_bits(right)),
    )["tensor"]

    decoded_max = _bf16_to_float32(bf16_max.data)
    decoded_min = _bf16_to_float32(bf16_min.data)
    assert np.all(np.isnan(decoded_max[:3]))
    assert np.all(np.isnan(decoded_min[:3]))
    assert decoded_max[3] == np.float32(-2.0)
    assert decoded_min[3] == np.float32(-3.0)


def test_c_backend_einsum_ellipsis_uses_stride_planner(monkeypatch):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left_data = (np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4) + 1.0) / 10.0
    right_data = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) - 3.0) / 7.0
    expected = np.einsum("...ij,...jk->...ik", left_data, right_data).astype(np.float32)

    def fail_einsum(*_args, **_kwargs):
        raise AssertionError("Einsum ellipsis path should be handled by the C stride planner")

    monkeypatch.setattr(np, "einsum", fail_einsum)
    actual = Einsum(["left", "right"], ["out"], equation="...ij,...jk->...ik", dtype="float32").forward(
        Tensor(*left_data.shape, dtype="float32", data=left_data),
        Tensor(*right_data.shape, dtype="float32", data=right_data),
    )["tensor"]

    assert actual.size == expected.shape
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


def test_global_pooling_and_dropout_optional_mask(monkeypatch):
    _disable_c_backend(monkeypatch)

    data = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    x = Tensor(*data.shape, dtype="float32", data=data)

    avg = GlobalAveragePool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_array_equal(avg.data, np.mean(data, axis=(2, 3, 4), keepdims=True))
    assert avg.size == (2, 3, 1, 1, 1)

    max_pool = GlobalMaxPool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_array_equal(max_pool.data, np.max(data, axis=(2, 3, 4), keepdims=True))

    lp = GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(lp.data, np.sum(np.abs(data) ** 2, axis=(2, 3, 4), keepdims=True) ** 0.5)
    assert GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward_(Tensor_(2, 3, 2, 2, 2, dtype="float32"))["tensor"].size == (2, 3, 1, 1, 1)

    drop_input = Tensor(4, dtype="float32", data=np.ones(4, dtype=np.float32))
    dropout = Dropout(["x"], ["y", "mask"], seed=123, ratio=0.5, training_mode=1)
    y, mask = dropout.forward(drop_input)["tensor"]
    assert y.size == (4,)
    assert mask.size == (4,)
    assert mask.dtype == "bool"
    np.testing.assert_array_equal(y.data, mask.data.astype(np.float32) * 2.0)

    seeded_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    seeded_dropout = Dropout(["x"], ["y", "mask"], seed=0, ratio=0.5, training_mode=1).forward(
        Tensor(*seeded_data.shape, dtype="float32", data=seeded_data)
    )["tensor"]
    np.random.seed(0)
    expected_mask = np.random.uniform(0.0, 1.0, seeded_data.shape) >= 0.5
    np.testing.assert_array_equal(seeded_dropout[1].data, expected_mask)
    np.testing.assert_allclose(seeded_dropout[0].data, seeded_data * expected_mask.astype(np.float32) * 2.0)

    inferred_y, inferred_mask = dropout.forward_(Tensor_(4, dtype="float32"))["tensor"]
    assert inferred_y.size == (4,)
    assert inferred_mask.dtype == "bool"


def test_python_pooling_fallback_bfloat16_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    values = np.array([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]], dtype=np.float32)
    bf16_x = Tensor(*values.shape, dtype="bfloat16", data=_bf16_bits(values))

    max_y, max_indices = MaxPool(
        ["x"],
        ["y", "idx"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 2],
        dtype="bfloat16",
    ).forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(max_y.data), np.array([[[[6.0, 8.0]]]], dtype=np.float32), rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(max_indices.data, np.array([[[[5, 7]]]], dtype=np.int64))

    avg = AveragePool(
        ["x"],
        ["y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 2],
        dtype="bfloat16",
    ).forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(avg.data), np.array([[[[3.5, 5.5]]]], dtype=np.float32), rtol=1e-2, atol=1e-2)

    lp = LpPool(
        ["x"],
        ["y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 2],
        p=2,
        dtype="bfloat16",
    ).forward(bf16_x)["tensor"]
    expected_lp = np.array([[[[np.sqrt(66.0), np.sqrt(138.0)]]]], dtype=np.float32)
    np.testing.assert_allclose(_bf16_to_float32(lp.data), expected_lp, rtol=1e-2, atol=1e-2)

    global_avg = GlobalAveragePool(["x"], ["y"], dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(global_avg.data), np.mean(values, axis=(2, 3), keepdims=True), rtol=1e-2, atol=1e-2)

    global_max = GlobalMaxPool(["x"], ["y"], dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(global_max.data), np.max(values, axis=(2, 3), keepdims=True), rtol=1e-2, atol=1e-2)

    global_lp = GlobalLpPool(["x"], ["y"], p=2, dtype="bfloat16").forward(bf16_x)["tensor"]
    expected_global_lp = np.sum(np.abs(values) ** 2, axis=(2, 3), keepdims=True) ** 0.5
    np.testing.assert_allclose(_bf16_to_float32(global_lp.data), expected_global_lp, rtol=1e-2, atol=1e-2)


def test_python_matrix_and_conv_fallback_bfloat16_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    right = np.array([[5.0], [6.0]], dtype=np.float32)
    matmul = MatMul(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(*left.shape, dtype="bfloat16", data=_bf16_bits(left)),
        Tensor(*right.shape, dtype="bfloat16", data=_bf16_bits(right)),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(matmul.data), np.matmul(left, right), rtol=1e-2, atol=1e-2)

    einsum = Einsum(["a", "b"], ["y"], equation="ij,jk->ik", dtype="bfloat16").forward(
        Tensor(*left.shape, dtype="bfloat16", data=_bf16_bits(left)),
        Tensor(*right.shape, dtype="bfloat16", data=_bf16_bits(right)),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(einsum.data), np.einsum("ij,jk->ik", left, right), rtol=1e-2, atol=1e-2)

    max_left = np.array([[-2.0, 1.0], [3.0, -4.0]], dtype=np.float32)
    max_right = np.array([[0.5, -3.0]], dtype=np.float32)
    max_out = Max(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(*max_left.shape, dtype="bfloat16", data=_bf16_bits(max_left)),
        Tensor(*max_right.shape, dtype="bfloat16", data=_bf16_bits(max_right)),
    )["tensor"]
    min_out = Min(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(*max_left.shape, dtype="bfloat16", data=_bf16_bits(max_left)),
        Tensor(*max_right.shape, dtype="bfloat16", data=_bf16_bits(max_right)),
    )["tensor"]
    np.testing.assert_array_equal(max_out.data, _bf16_bits(np.maximum(max_left, max_right)))
    np.testing.assert_array_equal(min_out.data, _bf16_bits(np.minimum(max_left, max_right)))

    clip_values = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    clipped = Clip(["x", "min", "max"], ["y"], dtype="bfloat16").forward(
        Tensor(*clip_values.shape, dtype="bfloat16", data=_bf16_bits(clip_values)),
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([-1.0], dtype=np.float32))),
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([1.0], dtype=np.float32))),
    )["tensor"]
    np.testing.assert_array_equal(clipped.data, _bf16_bits(np.clip(clip_values, -1.0, 1.0)))

    qa = Tensor(2, 2, dtype="uint8", data=np.array([[3, 5], [7, 9]], dtype=np.uint8))
    qb = Tensor(2, 2, dtype="uint8", data=np.array([[4, 6], [8, 10]], dtype=np.uint8))
    qa_zp = Tensor(2, dtype="uint8", data=np.array([1, 2], dtype=np.uint8))
    qb_zp = Tensor(2, dtype="uint8", data=np.array([2, 4], dtype=np.uint8))
    qa_scale_values = np.array([0.5, 0.25], dtype=np.float32)
    qb_scale_values = np.array([0.2, 0.4], dtype=np.float32)
    qy_scale_value = np.array([0.1], dtype=np.float32)
    qy_zp = Tensor(1, dtype="uint8", data=np.array([10], dtype=np.uint8))
    qmatmul = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        qa,
        Tensor(2, dtype="bfloat16", data=_bf16_bits(qa_scale_values)),
        qa_zp,
        qb,
        Tensor(2, dtype="bfloat16", data=_bf16_bits(qb_scale_values)),
        qb_zp,
        Tensor(1, dtype="bfloat16", data=_bf16_bits(qy_scale_value)),
        qy_zp,
    )["tensor"]
    qa_real = (qa.data.astype(np.float64) - qa_zp.data.astype(np.float64).reshape(2, 1)) * _bf16_to_float32(_bf16_bits(qa_scale_values)).reshape(2, 1)
    qb_real = (qb.data.astype(np.float64) - qb_zp.data.astype(np.float64).reshape(1, 2)) * _bf16_to_float32(_bf16_bits(qb_scale_values)).reshape(1, 2)
    expected_qmatmul = np.rint(np.matmul(qa_real, qb_real) / _bf16_to_float32(_bf16_bits(qy_scale_value)).item() + qy_zp.data.item())
    np.testing.assert_array_equal(qmatmul.data, np.clip(expected_qmatmul, 0, 255).astype(np.uint8))

    dequant = DequantizeLinear(["x", "scale", "zp"], ["y"], dtype="bfloat16").forward(
        Tensor(2, dtype="uint8", data=np.array([3, 5], dtype=np.uint8)),
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([0.5], dtype=np.float32))),
        Tensor(1, dtype="uint8", data=np.array([1], dtype=np.uint8)),
    )["tensor"]
    np.testing.assert_array_equal(dequant.data, _bf16_bits(np.array([1.0, 2.0], dtype=np.float32)))

    prelu_input = np.array([-2.0, -1.0, 0.0, 3.0], dtype=np.float32)
    slope = np.array([0.25], dtype=np.float32)
    prelu = PRelu(["x", "slope"], ["y"], dtype="bfloat16").forward(
        Tensor(*prelu_input.shape, dtype="bfloat16", data=_bf16_bits(prelu_input)),
        Tensor(*slope.shape, dtype="bfloat16", data=_bf16_bits(slope)),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(prelu.data), np.where(prelu_input >= 0, prelu_input, prelu_input * slope), rtol=1e-2, atol=1e-2)

    det_input = np.array([[[1.0, 2.0], [3.0, 5.0]]], dtype=np.float32)
    det = Det(["x"], ["y"], dtype="bfloat16").forward(
        Tensor(*det_input.shape, dtype="bfloat16", data=_bf16_bits(det_input))
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(det.data), np.linalg.det(det_input), rtol=1e-2, atol=1e-2)

    conv_x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    conv_w = np.array([[[[2.0]]]], dtype=np.float32)
    conv_b = np.array([0.5], dtype=np.float32)
    conv_t = ConvTranspose(["x", "w", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(*conv_x.shape, dtype="bfloat16", data=_bf16_bits(conv_x)),
        Tensor(*conv_w.shape, dtype="bfloat16", data=_bf16_bits(conv_w)),
        Tensor(*conv_b.shape, dtype="bfloat16", data=_bf16_bits(conv_b)),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(conv_t.data), conv_x * conv_w.reshape(1, 1, 1, 1) + conv_b.reshape(1, 1, 1, 1), rtol=1e-2, atol=1e-2)

    conv_1d_x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    conv_1d_w = np.array([[[0.5, -1.0]]], dtype=np.float32)
    conv_1d_b = np.array([0.25], dtype=np.float32)
    conv = Conv(
        ["x", "w", "b"],
        ["y"],
        pads=[0, 0],
        strides=[1],
        dilations=[1],
        group=1,
        dtype="bfloat16",
    ).forward(
        Tensor(*conv_1d_x.shape, dtype="bfloat16", data=_bf16_bits(conv_1d_x)),
        Tensor(*conv_1d_w.shape, dtype="bfloat16", data=_bf16_bits(conv_1d_w)),
        Tensor(*conv_1d_b.shape, dtype="bfloat16", data=_bf16_bits(conv_1d_b)),
    )["tensor"]
    expected_conv = np.array([[[-1.25, -1.75, -2.25]]], dtype=np.float32)
    np.testing.assert_allclose(_bf16_to_float32(conv.data), expected_conv, rtol=2e-2, atol=2e-2)

    qconv_x = Tensor(1, 1, 1, 2, dtype="uint8", data=np.array([[[[3, 5]]]], dtype=np.uint8))
    qconv_w = Tensor(1, 1, 1, 1, dtype="uint8", data=np.array([[[[4]]]], dtype=np.uint8))
    qconv = QLinearConv(
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"],
        ["y"],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dtype="uint8",
    ).forward(
        qconv_x,
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([0.5], dtype=np.float32))),
        Tensor(1, dtype="uint8", data=np.array([1], dtype=np.uint8)),
        qconv_w,
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([0.25], dtype=np.float32))),
        Tensor(1, dtype="uint8", data=np.array([2], dtype=np.uint8)),
        Tensor(1, dtype="bfloat16", data=_bf16_bits(np.array([0.125], dtype=np.float32))),
        Tensor(1, dtype="uint8", data=np.array([10], dtype=np.uint8)),
    )["tensor"]
    np.testing.assert_array_equal(qconv.data, np.array([[[[14, 18]]]], dtype=np.uint8))


def test_c_backend_unsigned_integer_dtype_cast_paths(monkeypatch):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int_values = np.array([-1, 0, 1, 2**32 - 1], dtype=np.int64)
    expected_u32 = int_values.astype(np.uint32)
    uint64_values = np.array([0, 2**63, 2**64 - 1], dtype=np.uint64)
    expected_i64 = uint64_values.astype(np.int64)
    float_values = np.array([-1.2, 1.9, 65536.0], dtype=np.float64)
    expected_u16 = float_values.astype(np.uint16)

    def fail_python_cast_fallback(*_args, **_kwargs):
        raise AssertionError("unsigned dtype Cast should be handled by the C backend")

    monkeypatch.setattr(np, "asarray", fail_python_cast_fallback)

    cast_u32 = Cast(["x"], ["y"], dtype="uint32").forward(Tensor(4, dtype="int64", data=int_values))["tensor"]
    assert cast_u32.dtype == "uint32"
    assert cast_u32.data.dtype == np.uint32
    np.testing.assert_array_equal(cast_u32.data, expected_u32)

    cast_i64 = Cast(["x"], ["y"], dtype="int64").forward(Tensor(3, dtype="uint64", data=uint64_values))["tensor"]
    assert cast_i64.dtype == "int64"
    np.testing.assert_array_equal(cast_i64.data, expected_i64)

    cast_u16 = Cast(["x"], ["y"], dtype="uint16").forward(Tensor(3, dtype="float64", data=float_values))["tensor"]
    assert cast_u16.dtype == "uint16"
    assert cast_u16.data.dtype == np.uint16
    np.testing.assert_array_equal(cast_u16.data, expected_u16)


def test_c_backend_bool_dtype_uses_boolean_semantics():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([-2.0, -0.1, 0.0, 0.5, np.nan], dtype=np.float32)
    cast_bool = Cast(["x"], ["y"], dtype="bool").forward(Tensor(5, dtype="float32", data=values))["tensor"]
    assert cast_bool.dtype == "bool"
    assert cast_bool.data.dtype == np.bool_
    np.testing.assert_array_equal(cast_bool.data, values.astype(np.bool_))

    left = Tensor(4, dtype="int32", data=np.array([1, 2, 3, 4], dtype=np.int32))
    right = Tensor(4, dtype="int32", data=np.array([1, 0, 3, 5], dtype=np.int32))
    equal = Equal(["a", "b"], ["y"]).forward(left, right)["tensor"]
    assert equal.dtype == "bool"
    assert equal.data.dtype == np.bool_
    np.testing.assert_array_equal(equal.data, np.array([True, False, True, False], dtype=np.bool_))

    logical_not = Not(["x"], ["y"]).forward(equal)["tensor"]
    np.testing.assert_array_equal(logical_not.data, np.logical_not(equal.data))

    logical_and = And(["a", "b"], ["y"]).forward(
        equal,
        Tensor(4, dtype="bool", data=np.array([True, True, False, False], dtype=np.bool_)),
    )["tensor"]
    np.testing.assert_array_equal(logical_and.data, np.array([True, False, False, False], dtype=np.bool_))


def test_c_backend_signed_integer_binary_ops_wrap_like_numpy():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int8_left = np.array([127, -128, 100, -1], dtype=np.int8)
    int8_right = np.array([1, -1, 2, 1], dtype=np.int8)
    left_tensor = Tensor(4, dtype="int8", data=int8_left)
    right_tensor = Tensor(4, dtype="int8", data=int8_right)

    add = ADD(["a", "b"], ["y"], dtype="int8").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(add.data, np.add(int8_left, int8_right))

    sub = SUB(["a", "b"], ["y"], dtype="int8").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(sub.data, np.subtract(int8_left, int8_right))

    mul = MUL(["a", "b"], ["y"], dtype="int8").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(mul.data, np.multiply(int8_left, int8_right))

    max_out = Max(["a", "b"], ["y"], dtype="int8").forward(left_tensor, right_tensor)["tensor"]
    min_out = Min(["a", "b"], ["y"], dtype="int8").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(max_out.data, np.maximum(int8_left, int8_right))
    np.testing.assert_array_equal(min_out.data, np.minimum(int8_left, int8_right))

    int32_left = np.array([2**30, -1, -8], dtype=np.int32)
    int32_right = np.array([4, 1, 1], dtype=np.int32)
    i32_l = Tensor(3, dtype="int32", data=int32_left)
    i32_r = Tensor(3, dtype="int32", data=int32_right)
    mul_i32 = MUL(["a", "b"], ["y"], dtype="int32").forward(i32_l, i32_r)["tensor"]
    np.testing.assert_array_equal(mul_i32.data, np.multiply(int32_left, int32_right))

    shift_left = BitShift(["a", "b"], ["y"], direction="LEFT", dtype="int32").forward(i32_l, i32_r)["tensor"]
    np.testing.assert_array_equal(shift_left.data, np.left_shift(int32_left, int32_right))


def test_c_backend_signed_integer_unary_ops_wrap_like_numpy():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int8_values = np.array([-128, -2, 0, 3, 127], dtype=np.int8)
    int8_tensor = Tensor(5, dtype="int8", data=int8_values)
    neg_i8 = Neg(["x"], ["y"], dtype="int8").forward(int8_tensor)["tensor"]
    abs_i8 = ABS(["x"], ["y"], dtype="int8").forward(int8_tensor)["tensor"]
    np.testing.assert_array_equal(neg_i8.data, np.negative(int8_values))
    np.testing.assert_array_equal(abs_i8.data, np.abs(int8_values))

    int16_values = np.array([-32768, -3, 0, 9, 32767], dtype=np.int16)
    int16_tensor = Tensor(5, dtype="int16", data=int16_values)
    neg_i16 = Neg(["x"], ["y"], dtype="int16").forward(int16_tensor)["tensor"]
    abs_i16 = ABS(["x"], ["y"], dtype="int16").forward(int16_tensor)["tensor"]
    np.testing.assert_array_equal(neg_i16.data, np.negative(int16_values))
    np.testing.assert_array_equal(abs_i16.data, np.abs(int16_values))


def test_c_backend_range_preserves_large_int64_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    start = np.array(2**60, dtype=np.int64)
    limit = np.array(2**60 + 4, dtype=np.int64)
    delta = np.array(1, dtype=np.int64)
    actual = Range(["start", "limit", "delta"], ["y"], dtype="int64").forward(
        Tensor(dtype="int64", data=start),
        Tensor(dtype="int64", data=limit),
        Tensor(dtype="int64", data=delta),
    )["tensor"]
    np.testing.assert_array_equal(actual.data, np.arange(start, limit, delta, dtype=np.int64))

    neg_start = np.array(2**60 + 4, dtype=np.int64)
    neg_limit = np.array(2**60, dtype=np.int64)
    neg_delta = np.array(-1, dtype=np.int64)
    actual_desc = Range(["start", "limit", "delta"], ["y"], dtype="int64").forward(
        Tensor(dtype="int64", data=neg_start),
        Tensor(dtype="int64", data=neg_limit),
        Tensor(dtype="int64", data=neg_delta),
    )["tensor"]
    np.testing.assert_array_equal(actual_desc.data, np.arange(neg_start, neg_limit, neg_delta, dtype=np.int64))


def test_c_backend_scatter_integer_paths_preserve_dtype_semantics():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([0, 0, 0], dtype=np.int64)
    indices = np.array([[1], [2]], dtype=np.int64)
    updates = np.array([2**60 + 1, 2**60 + 2], dtype=np.int64)
    scatter_nd = ScatterND(["data", "indices", "updates"], ["y"], dtype="int64").forward(
        Tensor(3, dtype="int64", data=data),
        Tensor(*indices.shape, dtype="int64", data=indices),
        Tensor(2, dtype="int64", data=updates),
    )["tensor"]
    expected_nd = data.copy()
    expected_nd[1] = updates[0]
    expected_nd[2] = updates[1]
    np.testing.assert_array_equal(scatter_nd.data, expected_nd)

    i8_data = np.array([127, -128, 10], dtype=np.int8)
    i8_indices = np.array([0, 1, 2], dtype=np.int64)
    i8_updates = np.array([1, -1, 120], dtype=np.int8)
    scatter_add = ScatterElements(["data", "indices", "updates"], ["y"], axis=0, reduction="add", dtype="int8").forward(
        Tensor(3, dtype="int8", data=i8_data),
        Tensor(3, dtype="int64", data=i8_indices),
        Tensor(3, dtype="int8", data=i8_updates),
    )["tensor"]
    np.testing.assert_array_equal(scatter_add.data, np.add(i8_data, i8_updates))

    u32_data = np.array([0xFFFFFFFF, 2], dtype=np.uint32)
    u32_updates = np.array([2, 0x80000000], dtype=np.uint32)
    scatter_mul = ScatterElements(["data", "indices", "updates"], ["y"], axis=0, reduction="mul", dtype="uint32").forward(
        Tensor(2, dtype="uint32", data=u32_data),
        Tensor(2, dtype="int64", data=np.array([0, 1], dtype=np.int64)),
        Tensor(2, dtype="uint32", data=u32_updates),
    )["tensor"]
    np.testing.assert_array_equal(scatter_mul.data, np.multiply(u32_data, u32_updates))


def test_c_backend_matmul_integer_dtypes_preserve_wrap_and_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int64_a = np.array([[2**60 + 1, 1]], dtype=np.int64)
    int64_b = np.array([[1, 1], [1, -1]], dtype=np.int64)
    int64_out = MatMul(["a", "b"], ["y"], dtype="int64").forward(
        Tensor(*int64_a.shape, dtype="int64", data=int64_a),
        Tensor(*int64_b.shape, dtype="int64", data=int64_b),
    )["tensor"]
    np.testing.assert_array_equal(int64_out.data, np.matmul(int64_a, int64_b))

    int64_overflow_a = np.array([[np.iinfo(np.int64).max, 2]], dtype=np.int64)
    int64_overflow_b = np.array([[2], [3]], dtype=np.int64)
    int64_overflow = MatMul(["a", "b"], ["y"], dtype="int64").forward(
        Tensor(*int64_overflow_a.shape, dtype="int64", data=int64_overflow_a),
        Tensor(*int64_overflow_b.shape, dtype="int64", data=int64_overflow_b),
    )["tensor"]
    np.testing.assert_array_equal(int64_overflow.data, np.matmul(int64_overflow_a, int64_overflow_b))

    uint32_a = np.array([[0xFFFFFFFF, 2]], dtype=np.uint32)
    uint32_b = np.array([[2], [0x80000000]], dtype=np.uint32)
    uint32_out = MatMul(["a", "b"], ["y"], dtype="uint32").forward(
        Tensor(*uint32_a.shape, dtype="uint32", data=uint32_a),
        Tensor(*uint32_b.shape, dtype="uint32", data=uint32_b),
    )["tensor"]
    np.testing.assert_array_equal(uint32_out.data, np.matmul(uint32_a, uint32_b))

    uint64_a = np.array([[2**63 + 1, 3]], dtype=np.uint64)
    uint64_b = np.array([[1], [2]], dtype=np.uint64)
    uint64_out = MatMul(["a", "b"], ["y"], dtype="uint64").forward(
        Tensor(*uint64_a.shape, dtype="uint64", data=uint64_a),
        Tensor(*uint64_b.shape, dtype="uint64", data=uint64_b),
    )["tensor"]
    np.testing.assert_array_equal(uint64_out.data, np.matmul(uint64_a, uint64_b))


def test_c_backend_clip_integer_dtypes_preserve_large_values():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int64_values = np.array([2**60 + 1, 2**60 + 3, 2**60 + 7, -(2**60 + 3)], dtype=np.int64)
    int64_min = np.array([2**60 + 2, 2**60 + 2, 2**60 + 2, -(2**60 + 2)], dtype=np.int64)
    int64_max = np.array([2**60 + 4, 2**60 + 4, 2**60 + 4, -(2**60)], dtype=np.int64)
    int64_out = Clip(["x", "min", "max"], ["y"], dtype="int64").forward(
        Tensor(4, dtype="int64", data=int64_values),
        Tensor(4, dtype="int64", data=int64_min),
        Tensor(4, dtype="int64", data=int64_max),
    )["tensor"]
    np.testing.assert_array_equal(int64_out.data, np.clip(int64_values, int64_min, int64_max))

    uint64_values = np.array([2**63 + 1, 2**63 + 3, 2**63 + 7, 2**64 - 1], dtype=np.uint64)
    uint64_min = np.array([2**63 + 2], dtype=np.uint64)
    uint64_max = np.array([2**63 + 4], dtype=np.uint64)
    uint64_out = Clip(["x", "min", "max"], ["y"], dtype="uint64").forward(
        Tensor(4, dtype="uint64", data=uint64_values),
        Tensor(1, dtype="uint64", data=uint64_min),
        Tensor(1, dtype="uint64", data=uint64_max),
    )["tensor"]
    np.testing.assert_array_equal(uint64_out.data, np.clip(uint64_values, uint64_min, uint64_max))


def test_c_backend_mod_integer_dtypes_match_python_remainder_semantics():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int64_a = np.array([-(2**60 + 5), 2**60 + 5, -(2**60 + 5), 2**60 + 5, np.iinfo(np.int64).min, 5], dtype=np.int64)
    int64_b = np.array([3, -3, -3, 3, -1, 0], dtype=np.int64)
    int64_out = Mod(["a", "b"], ["y"], dtype="int64").forward(
        Tensor(6, dtype="int64", data=int64_a),
        Tensor(6, dtype="int64", data=int64_b),
    )["tensor"]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_int64 = np.mod(int64_a, int64_b)
    np.testing.assert_array_equal(int64_out.data, expected_int64)

    uint64_a = np.array([2**63 + 5, 2**64 - 1, 5], dtype=np.uint64)
    uint64_b = np.array([3, 2**63 + 7, 0], dtype=np.uint64)
    uint64_out = Mod(["a", "b"], ["y"], dtype="uint64").forward(
        Tensor(3, dtype="uint64", data=uint64_a),
        Tensor(3, dtype="uint64", data=uint64_b),
    )["tensor"]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_uint64 = np.mod(uint64_a, uint64_b)
    np.testing.assert_array_equal(uint64_out.data, expected_uint64)


def test_c_backend_topk_integer_dtypes_sort_without_float_precision_loss():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    k = Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64))

    uint64_values = np.array([2**63 + 1, 2**63 + 3, 2**63 + 2, 7], dtype=np.uint64)
    uint64_top = TopK(["x", "k"], ["values", "indices"], axis=-1, largest=1, sorted=1, dtype="uint64").forward(
        Tensor(4, dtype="uint64", data=uint64_values),
        k,
    )["tensor"]
    np.testing.assert_array_equal(uint64_top[0].data, np.array([2**63 + 3, 2**63 + 2], dtype=np.uint64))
    np.testing.assert_array_equal(uint64_top[1].data, np.array([1, 2], dtype=np.int64))

    int64_values = np.array([2**60 + 1, 2**60 + 3, 2**60 + 2, -(2**60 + 4)], dtype=np.int64)
    int64_bottom = TopK(["x", "k"], ["values", "indices"], axis=-1, largest=0, sorted=1, dtype="int64").forward(
        Tensor(4, dtype="int64", data=int64_values),
        k,
    )["tensor"]
    np.testing.assert_array_equal(int64_bottom[0].data, np.array([-(2**60 + 4), 2**60 + 1], dtype=np.int64))
    np.testing.assert_array_equal(int64_bottom[1].data, np.array([3, 0], dtype=np.int64))


# 验证 TopK 的 Python fallback 在 bfloat16 下按数值排序，而不是按 uint16 位模式排序。
def test_python_topk_fallback_bfloat16_uses_numeric_order(monkeypatch):
    _disable_c_backend(monkeypatch)

    values = np.array([-1.0, 0.5, -2.0, 3.0], dtype=np.float32)
    k = Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64))
    top_values, top_indices = TopK(["x", "k"], ["values", "indices"], axis=-1, largest=1, sorted=1, dtype="bfloat16").forward(
        Tensor(4, dtype="bfloat16", data=_bf16_bits(values)),
        k,
    )["tensor"]
    np.testing.assert_array_equal(top_values.data, _bf16_bits(np.array([3.0, 0.5], dtype=np.float32)))
    np.testing.assert_array_equal(top_indices.data, np.array([3, 1], dtype=np.int64))

    bottom_values, bottom_indices = TopK(["x", "k"], ["values", "indices"], axis=-1, largest=0, sorted=1, dtype="bfloat16").forward(
        Tensor(4, dtype="bfloat16", data=_bf16_bits(values)),
        k,
    )["tensor"]
    np.testing.assert_array_equal(bottom_values.data, _bf16_bits(np.array([-2.0, -1.0], dtype=np.float32)))
    np.testing.assert_array_equal(bottom_indices.data, np.array([2, 0], dtype=np.int64))


def test_c_backend_cumsum_integer_dtypes_preserve_wrap_and_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    axis = Tensor(dtype="int64", data=np.array(1, dtype=np.int64))
    int64_values = np.array(
        [[2**60 + 1, 1, 2**60 + 3], [np.iinfo(np.int64).max, 2, 3]],
        dtype=np.int64,
    )
    int64_out = CumSum(["x", "axis"], ["y"], dtype="int64").forward(
        Tensor(*int64_values.shape, dtype="int64", data=int64_values),
        axis,
    )["tensor"]
    np.testing.assert_array_equal(int64_out.data, np.cumsum(int64_values, axis=1, dtype=np.int64))

    uint64_values = np.array([[2**63 + 1, 2**63 + 2, 7]], dtype=np.uint64)
    uint64_out = CumSum(["x", "axis"], ["y"], dtype="uint64").forward(
        Tensor(*uint64_values.shape, dtype="uint64", data=uint64_values),
        axis,
    )["tensor"]
    np.testing.assert_array_equal(uint64_out.data, np.cumsum(uint64_values, axis=1, dtype=np.uint64))

    reverse_exclusive = CumSum(["x", "axis"], ["y"], dtype="uint64", exclusive=1, reverse=1).forward(
        Tensor(*uint64_values.shape, dtype="uint64", data=uint64_values),
        axis,
    )["tensor"]
    reversed_values = uint64_values[:, ::-1]
    expected_reversed = np.zeros_like(reversed_values)
    expected_reversed[:, 1:] = np.cumsum(reversed_values[:, :-1], axis=1, dtype=np.uint64)
    np.testing.assert_array_equal(reverse_exclusive.data, expected_reversed[:, ::-1])


def test_c_backend_reduce_integer_dtypes_preserve_wrap_and_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    axes = Tensor(1, dtype="int64", data=np.array([1], dtype=np.int64))
    int64_values = np.array(
        [[2**60 + 1, 1, 2**60 + 3], [np.iinfo(np.int64).max, 2, 3]],
        dtype=np.int64,
    )
    sum_out = ReduceSum(["x", "axes"], ["y"], dtype="int64").forward(
        Tensor(*int64_values.shape, dtype="int64", data=int64_values),
        axes,
    )["tensor"]
    np.testing.assert_array_equal(sum_out.data, np.sum(int64_values, axis=1, keepdims=True, dtype=np.int64))

    prod_values = np.array([[2**32, 2**31, 3], [-1, 2**62, 3]], dtype=np.int64)
    prod_out = ReduceProd(["x", "axes"], ["y"], dtype="int64").forward(
        Tensor(*prod_values.shape, dtype="int64", data=prod_values),
        axes,
    )["tensor"]
    np.testing.assert_array_equal(prod_out.data, np.prod(prod_values, axis=1, keepdims=True, dtype=np.int64))

    uint64_values = np.array(
        [[2**63 + 1, 2**63 + 3, 2**63 + 2], [2**64 - 1, 7, 2**63]],
        dtype=np.uint64,
    )
    max_out = ReduceMax(["x", "axes"], ["y"], dtype="uint64").forward(
        Tensor(*uint64_values.shape, dtype="uint64", data=uint64_values),
        axes,
    )["tensor"]
    min_out = ReduceMin(["x", "axes"], ["y"], dtype="uint64").forward(
        Tensor(*uint64_values.shape, dtype="uint64", data=uint64_values),
        axes,
    )["tensor"]
    np.testing.assert_array_equal(max_out.data, np.max(uint64_values, axis=1, keepdims=True))
    np.testing.assert_array_equal(min_out.data, np.min(uint64_values, axis=1, keepdims=True))


def test_c_backend_reduce_mean_integer_dtypes_match_onnx_reference_dtype_accumulation():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    axes = Tensor(1, dtype="int64", data=np.array([1], dtype=np.int64))
    cases = [
        ("int32", np.array([[1, 2], [np.iinfo(np.int32).max, 1]], dtype=np.int32)),
        ("int64", np.array([[1, 2], [np.iinfo(np.int64).max, 1], [2**60 + 1, 2**60 + 2]], dtype=np.int64)),
        ("uint32", np.array([[1, 2], [np.iinfo(np.uint32).max, 1], [2**31 + 1, 2**31 + 3]], dtype=np.uint32)),
        ("uint64", np.array([[1, 2], [np.iinfo(np.uint64).max, 1], [2**63 + 1, 2**63 + 3]], dtype=np.uint64)),
    ]

    for dtype, values in cases:
        actual = ReduceMean(["x", "axes"], ["y"], dtype=dtype).forward(
            Tensor(*values.shape, dtype=dtype, data=values),
            axes,
        )["tensor"]
        expected = np.mean(values, axis=1, keepdims=True, dtype=values.dtype)
        np.testing.assert_array_equal(actual.data, expected)


def test_c_backend_formula_reduce_integer_dtypes_match_reference_casting():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    axes = Tensor(1, dtype="int64", data=np.array([1], dtype=np.int64))
    cases = [
        (
            "int32",
            np.array(
                [[np.iinfo(np.int32).min, -2, 3], [50000, 50000, 1]],
                dtype=np.int32,
            ),
        ),
        (
            "int64",
            np.array(
                [[np.iinfo(np.int64).min, -2, 3], [3037000500, 3037000500, 1]],
                dtype=np.int64,
            ),
        ),
        (
            "uint32",
            np.array(
                [[np.iinfo(np.uint32).max, 2, 3], [2**31 + 1, 2, 3]],
                dtype=np.uint32,
            ),
        ),
        (
            "uint64",
            np.array(
                [[np.iinfo(np.uint64).max, 2, 3], [2**63 + 1, 2, 3]],
                dtype=np.uint64,
            ),
        ),
    ]

    for dtype, values in cases:
        with np.errstate(all="ignore"):
            expected_l1 = np.sum(np.abs(values), axis=1, keepdims=True).astype(values.dtype)
            expected_l2 = np.sqrt(np.sum(np.square(values), axis=1, keepdims=True)).astype(values.dtype)
            expected_sum_square = np.sum(np.square(values), axis=1, keepdims=True).astype(values.dtype)

        l1 = ReduceL1(["x", "axes"], ["y"], dtype=dtype).forward(
            Tensor(*values.shape, dtype=dtype, data=values),
            axes,
        )["tensor"]
        l2 = ReduceL2(["x", "axes"], ["y"], dtype=dtype).forward(
            Tensor(*values.shape, dtype=dtype, data=values),
            axes,
        )["tensor"]
        sum_square = ReduceSumSquare(["x", "axes"], ["y"], dtype=dtype).forward(
            Tensor(*values.shape, dtype=dtype, data=values),
            axes,
        )["tensor"]

        np.testing.assert_array_equal(l1.data, expected_l1)
        np.testing.assert_array_equal(l2.data, expected_l2)
        np.testing.assert_array_equal(sum_square.data, expected_sum_square)


def test_c_backend_pow_integer_dtypes_match_numpy_power_wrap():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    cases = [
        ("int32", np.array([2, -2, 2**16, -3], dtype=np.int32), np.array([3, 3, 2, 3], dtype=np.int32)),
        ("int64", np.array([2, -2, 2**32, -3], dtype=np.int64), np.array([63, 3, 2, 3], dtype=np.int64)),
        ("uint32", np.array([2, 2**16, 3], dtype=np.uint32), np.array([31, 2, 3], dtype=np.uint32)),
        ("uint64", np.array([2, 2**32, 2**63 + 1], dtype=np.uint64), np.array([63, 2, 2], dtype=np.uint64)),
    ]

    for dtype, bases, exponents in cases:
        actual = Pow(["a", "b"], ["y"], dtype=dtype).forward(
            Tensor(*bases.shape, dtype=dtype, data=bases),
            Tensor(*exponents.shape, dtype=dtype, data=exponents),
        )["tensor"]
        with np.errstate(all="ignore"):
            expected = np.power(bases, exponents).astype(bases.dtype)
        np.testing.assert_array_equal(actual.data, expected)


def test_c_backend_pow_rejects_only_integer_negative_exponents():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    with pytest.raises(ValueError, match="Integers to negative integer powers are not allowed"):
        Pow(["a", "b"], ["y"], dtype="int32").forward(
            Tensor(1, dtype="int32", data=np.array([2], dtype=np.int32)),
            Tensor(1, dtype="int32", data=np.array([-1], dtype=np.int32)),
        )

    float16_base = np.array([2.0, 4.0], dtype=np.float16)
    signed_exp = np.array([-1, -2], dtype=np.int16)
    float16_actual = Pow(["a", "b"], ["y"], dtype="float16").forward(
        Tensor(2, dtype="float16", data=float16_base),
        Tensor(2, dtype="int16", data=signed_exp),
    )["tensor"]
    np.testing.assert_allclose(
        float16_actual.data,
        np.power(float16_base, signed_exp).astype(np.float16),
        rtol=1e-3,
        atol=1e-3,
    )

    bf16_values = np.array([2.0, 4.0], dtype=np.float32)
    bf16_actual = Pow(["a", "b"], ["y"], dtype="bfloat16").forward(
        Tensor(2, dtype="bfloat16", data=_bf16_bits(bf16_values)),
        Tensor(2, dtype="int16", data=signed_exp),
    )["tensor"]
    expected_bf16 = np.power(bf16_values, signed_exp.astype(np.float32)).astype(np.float32)
    np.testing.assert_allclose(_bf16_to_float32(bf16_actual.data), expected_bf16, rtol=1e-2, atol=1e-2)


def test_python_reduce_log_sum_exp_bfloat16_fallback_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    values = np.array([[10.0, 11.0], [2.0, 3.0]], dtype=np.float32)
    axes = np.array([1], dtype=np.int64)
    actual = ReduceLogSumExp(["x", "axes"], ["y"], keepdims=1, dtype="bfloat16").forward(
        Tensor(*values.shape, dtype="bfloat16", data=_bf16_bits(values)),
        Tensor(1, dtype="int64", data=axes),
    )["tensor"]

    max_values = np.max(values, axis=1, keepdims=True)
    expected = np.log(np.sum(np.exp(values - max_values), axis=1, keepdims=True, dtype=np.float32)) + max_values
    np.testing.assert_allclose(_bf16_to_float32(actual.data), expected, rtol=1e-2, atol=1e-2)


def test_c_backend_compare_and_arg_integer_dtypes_preserve_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    int64_left = np.array([2**60 + 1, 2**60 + 2, -(2**60 + 3), -7], dtype=np.int64)
    int64_right = np.array([2**60 + 2, 2**60 + 1, -(2**60 + 2), -7], dtype=np.int64)
    int64_l = Tensor(4, dtype="int64", data=int64_left)
    int64_r = Tensor(4, dtype="int64", data=int64_right)

    np.testing.assert_array_equal(Equal(["a", "b"], ["y"]).forward(int64_l, int64_r)["tensor"].data, int64_left == int64_right)
    np.testing.assert_array_equal(Greater(["a", "b"], ["y"]).forward(int64_l, int64_r)["tensor"].data, int64_left > int64_right)
    np.testing.assert_array_equal(Less(["a", "b"], ["y"]).forward(int64_l, int64_r)["tensor"].data, int64_left < int64_right)
    np.testing.assert_array_equal(
        GreaterOrEqual(["a", "b"], ["y"]).forward(int64_l, int64_r)["tensor"].data,
        int64_left >= int64_right,
    )
    np.testing.assert_array_equal(
        LessOrEqual(["a", "b"], ["y"]).forward(int64_l, int64_r)["tensor"].data,
        int64_left <= int64_right,
    )

    uint64_left = np.array([2**63 + 1, 2**63 + 2, 2**64 - 1, 0], dtype=np.uint64)
    uint64_right = np.array([2**63 + 2, 2**63 + 1, 2**64 - 2, 0], dtype=np.uint64)
    uint64_l = Tensor(4, dtype="uint64", data=uint64_left)
    uint64_r = Tensor(4, dtype="uint64", data=uint64_right)

    np.testing.assert_array_equal(Equal(["a", "b"], ["y"]).forward(uint64_l, uint64_r)["tensor"].data, uint64_left == uint64_right)
    np.testing.assert_array_equal(Greater(["a", "b"], ["y"]).forward(uint64_l, uint64_r)["tensor"].data, uint64_left > uint64_right)
    np.testing.assert_array_equal(Less(["a", "b"], ["y"]).forward(uint64_l, uint64_r)["tensor"].data, uint64_left < uint64_right)

    int64_arg_values = np.array(
        [[2**60 + 1, 2**60 + 2, 2**60 + 2], [-(2**60 + 1), -(2**60 + 3), -(2**60 + 2)]],
        dtype=np.int64,
    )
    int64_arg_tensor = Tensor(*int64_arg_values.shape, dtype="int64", data=int64_arg_values)
    argmax_first = ArgMax(["x"], ["y"], axis=1, keepdims=0, select_last_index=0).forward(int64_arg_tensor)["tensor"]
    argmax_last = ArgMax(["x"], ["y"], axis=1, keepdims=0, select_last_index=1).forward(int64_arg_tensor)["tensor"]
    argmin_last = ArgMin(["x"], ["y"], axis=1, keepdims=0, select_last_index=1).forward(int64_arg_tensor)["tensor"]
    np.testing.assert_array_equal(argmax_first.data, np.argmax(int64_arg_values, axis=1))
    np.testing.assert_array_equal(argmax_last.data, np.array([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(argmin_last.data, np.array([0, 1], dtype=np.int64))

    uint64_arg_values = np.array([[2**63 + 1, 2**63 + 2, 2**63 + 2], [2**64 - 2, 7, 2**64 - 1]], dtype=np.uint64)
    uint64_arg_tensor = Tensor(*uint64_arg_values.shape, dtype="uint64", data=uint64_arg_values)
    uint64_argmax = ArgMax(["x"], ["y"], axis=1, keepdims=0, select_last_index=1).forward(uint64_arg_tensor)["tensor"]
    uint64_argmin = ArgMin(["x"], ["y"], axis=1, keepdims=0, select_last_index=0).forward(uint64_arg_tensor)["tensor"]
    np.testing.assert_array_equal(uint64_argmax.data, np.array([2, 2], dtype=np.int64))
    np.testing.assert_array_equal(uint64_argmin.data, np.argmin(uint64_arg_values, axis=1))


def test_c_backend_unsigned_integer_binary_ops_wrap_like_numpy():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    u32_left = np.array([0xFFFFFFFF, 0, 0x80000000, 5], dtype=np.uint32)
    u32_right = np.array([1, 1, 2, 7], dtype=np.uint32)
    left_tensor = Tensor(4, dtype="uint32", data=u32_left)
    right_tensor = Tensor(4, dtype="uint32", data=u32_right)

    add = ADD(["a", "b"], ["y"], dtype="uint32").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(add.data, np.add(u32_left, u32_right))

    sub = SUB(["a", "b"], ["y"], dtype="uint32").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(sub.data, np.subtract(u32_left, u32_right))

    mul = MUL(["a", "b"], ["y"], dtype="uint32").forward(left_tensor, right_tensor)["tensor"]
    np.testing.assert_array_equal(mul.data, np.multiply(u32_left, u32_right))

    not_out = BitwiseNot(["x"], ["y"], dtype="uint32").forward(left_tensor)["tensor"]
    np.testing.assert_array_equal(not_out.data, np.bitwise_not(u32_left))

    shift_amounts = np.array([31, 1, 31, 0], dtype=np.uint32)
    shift_tensor = Tensor(4, dtype="uint32", data=shift_amounts)
    shift_left = BitShift(["a", "b"], ["y"], direction="LEFT", dtype="uint32").forward(
        left_tensor, shift_tensor
    )["tensor"]
    np.testing.assert_array_equal(shift_left.data, np.left_shift(u32_left, shift_amounts))

    shift_right = BitShift(["a", "b"], ["y"], direction="RIGHT", dtype="uint32").forward(
        left_tensor, shift_tensor
    )["tensor"]
    np.testing.assert_array_equal(shift_right.data, np.right_shift(u32_left, shift_amounts))

    u64_left = np.array([2**63, 3, 2**64 - 1], dtype=np.uint64)
    u64_right = np.array([2**63 - 1, 4, 1], dtype=np.uint64)
    u64_l = Tensor(3, dtype="uint64", data=u64_left)
    u64_r = Tensor(3, dtype="uint64", data=u64_right)
    max_out = Max(["a", "b"], ["y"], dtype="uint64").forward(u64_l, u64_r)["tensor"]
    min_out = Min(["a", "b"], ["y"], dtype="uint64").forward(u64_l, u64_r)["tensor"]
    np.testing.assert_array_equal(max_out.data, np.maximum(u64_left, u64_right))
    np.testing.assert_array_equal(min_out.data, np.minimum(u64_left, u64_right))


def test_c_backend_pool_mean_and_norm_numeric_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    big_int = np.array([2**60, 2**60 + 3], dtype=np.int64)
    same_cast = Cast(["x"], ["y"], dtype="int64").forward(Tensor(2, dtype="int64", data=big_int))["tensor"]
    np.testing.assert_array_equal(same_cast.data, big_int)

    float_values = Tensor(4, dtype="float32", data=np.array([1.9, -2.2, 0.0, 127.0], dtype=np.float32))
    int_cast = Cast(["x"], ["y"], dtype="int32").forward(float_values)["tensor"]
    np.testing.assert_array_equal(int_cast.data, float_values.data.astype(np.int32))

    cast_target = Tensor(1, dtype="float64", data=np.array([0.0], dtype=np.float64))
    cast_like = CastLike(["x", "target"], ["y"]).forward(float_values, cast_target)["tensor"]
    assert cast_like.dtype == "float64"
    np.testing.assert_array_equal(cast_like.data, float_values.data.astype(np.float64))

    sum_left = Tensor(2, 1, dtype="float32", data=np.array([[1.0], [-2.0]], dtype=np.float32))
    sum_right = Tensor(1, 3, dtype="float32", data=np.array([[3.0, 4.0, 5.0]], dtype=np.float32))
    sum_bias = Tensor(2, 3, dtype="float32", data=np.ones((2, 3), dtype=np.float32))
    summed = Sum(["left", "right", "bias"], ["out"], dtype="float32").forward(sum_left, sum_right, sum_bias)["tensor"]
    expected_sum = sum_left.data + sum_right.data + sum_bias.data
    np.testing.assert_allclose(summed.data, expected_sum, rtol=1e-6)

    slope = Tensor(1, 3, dtype="float32", data=np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    prelu = PRelu(["x", "slope"], ["out"], dtype="float32").forward(summed, slope)["tensor"]
    np.testing.assert_allclose(prelu.data, np.where(expected_sum >= 0, expected_sum, expected_sum * slope.data), rtol=1e-6)

    lrn_data = np.arange(1, 1 + 2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3) / 5.0
    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="float32").forward(
        Tensor(*lrn_data.shape, dtype="float32", data=lrn_data)
    )["tensor"]
    expected_lrn = np.empty_like(lrn_data)
    for n in range(lrn_data.shape[0]):
        for c in range(lrn_data.shape[1]):
            begin, end = max(0, c - 1), min(lrn_data.shape[1], c + 2)
            square_sum = np.sum(lrn_data[n:n + 1, begin:end, ...] ** 2, axis=1)
            expected_lrn[n, c, ...] = lrn_data[n, c, ...] / np.sqrt(1.0 + 0.3 / 3 * square_sum)
    np.testing.assert_allclose(lrn.data, expected_lrn, rtol=1e-6, atol=1e-6)

    mvn_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0, 2], dtype="float32").forward(
        Tensor(*mvn_data.shape, dtype="float32", data=mvn_data)
    )["tensor"]
    mvn_mean = np.mean(mvn_data, axis=(0, 2), keepdims=True)
    expected_mvn = (mvn_data - mvn_mean) / np.sqrt(np.mean((mvn_data - mvn_mean) ** 2, axis=(0, 2), keepdims=True))
    np.testing.assert_allclose(mvn.data, expected_mvn, rtol=1e-6, atol=1e-6)

    eye = EyeLike(["x"], ["out"], k=-1, dtype="int64").forward(Tensor_(4, 3, dtype="float32"))["tensor"]
    np.testing.assert_array_equal(eye.data, np.eye(4, 3, k=-1, dtype=np.int64))

    data = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2) - 5.0
    x = Tensor(*data.shape, dtype="float32", data=data)

    avg = GlobalAveragePool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(avg.data, np.mean(data, axis=(2, 3, 4), keepdims=True), rtol=1e-6)
    assert avg.size == (2, 3, 1, 1, 1)

    max_pool = GlobalMaxPool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(max_pool.data, np.max(data, axis=(2, 3, 4), keepdims=True), rtol=1e-6)

    lp = GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(lp.data, np.sum(np.abs(data) ** 2, axis=(2, 3, 4), keepdims=True) ** 0.5, rtol=1e-6)

    left = Tensor(2, 3, dtype="float32", data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    right = Tensor(1, 3, dtype="float32", data=np.array([[3, 2, 1]], dtype=np.float32))
    mean = Mean(["left", "right"], ["out"], dtype="float32").forward(left, right)["tensor"]
    expected_mean = np.mean(np.stack(np.broadcast_arrays(left.data, right.data), axis=0), axis=0)
    np.testing.assert_allclose(mean.data, expected_mean, rtol=1e-6)

    bn_data = np.linspace(-1, 1, 12, dtype=np.float32).reshape(2, 3, 2)
    bn_x = Tensor(*bn_data.shape, dtype="float32", data=bn_data)
    bn_scale = Tensor(3, dtype="float32", data=np.array([1.0, 1.5, 0.5], dtype=np.float32))
    bn_bias = Tensor(3, dtype="float32", data=np.array([0.0, 0.1, -0.2], dtype=np.float32))
    bn_mean = Tensor(3, dtype="float32", data=np.array([0.1, -0.2, 0.3], dtype=np.float32))
    bn_var = Tensor(3, dtype="float32", data=np.array([0.8, 1.1, 0.6], dtype=np.float32))
    bn = BatchNormalization(["x", "scale", "bias", "mean", "var"], ["y"], epsilon=1e-5, dtype="float32")
    bn_y = bn.forward(bn_x, bn_scale, bn_bias, bn_mean, bn_var)["tensor"]
    expected_bn = (
        bn_scale.data.reshape(1, 3, 1)
        * (bn_data - bn_mean.data.reshape(1, 3, 1))
        / np.sqrt(bn_var.data.reshape(1, 3, 1) + 1e-5)
        + bn_bias.data.reshape(1, 3, 1)
    )
    np.testing.assert_allclose(bn_y.data, expected_bn, rtol=1e-6, atol=1e-6)

    ln_data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float32)
    ln_x = Tensor(*ln_data.shape, dtype="float32", data=ln_data)
    ln_scale = Tensor(4, dtype="float32", data=np.array([1.0, 0.5, 1.5, 2.0], dtype=np.float32))
    ln_bias = Tensor(4, dtype="float32", data=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32))
    ln_y = LayerNormalization(["x", "scale", "bias"], ["y"], axis=-1, epsilon=1e-5, dtype="float32").forward(
        ln_x, ln_scale, ln_bias
    )["tensor"]
    ln_mean = ln_data.mean(axis=1, keepdims=True)
    ln_inv_std = np.reciprocal(np.sqrt(((ln_data - ln_mean) ** 2).mean(axis=1, keepdims=True) + 1e-5))
    expected_ln = (ln_data - ln_mean) * ln_inv_std * ln_scale.data + ln_bias.data
    np.testing.assert_allclose(ln_y.data, expected_ln, rtol=1e-6, atol=1e-6)

def test_c_backend_quantized_matmul_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a_data = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.uint8)
    b_data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    a = Tensor(*a_data.shape, dtype="uint8", data=a_data)
    b = Tensor(*b_data.shape, dtype="int8", data=b_data)
    a_zp = Tensor(2, dtype="uint8", data=np.array([2, 5], dtype=np.uint8))
    b_zp = Tensor(2, dtype="int8", data=np.array([1, -1], dtype=np.int8))
    matmul_int = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(a, b, a_zp, b_zp)["tensor"]
    expected_int = np.matmul(
        a_data.astype(np.int32) - a_zp.data.astype(np.int32).reshape(2, 1),
        b_data.astype(np.int32) - b_zp.data.astype(np.int32).reshape(1, 2),
    ).astype(np.int32)
    np.testing.assert_array_equal(matmul_int.data, expected_int)

    batch_a = np.array([[[2, 3, 4], [5, 6, 7]], [[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    batch_b = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.uint8)
    batched = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(
        Tensor(*batch_a.shape, dtype="uint8", data=batch_a),
        Tensor(*batch_b.shape, dtype="uint8", data=batch_b),
        Tensor(2, 2, 1, dtype="uint8", data=np.array([[[1], [2]], [[0], [1]]], dtype=np.uint8)),
        Tensor(1, 1, 2, dtype="uint8", data=np.array([[[1, 2]]], dtype=np.uint8)),
    )["tensor"]
    expected_batched = np.matmul(
        batch_a.astype(np.int32) - np.array([[[1], [2]], [[0], [1]]], dtype=np.int32),
        batch_b.astype(np.int32) - np.array([[[1, 2]]], dtype=np.int32),
    ).astype(np.int32)
    np.testing.assert_array_equal(batched.data, expected_batched)

    a_scale = Tensor(2, dtype="float32", data=np.array([0.5, 0.25], dtype=np.float32))
    b_scale = Tensor(2, dtype="float32", data=np.array([0.2, 0.4], dtype=np.float32))
    y_scale = Tensor(1, dtype="float32", data=np.array([0.1], dtype=np.float32))
    y_zp = Tensor(1, dtype="uint8", data=np.array([100], dtype=np.uint8))
    qlinear = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
    )["tensor"]
    a_real = (a_data.astype(np.float64) - a_zp.data.reshape(2, 1)) * a_scale.data.reshape(2, 1)
    b_real = (b_data.astype(np.float64) - b_zp.data.reshape(1, 2)) * b_scale.data.reshape(1, 2)
    expected_q = np.rint(np.matmul(a_real, b_real) / y_scale.data.item() + y_zp.data.item())
    expected_q = np.clip(expected_q, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(qlinear.data, expected_q)

def test_c_backend_conv_integer_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.uint8).reshape(1, 2, 4, 4)
    w_data = np.array(
        [
            [[[1, 0], [0, 1]], [[1, 1], [0, 0]]],
            [[[0, 1], [1, 0]], [[1, 0], [1, 0]]],
        ],
        dtype=np.int8,
    )
    x = Tensor(*x_data.shape, dtype="uint8", data=x_data)
    w = Tensor(*w_data.shape, dtype="int8", data=w_data)
    x_zp = Tensor(1, dtype="uint8", data=np.array([2], dtype=np.uint8))
    w_zp = Tensor(2, dtype="int8", data=np.array([0, 1], dtype=np.int8))
    conv_int = ConvInteger(
        ["x", "w", "xz", "wz"], ["y"], pads=[1, 1, 1, 1], strides=[2, 2], dilations=[1, 1], group=1
    ).forward(x, w, x_zp, w_zp)["tensor"]
    expected = np.zeros((1, 2, 3, 3), dtype=np.int32)
    x_centered = x_data.astype(np.int32) - 2
    w_centered = w_data.astype(np.int32) - w_zp.data.astype(np.int32).reshape(2, 1, 1, 1)
    x_padded = np.pad(x_centered, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    for oc in range(2):
        for oh in range(3):
            for ow in range(3):
                patch = x_padded[0, :, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                expected[0, oc, oh, ow] = np.sum(patch * w_centered[oc])
    np.testing.assert_array_equal(conv_int.data, expected)

    grouped_x = Tensor(1, 2, 3, 3, dtype="uint8", data=np.arange(1, 19, dtype=np.uint8).reshape(1, 2, 3, 3))
    grouped_w_data = np.array([[[[1, 0], [0, 1]]], [[[1, 1], [1, 0]]]], dtype=np.uint8)
    grouped = ConvInteger(["x", "w"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1], group=2).forward(
        grouped_x, Tensor(*grouped_w_data.shape, dtype="uint8", data=grouped_w_data)
    )["tensor"]
    expected_grouped = np.empty((1, 2, 2, 2), dtype=np.int32)
    for oc in range(2):
        for oh in range(2):
            for ow in range(2):
                patch = grouped_x.data[0, oc:oc + 1, oh:oh + 2, ow:ow + 2].astype(np.int32)
                expected_grouped[0, oc, oh, ow] = np.sum(patch * grouped_w_data[oc].astype(np.int32))
    np.testing.assert_array_equal(grouped.data, expected_grouped)

def test_c_backend_conv_transpose_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    w_data = np.array(
        [
            [[[1.0, -1.0], [0.5, 2.0]], [[0.0, 1.0], [1.5, -0.5]]],
            [[[-0.5, 1.0], [2.0, 0.0]], [[1.0, 0.5], [-1.0, 1.0]]],
        ],
        dtype=np.float32,
    )
    b_data = np.array([0.25, -0.75], dtype=np.float32)
    x = Tensor(*x_data.shape, dtype="float32", data=x_data)
    w = Tensor(*w_data.shape, dtype="float32", data=w_data)
    b = Tensor(*b_data.shape, dtype="float32", data=b_data)
    op = ConvTranspose(
        ["x", "w", "b"],
        ["y"],
        pads=[1, 0, 0, 1],
        strides=[2, 1],
        dilations=[1, 1],
        output_padding=[1, 0],
        dtype="float32",
    )
    out = op.forward(x, w, b)["tensor"]

    expected = np.zeros((1, 2, 4, 3), dtype=np.float32)
    for n in range(x_data.shape[0]):
        for ic in range(x_data.shape[1]):
            for ih in range(x_data.shape[2]):
                for iw in range(x_data.shape[3]):
                    for oc in range(w_data.shape[1]):
                        for kh in range(w_data.shape[2]):
                            for kw in range(w_data.shape[3]):
                                oh = ih * 2 + kh - 1
                                ow = iw + kw
                                if 0 <= oh < expected.shape[2] and 0 <= ow < expected.shape[3]:
                                    expected[n, oc, oh, ow] += x_data[n, ic, ih, iw] * w_data[ic, oc, kh, kw]
    expected += b_data.reshape(1, 2, 1, 1)

    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-6)

def test_c_backend_qlinear_conv_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.uint8).reshape(1, 2, 4, 4)
    w_data = np.array(
        [
            [[[1, 2], [3, 4]], [[2, 1], [0, 3]]],
            [[[4, 1], [2, 0]], [[1, 3], [2, 4]]],
        ],
        dtype=np.uint8,
    )
    x = Tensor(*x_data.shape, dtype="uint8", data=x_data)
    w = Tensor(*w_data.shape, dtype="uint8", data=w_data)
    x_scale = Tensor(1, dtype="float32", data=np.array([0.2], dtype=np.float32))
    x_zp = Tensor(1, dtype="uint8", data=np.array([3], dtype=np.uint8))
    w_scale = Tensor(2, dtype="float32", data=np.array([0.25, 0.5], dtype=np.float32))
    w_zp = Tensor(2, dtype="uint8", data=np.array([1, 2], dtype=np.uint8))
    y_scale = Tensor(1, dtype="float32", data=np.array([0.1], dtype=np.float32))
    y_zp = Tensor(1, dtype="uint8", data=np.array([7], dtype=np.uint8))
    bias = Tensor(2, dtype="int32", data=np.array([3, -4], dtype=np.int32))

    qconv = QLinearConv(
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"],
        ["y"],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dtype="uint8",
    ).forward(x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias)["tensor"]

    expected = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    x_centered = x_data.astype(np.int32) - int(x_zp.data[0])
    w_centered = w_data.astype(np.int32) - w_zp.data.astype(np.int32).reshape(2, 1, 1, 1)
    x_padded = np.pad(x_centered, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    for oc in range(2):
        for oh in range(3):
            for ow in range(3):
                patch = x_padded[0, :, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                acc = int(np.sum(patch * w_centered[oc])) + int(bias.data[oc])
                scaled = acc * float(x_scale.data[0]) * float(w_scale.data[oc]) / float(y_scale.data[0])
                expected[0, oc, oh, ow] = np.clip(np.rint(scaled + int(y_zp.data[0])), 0, 255).astype(np.uint8)

    np.testing.assert_array_equal(qconv.data, expected)

def test_c_backend_max_unpool_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    pooled = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    indices = Tensor(1, 1, 2, 2, dtype="int64", data=np.array([[[[5, 7], [13, 15]]]], dtype=np.int64))
    output_shape = Tensor(4, dtype="int64", data=np.array([1, 1, 5, 5], dtype=np.int64))

    unpooled = MaxUnpool(["x", "i", "shape"], ["y"], kernel_shape=[2, 2], strides=[2, 2], dtype="float32").forward(
        pooled, indices, output_shape
    )["tensor"]

    expected = np.zeros((1, 1, 5, 5), dtype=np.float32)
    expected.reshape(-1)[[6, 8, 16, 18]] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(unpooled.data, expected)

def test_c_backend_unique_and_mel_weight_matrix_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    unique_input = Tensor(6, dtype="int64", data=np.array([3, 1, 3, 2, 1, 3], dtype=np.int64))
    y, indices, inverse, counts = Unique(
        ["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64"
    ).forward(unique_input)["tensor"]
    np.testing.assert_array_equal(y.data, np.array([3, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(indices.data, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(inverse.data, np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(counts.data, np.array([3, 2, 1], dtype=np.int64))

    mel = MelWeightMatrix([], ["mel"], output_datatype=TensorProto.FLOAT).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(8, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(16000, dtype=np.int64)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(8000.0, dtype=np.float32)),
    )["tensor"]
    expected_mel = np.array(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(mel.data, expected_mel, rtol=1e-6, atol=1e-6)

def test_c_backend_dft_and_stft_paths_against_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    from onnx.reference import ReferenceEvaluator

    signal_data = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    dft_len = np.array(4, dtype=np.int64)
    signal = Tensor(*signal_data.shape, dtype="float32", data=signal_data)
    dft = DFT(["x", "dft_len"], ["y"], axis=1, onesided=1, dtype="float32").forward(
        signal, Tensor(dtype="int64", data=dft_len)
    )["tensor"]
    dft_graph = helper.make_graph(
        [helper.make_node("DFT", ["x", "dft_len"], ["y"], axis=1, onesided=1)],
        "dft_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(signal_data.shape)),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])],
    )
    dft_expected = ReferenceEvaluator(helper.make_model(dft_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": signal_data, "dft_len": dft_len}
    )[0]
    np.testing.assert_allclose(dft.data, dft_expected, rtol=1e-6, atol=1e-6)

    inverse = DFT(["x", "dft_len"], ["y"], axis=1, inverse=1, onesided=1, dtype="float32").forward(
        dft, Tensor(dtype="int64", data=dft_len)
    )["tensor"]
    inv_graph = helper.make_graph(
        [helper.make_node("DFT", ["x", "dft_len"], ["y"], axis=1, inverse=1, onesided=1)],
        "dft_inverse_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(dft.data.shape)),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(signal_data.shape))],
    )
    inv_expected = ReferenceEvaluator(helper.make_model(inv_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": dft.data, "dft_len": dft_len}
    )[0]
    np.testing.assert_allclose(inverse.data, inv_expected, rtol=1e-6, atol=1e-6)

    frame_step = np.array(2, dtype=np.int64)
    frame_length = np.array(2, dtype=np.int64)
    window_data = np.ones((2,), dtype=np.float32)
    stft = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="float32").forward(
        signal,
        Tensor(dtype="int64", data=frame_step),
        Tensor(*window_data.shape, dtype="float32", data=window_data),
        Tensor(dtype="int64", data=frame_length),
    )["tensor"]
    stft_graph = helper.make_graph(
        [helper.make_node("STFT", ["x", "step", "window", "length"], ["y"], onesided=1)],
        "stft_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(signal_data.shape)),
            helper.make_tensor_value_info("step", TensorProto.INT64, []),
            helper.make_tensor_value_info("window", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("length", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])],
    )
    stft_expected = ReferenceEvaluator(helper.make_model(stft_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": signal_data, "step": frame_step, "window": window_data, "length": frame_length}
    )[0]
    np.testing.assert_allclose(stft.data, stft_expected, rtol=1e-6, atol=1e-6)

def test_c_backend_recurrent_paths_match_python_semantics():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = Tensor(3, 2, 2, dtype="float32", data=np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[1.0, 0.3], [-0.3, 0.2]],
            [[0.2, 0.7], [0.8, -0.5]],
        ],
        dtype=np.float32,
    ))
    sequence_lens = Tensor(2, dtype="int32", data=np.array([3, 2], dtype=np.int32))

    rnn_w = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.1, 0.2], [-0.2, 0.3]], [[-0.3, 0.4], [0.2, 0.1]]], dtype=np.float32
    ))
    rnn_r = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.5, 0.1], [0.2, 0.4]], [[0.2, -0.1], [0.3, 0.2]]], dtype=np.float32
    ))
    rnn_b = Tensor(2, 4, dtype="float32", data=np.array([[0.1, -0.1, 0.05, 0.02], [0.0, 0.1, -0.03, 0.04]], dtype=np.float32))
    rnn_initial = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.1, 0.0], [0.0, 0.2]], [[-0.1, 0.1], [0.2, -0.2]]], dtype=np.float32
    ))
    rnn_c = RNN(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, direction="bidirectional", dtype="float32")
    rnn_py = RNN(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, direction="bidirectional", dtype="float32")
    rnn_py.lib = None
    c_y, c_h = rnn_c.forward(x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial)["tensor"]
    py_y, py_h = rnn_py.forward(x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)

    gru_w = Tensor(1, 6, 2, dtype="float32", data=np.array([[
        [0.2, -0.1], [0.1, 0.3],
        [-0.2, 0.4], [0.3, -0.2],
        [0.4, 0.1], [-0.1, 0.2],
    ]], dtype=np.float32))
    gru_r = Tensor(1, 6, 2, dtype="float32", data=np.array([[
        [0.1, 0.2], [0.2, -0.1],
        [0.3, 0.1], [-0.2, 0.4],
        [0.2, 0.3], [0.1, -0.3],
    ]], dtype=np.float32))
    gru_b = Tensor(1, 12, dtype="float32", data=np.linspace(-0.2, 0.2, 12, dtype=np.float32).reshape(1, 12))
    gru_initial = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.1, -0.1], [0.2, 0.0]]], dtype=np.float32))
    gru_c = GRU(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, linear_before_reset=1, dtype="float32")
    gru_py = GRU(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, linear_before_reset=1, dtype="float32")
    gru_py.lib = None
    c_y, c_h = gru_c.forward(x, gru_w, gru_r, gru_b, sequence_lens, gru_initial)["tensor"]
    py_y, py_h = gru_py.forward(x, gru_w, gru_r, gru_b, sequence_lens, gru_initial)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)

    lstm_w = Tensor(1, 8, 2, dtype="float32", data=np.linspace(-0.3, 0.4, 16, dtype=np.float32).reshape(1, 8, 2))
    lstm_r = Tensor(1, 8, 2, dtype="float32", data=np.linspace(0.2, -0.2, 16, dtype=np.float32).reshape(1, 8, 2))
    lstm_b = Tensor(1, 16, dtype="float32", data=np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(1, 16))
    lstm_initial_h = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.1, 0.0], [-0.1, 0.2]]], dtype=np.float32))
    lstm_initial_c = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.0, 0.2], [0.1, -0.1]]], dtype=np.float32))
    peepholes = Tensor(1, 6, dtype="float32", data=np.linspace(-0.05, 0.05, 6, dtype=np.float32).reshape(1, 6))
    lstm_c = LSTM(["x", "w", "r", "b", "seq", "h", "c", "p"], ["y", "yh", "yc"], hidden_size=2, input_forget=1, dtype="float32")
    lstm_py = LSTM(["x", "w", "r", "b", "seq", "h", "c", "p"], ["y", "yh", "yc"], hidden_size=2, input_forget=1, dtype="float32")
    lstm_py.lib = None
    c_y, c_h, c_c = lstm_c.forward(x, lstm_w, lstm_r, lstm_b, sequence_lens, lstm_initial_h, lstm_initial_c, peepholes)["tensor"]
    py_y, py_h, py_c = lstm_py.forward(x, lstm_w, lstm_r, lstm_b, sequence_lens, lstm_initial_h, lstm_initial_c, peepholes)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_c.data, py_c.data, rtol=1e-6, atol=1e-6)

def test_c_backend_probability_and_loss_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    probabilities = Tensor(
        2, 3, dtype="float32",
        data=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    samples = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=4, seed=7.0).forward(probabilities)["tensor"]
    np.testing.assert_array_equal(samples.data, np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64))

    log_probs = Tensor(
        2, 3, 2, dtype="float32",
        data=np.array(
            [[[-0.1, -0.2], [-1.0, -1.1], [-2.0, -2.1]],
             [[-0.3, -0.4], [-1.2, -1.3], [-2.2, -2.3]]],
            dtype=np.float32,
        ),
    )
    labels = Tensor(2, 2, dtype="int64", data=np.array([[0, 2], [1, -1]], dtype=np.int64))
    weights = Tensor(3, dtype="float32", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    nll = NegativeLogLikelihoodLoss(
        ["x", "target", "w"], ["loss"], reduction="mean", ignore_index=-1, dtype="float32"
    ).forward(log_probs, labels, weights)["tensor"]
    expected_weighted = np.array([[0.1, 2.1 * 3.0], [1.2 * 2.0, 0.0]], dtype=np.float32)
    expected_denom = np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32).sum()
    np.testing.assert_allclose(nll.data, expected_weighted.sum() / expected_denom, rtol=1e-6)

    scores = Tensor(2, 3, dtype="float32", data=np.array([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=np.float32))
    labels_1d = Tensor(2, dtype="int64", data=np.array([2, 0], dtype=np.int64))
    sce_loss, log_prob = SoftmaxCrossEntropyLoss(
        ["scores", "labels"], ["loss", "log_prob"], reduction="none", dtype="float32"
    ).forward(scores, labels_1d)["tensor"]
    shifted = scores.data - np.max(scores.data, axis=1, keepdims=True)
    expected_log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    np.testing.assert_allclose(log_prob.data, expected_log_prob, rtol=1e-6)
    np.testing.assert_allclose(sce_loss.data, -expected_log_prob[np.arange(2), labels_1d.data], rtol=1e-6)

def test_c_backend_non_max_suppression_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    boxes = Tensor(
        1, 3, 4, dtype="float32",
        data=np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 10.0, 1.0, 11.0]]], dtype=np.float32),
    )
    scores = Tensor(1, 1, 3, dtype="float32", data=np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32))
    selected = NonMaxSuppression(["boxes", "scores", "max", "iou"], ["selected"]).forward(
        boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(selected.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))

    center_boxes = Tensor(
        1, 3, 4, dtype="float32",
        data=np.array([[[0.5, 0.5, 1.0, 1.0], [0.55, 0.5, 1.0, 1.0], [10.5, 0.5, 1.0, 1.0]]], dtype=np.float32),
    )
    center_selected = NonMaxSuppression(
        ["boxes", "scores", "max", "iou"], ["selected"], center_point_box=1
    ).forward(
        center_boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(center_selected.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))

def test_c_backend_grid_sample_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    from onnx.reference import ReferenceEvaluator

    x_data = np.arange(1, 1 + 1 * 1 * 3 * 3, dtype=np.float32).reshape(1, 1, 3, 3)
    grid_data = np.array(
        [[[[-1.0, -1.0], [0.0, 0.0], [1.2, 1.2]], [[0.5, -0.5], [-1.5, 0.5], [0.2, 1.5]]]],
        dtype=np.float32,
    )
    for mode, padding_mode, align_corners in [
        ("linear", "zeros", 0),
        ("nearest", "border", 0),
        ("cubic", "reflection", 1),
    ]:
        graph = helper.make_graph(
            [helper.make_node("GridSample", ["x", "grid"], ["y"], mode=mode, padding_mode=padding_mode, align_corners=align_corners)],
            "grid_sample_ref",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, x_data.shape),
                helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_data.shape),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        expected = ReferenceEvaluator(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])).run(
            None, {"x": x_data, "grid": grid_data}
        )[0]
        actual = GridSample(
            ["x", "grid"], ["y"], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype="float32"
        ).forward(
            Tensor(*x_data.shape, dtype="float32", data=x_data),
            Tensor(*grid_data.shape, dtype="float32", data=grid_data),
        )["tensor"].data
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    alias = GridSample(["x", "grid"], ["y"], mode="bilinear", dtype="float32").forward(
        Tensor(*x_data.shape, dtype="float32", data=x_data),
        Tensor(*grid_data.shape, dtype="float32", data=grid_data),
    )["tensor"].data
    linear = GridSample(["x", "grid"], ["y"], mode="linear", dtype="float32").forward(
        Tensor(*x_data.shape, dtype="float32", data=x_data),
        Tensor(*grid_data.shape, dtype="float32", data=grid_data),
    )["tensor"].data
    np.testing.assert_allclose(alias, linear, rtol=1e-6, atol=1e-6)
