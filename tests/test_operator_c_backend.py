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
