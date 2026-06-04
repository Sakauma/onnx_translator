# /**
#   ******************************************************************************
#   * @file        test_operator_gap_ops.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 ONNX17 缺口补齐算子、概率损失、谱算子、循环算子和字符串算子。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


def test_independent_onnx17_gap_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    det_input = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [0.0, 5.0]]], dtype=np.float32
    ))
    det = Det(["x"], ["y"], dtype="float32").forward(det_input)["tensor"]
    np.testing.assert_allclose(det.data, np.linalg.det(det_input.data).astype(np.float32))
    assert Det(["x"], ["y"], dtype="float32").forward_(Tensor_(2, 2, 2, dtype="float32"))["tensor"].size == (2,)

    unique_input = Tensor(6, dtype="int64", data=np.array([3, 1, 3, 2, 1, 3], dtype=np.int64))
    unique_y, unique_idx, unique_inv, unique_counts = Unique(
        ["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64"
    ).forward(unique_input)["tensor"]
    np.testing.assert_array_equal(unique_y.data, np.array([3, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(unique_idx.data, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(unique_inv.data, np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(unique_counts.data, np.array([3, 2, 1], dtype=np.int64))

    lrn_data = np.arange(1, 1 + 1 * 4 * 1 * 1, dtype=np.float32).reshape(1, 4, 1, 1)
    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="float32").forward(
        Tensor(*lrn_data.shape, dtype="float32", data=lrn_data)
    )["tensor"]
    expected_lrn = np.empty_like(lrn_data)
    for c in range(4):
        begin, end = max(0, c - 1), min(4, c + 2)
        square_sum = np.sum(lrn_data[:, begin:end, ...] ** 2, axis=1)
        expected_lrn[:, c, ...] = lrn_data[:, c, ...] / np.sqrt(1.0 + 0.3 / 3 * square_sum)
    np.testing.assert_allclose(lrn.data, expected_lrn)

    mvn_input = Tensor(2, 2, dtype="float32", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0], dtype="float32").forward(mvn_input)["tensor"]
    mean = np.mean(mvn_input.data, axis=(0,), keepdims=True)
    expected_mvn = (mvn_input.data - mean) / np.sqrt(np.mean((mvn_input.data - mean) ** 2, axis=(0,), keepdims=True))
    np.testing.assert_allclose(mvn.data, expected_mvn)

    bn_data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    bn_x = Tensor(*bn_data.shape, dtype="float32", data=bn_data)
    bn_scale = Tensor(2, dtype="float32", data=np.array([1.5, 0.5], dtype=np.float32))
    bn_bias = Tensor(2, dtype="float32", data=np.array([0.25, -0.75], dtype=np.float32))
    bn_mean = Tensor(2, dtype="float32", data=np.array([10.0, 20.0], dtype=np.float32))
    bn_var = Tensor(2, dtype="float32", data=np.array([4.0, 9.0], dtype=np.float32))
    bn = BatchNormalization(
        ["x", "scale", "bias", "mean", "var"], ["y", "running_mean", "running_var"],
        epsilon=1e-5, momentum=0.8, training_mode=1, dtype="float32"
    )
    bn_y, bn_running_mean, bn_running_var = bn.forward(bn_x, bn_scale, bn_bias, bn_mean, bn_var)["tensor"]
    saved_mean = np.mean(bn_data, axis=(0, 2))
    saved_var = np.var(bn_data, axis=(0, 2))
    expected_bn = (
        bn_scale.data.reshape(1, 2, 1)
        * (bn_data - saved_mean.reshape(1, 2, 1))
        / np.sqrt(saved_var.reshape(1, 2, 1) + 1e-5)
        + bn_bias.data.reshape(1, 2, 1)
    )
    np.testing.assert_allclose(bn_y.data, expected_bn, rtol=1e-6)
    np.testing.assert_allclose(bn_running_mean.data, bn_mean.data * 0.8 + saved_mean * 0.2)
    np.testing.assert_allclose(bn_running_var.data, bn_var.data * 0.8 + saved_var * 0.2)
    inferred_bn = bn.forward_(
        Tensor_(2, 2, 2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
    )["tensor"]
    assert [out.size for out in inferred_bn] == [(2, 2, 2), (2,), (2,)]

    bn_model_path = tmp_path / "batch_norm_training.onnx"
    bn_graph = helper.make_graph(
        [helper.make_node(
            "BatchNormalization",
            ["x", "scale", "bias", "mean", "var"],
            ["y", "running_mean", "running_var"],
            epsilon=1e-5,
            momentum=0.8,
            training_mode=1,
        )],
        "batch_norm_training",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("running_mean", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("running_var", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(bn_graph, opset_imports=[helper.make_opsetid("", 17)]), bn_model_path)
    bn_imported = [op for op in ONNXImport(str(bn_model_path), strict=True) if isinstance(op, BatchNormalization)]
    assert bn_imported[0].training_mode == 1

    ln_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ln_scale = np.linspace(0.5, 1.6, 12, dtype=np.float32).reshape(3, 4)
    ln_bias = np.linspace(-0.3, 0.8, 12, dtype=np.float32).reshape(3, 4)
    ln = LayerNormalization(
        ["x", "scale", "bias"], ["y", "mean", "inv_std"],
        axis=1, epsilon=1e-5, stash_type=1, dtype="float32"
    )
    ln_y, ln_mean, ln_inv_std = ln.forward(
        Tensor(*ln_data.shape, dtype="float32", data=ln_data),
        Tensor(*ln_scale.shape, dtype="float32", data=ln_scale),
        Tensor(*ln_bias.shape, dtype="float32", data=ln_bias),
    )["tensor"]
    ln_mat = ln_data.reshape(2, 12)
    expected_mean = np.mean(ln_mat, axis=1, keepdims=True).reshape(2, 1, 1)
    expected_inv_std = np.reciprocal(np.sqrt(np.mean((ln_mat - expected_mean.reshape(2, 1)) ** 2, axis=1, keepdims=True) + 1e-5)).reshape(2, 1, 1)
    expected_ln = ((ln_data - expected_mean) * expected_inv_std) * ln_scale + ln_bias
    np.testing.assert_allclose(ln_y.data, expected_ln, rtol=1e-6)
    np.testing.assert_allclose(ln_mean.data, expected_mean)
    np.testing.assert_allclose(ln_inv_std.data, expected_inv_std)
    inferred_ln = ln.forward_(
        Tensor_(2, 3, 4, dtype="float32"),
        Tensor_(3, 4, dtype="float32"),
        Tensor_(3, 4, dtype="float32"),
    )["tensor"]
    assert [out.size for out in inferred_ln] == [(2, 3, 4), (2, 1, 1), (2, 1, 1)]

    ln_model_path = tmp_path / "layer_norm_stash.onnx"
    ln_graph = helper.make_graph(
        [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y", "mean", "inv_std"], axis=1, epsilon=1e-5, stash_type=1)],
        "layer_norm_stash",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3, 4]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2, 1, 1]),
            helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, [2, 1, 1]),
        ],
    )
    onnx.save(helper.make_model(ln_graph, opset_imports=[helper.make_opsetid("", 17)]), ln_model_path)
    ln_imported = [op for op in ONNXImport(str(ln_model_path), strict=True) if isinstance(op, LayerNormalization)]
    assert ln_imported[0].stash_type == 1

    a = Tensor(2, 3, dtype="uint8", data=np.array([[2, 3, 4], [5, 6, 7]], dtype=np.uint8))
    b = Tensor(3, 2, dtype="int8", data=np.array([[1, -2], [3, 4], [-1, 2]], dtype=np.int8))
    a_zp = Tensor(1, dtype="uint8", data=np.array([2], dtype=np.uint8))
    b_zp = Tensor(2, dtype="int8", data=np.array([1, -1], dtype=np.int8))
    matmul_int = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(a, b, a_zp, b_zp)["tensor"]
    expected_int = np.matmul(a.data.astype(np.int32) - 2, b.data.astype(np.int32) - np.array([1, -1], dtype=np.int32))
    np.testing.assert_array_equal(matmul_int.data, expected_int.astype(np.int32))

    qlinear = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        a,
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        a_zp,
        Tensor(3, 2, dtype="uint8", data=np.array([[3, 4], [5, 6], [7, 8]], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.25], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([3], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([10], dtype=np.uint8)),
    )["tensor"]
    assert qlinear.dtype == "uint8"
    assert qlinear.size == (2, 2)

    conv_x = Tensor(1, 1, 3, 3, dtype="uint8", data=np.arange(1, 10, dtype=np.uint8).reshape(1, 1, 3, 3))
    conv_w = Tensor(1, 1, 2, 2, dtype="int8", data=np.array([[[[1, 0], [0, 1]]]], dtype=np.int8))
    conv_int = ConvInteger(["x", "w", "xzp", "wzp"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1]).forward(
        conv_x,
        conv_w,
        Tensor(1, dtype="uint8", data=np.array([1], dtype=np.uint8)),
        Tensor(1, dtype="int8", data=np.array([0], dtype=np.int8)),
    )["tensor"]
    expected_conv_int = np.array([[[[4, 6], [10, 12]]]], dtype=np.int32)
    np.testing.assert_array_equal(conv_int.data, expected_conv_int)
    assert ConvInteger(["x", "w"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1]).forward_(
        Tensor_(1, 1, 3, 3, dtype="uint8"), Tensor_(1, 1, 2, 2, dtype="int8")
    )["tensor"].size == (1, 1, 2, 2)

    qconv = QLinearConv(["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"], ["y"], pads=[0, 0, 0, 0], dtype="uint8").forward(
        conv_x,
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
        Tensor(1, 1, 2, 2, dtype="uint8", data=np.array([[[[1, 0], [0, 1]]]], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.25], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.125], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
    )["tensor"]
    np.testing.assert_array_equal(qconv.data, np.array([[[[6, 8], [12, 14]]]], dtype=np.uint8))

    deconv_x = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    deconv_w = Tensor(1, 1, 2, 2, dtype="float32", data=np.ones((1, 1, 2, 2), dtype=np.float32))
    deconv = ConvTranspose(["x", "w"], ["y"], strides=[2, 2], pads=[0, 0, 0, 0], dtype="float32").forward(
        deconv_x, deconv_w
    )["tensor"]
    np.testing.assert_array_equal(
        deconv.data,
        np.array([[[[1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0]]]], dtype=np.float32),
    )
    assert ConvTranspose(["x", "w"], ["y"], strides=[2, 2], pads=[0, 0, 0, 0]).forward_(
        Tensor_(1, 1, 2, 2, dtype="float32"), Tensor_(1, 1, 2, 2, dtype="float32")
    )["tensor"].size == (1, 1, 4, 4)

    boxes = Tensor(1, 3, 4, dtype="float32", data=np.array([[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 10, 1, 11]]], dtype=np.float32))
    scores = Tensor(1, 1, 3, dtype="float32", data=np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32))
    nms = NonMaxSuppression(["boxes", "scores", "max", "iou"], ["selected"]).forward(
        boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(nms.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))

    model_path = tmp_path / "onnx17_independent_gap_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Det", ["det_x"], ["det_y"]),
            helper.make_node("LRN", ["lrn_x"], ["lrn_y"], size=3),
            helper.make_node("MeanVarianceNormalization", ["mvn_x"], ["mvn_y"], axes=[0]),
            helper.make_node("MatMulInteger", ["mma", "mmb", "mma_zp", "mmb_zp"], ["mmi_y"]),
            helper.make_node("QLinearMatMul", ["qa", "qa_s", "qa_zp", "qb", "qb_s", "qb_zp", "qy_s", "qy_zp"], ["qmm_y"]),
            helper.make_node("ConvTranspose", ["deconv_x", "deconv_w"], ["deconv_y"], strides=[2, 2], pads=[0, 0, 0, 0]),
            helper.make_node("ConvInteger", ["conv_x", "conv_w", "conv_x_zp", "conv_w_zp"], ["conv_int_y"], pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("QLinearConv", ["qconv_x", "qconv_x_s", "qconv_x_zp", "qconv_w", "qconv_w_s", "qconv_w_zp", "qconv_y_s", "qconv_y_zp"], ["qconv_y"], pads=[0, 0, 0, 0]),
            helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_boxes", "iou"], ["selected"], center_point_box=0),
        ],
        "onnx17_independent_gap_ops",
        [
            helper.make_tensor_value_info("det_x", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("lrn_x", TensorProto.FLOAT, [1, 4, 1, 1]),
            helper.make_tensor_value_info("mvn_x", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("mma", TensorProto.UINT8, [2, 3]),
            helper.make_tensor_value_info("mmb", TensorProto.INT8, [3, 2]),
            helper.make_tensor_value_info("mma_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("mmb_zp", TensorProto.INT8, [2]),
            helper.make_tensor_value_info("qa", TensorProto.UINT8, [2, 3]),
            helper.make_tensor_value_info("qa_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qa_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qb", TensorProto.UINT8, [3, 2]),
            helper.make_tensor_value_info("qb_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qb_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qy_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qy_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("deconv_x", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("deconv_w", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("conv_x", TensorProto.UINT8, [1, 1, 3, 3]),
            helper.make_tensor_value_info("conv_w", TensorProto.INT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("conv_x_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("conv_w_zp", TensorProto.INT8, [1]),
            helper.make_tensor_value_info("qconv_x", TensorProto.UINT8, [1, 1, 3, 3]),
            helper.make_tensor_value_info("qconv_x_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_x_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qconv_w", TensorProto.UINT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("qconv_w_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_w_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qconv_y_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_y_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 3]),
            helper.make_tensor_value_info("max_boxes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("iou", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("det_y", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("lrn_y", TensorProto.FLOAT, [1, 4, 1, 1]),
            helper.make_tensor_value_info("mvn_y", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("mmi_y", TensorProto.INT32, [2, 2]),
            helper.make_tensor_value_info("qmm_y", TensorProto.UINT8, [2, 2]),
            helper.make_tensor_value_info("deconv_y", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("conv_int_y", TensorProto.INT32, [1, 1, 2, 2]),
            helper.make_tensor_value_info("qconv_y", TensorProto.UINT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("selected", TensorProto.INT64, [2, 3]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "Det", "LRN", "MeanVarianceNormalization", "MatMulInteger", "QLinearMatMul", "ConvTranspose",
        "ConvInteger", "QLinearConv", "NonMaxSuppression"
    ]

def test_onnx17_probability_loss_and_spectral_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    probabilities = Tensor(
        2, 3, dtype="float32",
        data=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    samples = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=4, seed=7.0).forward(probabilities)["tensor"]
    assert samples.dtype == "int64"
    np.testing.assert_array_equal(samples.data, np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64))
    assert Multinomial(["p"], ["y"], sample_size=3).forward_(Tensor_(2, 5, dtype="float32"))["tensor"].size == (2, 3)

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
    nll_none = NegativeLogLikelihoodLoss(["x", "target"], ["loss"], reduction="none", ignore_index=-1, dtype="float32").forward(
        log_probs, labels
    )["tensor"]
    np.testing.assert_allclose(nll_none.data, np.array([[0.1, 2.1], [1.2, 0.0]], dtype=np.float32))
    nll_mean = NegativeLogLikelihoodLoss(["x", "target", "w"], ["loss"], reduction="mean", ignore_index=-1, dtype="float32").forward(
        log_probs, labels, weights
    )["tensor"]
    expected_weighted = (0.1 * 1.0 + 2.1 * 3.0 + 1.2 * 2.0) / (1.0 + 3.0 + 2.0)
    np.testing.assert_allclose(nll_mean.data, np.array(expected_weighted, dtype=np.float32))
    assert NegativeLogLikelihoodLoss(["x", "target"], ["loss"], reduction="none").forward_(
        Tensor_(2, 3, 2, dtype="float32"), Tensor_(2, 2, dtype="int64")
    )["tensor"].size == (2, 2)

    scores = Tensor(2, 3, dtype="float32", data=np.array([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=np.float32))
    labels_1d = Tensor(2, dtype="int64", data=np.array([2, 0], dtype=np.int64))
    sce_loss, log_prob = SoftmaxCrossEntropyLoss(
        ["scores", "labels"], ["loss", "log_prob"], reduction="none", dtype="float32"
    ).forward(scores, labels_1d)["tensor"]
    shifted = scores.data - np.max(scores.data, axis=1, keepdims=True)
    expected_log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    np.testing.assert_allclose(log_prob.data, expected_log_prob, rtol=1e-6)
    np.testing.assert_allclose(sce_loss.data, -expected_log_prob[np.arange(2), labels_1d.data], rtol=1e-6)

    mel = MelWeightMatrix([], ["mel"], output_datatype=TensorProto.FLOAT).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(8, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(16000, dtype=np.int64)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(8000.0, dtype=np.float32)),
    )["tensor"]
    assert mel.size == (5, 3)
    assert np.max(mel.data) <= 1.0
    assert np.count_nonzero(mel.data) > 0

    signal = Tensor(1, 4, 1, dtype="float32", data=np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32))
    dft = DFT(["x"], ["y"], axis=1, onesided=1, dtype="float32").forward(
        signal, Tensor(dtype="int64", data=np.array(4, dtype=np.int64))
    )["tensor"]
    expected_fft = np.fft.fft(signal.data.squeeze(-1), n=4, axis=1)[:, :3]
    expected_dft = np.stack([expected_fft.real, expected_fft.imag], axis=-1).astype(np.float32)
    np.testing.assert_allclose(dft.data, expected_dft, rtol=1e-6, atol=1e-6)
    assert DFT(["x"], ["y"], axis=1, onesided=1).forward_(Tensor_(1, 4, 1, dtype="float32"))["tensor"].size == (1, 3, 2)

    stft = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="float32").forward(
        signal,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
        Tensor(2, dtype="float32", data=np.ones((2,), dtype=np.float32)),
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
    )["tensor"]
    expected_frames = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    expected_stft_complex = np.fft.fft(expected_frames.squeeze(-1), n=2, axis=-1)[..., :2]
    expected_stft = np.stack([expected_stft_complex.real, expected_stft_complex.imag], axis=-1).astype(np.float32)
    np.testing.assert_allclose(stft.data, expected_stft, rtol=1e-6, atol=1e-6)


    window_size = Tensor(dtype="int64", data=np.array(5, dtype=np.int64))
    assert HannWindow(["size"], ["hann"]).forward_(window_size)["tensor"].size == (5,)
    hamming_shape = HammingWindow(["size"], ["hamming"], output_datatype=TensorProto.DOUBLE).forward_(window_size)["tensor"]
    assert hamming_shape.size == (5,)
    assert hamming_shape.dtype == "float64"
    blackman_shape = BlackmanWindow(["size"], ["blackman"], output_datatype=TensorProto.INT32).forward_(window_size)["tensor"]
    assert blackman_shape.size == (5,)
    assert blackman_shape.dtype == "int32"
    from onnx.reference import ReferenceEvaluator

    for op_name, op_cls in [("HannWindow", HannWindow), ("HammingWindow", HammingWindow), ("BlackmanWindow", BlackmanWindow)]:
        window_graph = helper.make_graph(
            [helper.make_node(op_name, ["size"], ["y"], periodic=0, output_datatype=TensorProto.DOUBLE)],
            f"{op_name}_ref",
            [helper.make_tensor_value_info("size", TensorProto.INT64, [])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [5])],
        )
        window_model = helper.make_model(window_graph, opset_imports=[helper.make_opsetid("", 17)])
        expected_window = ReferenceEvaluator(window_model).run(None, {"size": np.array(5, dtype=np.int64)})[0]
        actual_window = op_cls(["size"], ["y"], periodic=0, output_datatype=TensorProto.DOUBLE).forward(window_size)["tensor"]
        np.testing.assert_allclose(actual_window.data, expected_window, rtol=1e-12, atol=1e-12)

    uint_window = HannWindow(["size"], ["hann"], output_datatype=TensorProto.UINT32).forward(window_size)["tensor"]
    assert uint_window.dtype == "uint32"
    assert uint_window.data.dtype == np.uint32

    model_path = tmp_path / "onnx17_probability_loss_spectral_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Multinomial", ["prob"], ["sample"], dtype=TensorProto.INT64, sample_size=2, seed=1.0),
            helper.make_node("NegativeLogLikelihoodLoss", ["log_probs", "labels", "class_weights"], ["nll"], reduction="mean", ignore_index=-1),
            helper.make_node("SoftmaxCrossEntropyLoss", ["scores", "labels_1d"], ["sce", "logp"], reduction="none"),
            helper.make_node("MelWeightMatrix", ["num_mel", "dft_len", "sample_rate", "lower", "upper"], ["mel"]),
            helper.make_node("DFT", ["signal", "dft_len"], ["dft"], axis=1, onesided=1),
            helper.make_node("STFT", ["signal", "frame_step", "window", "frame_length"], ["stft"], onesided=1),
            helper.make_node("HannWindow", ["window_size"], ["hann"], periodic=1),
            helper.make_node("HammingWindow", ["window_size"], ["hamming"], output_datatype=TensorProto.DOUBLE),
            helper.make_node("BlackmanWindow", ["window_size"], ["blackman"], output_datatype=TensorProto.INT32),
        ],
        "onnx17_probability_loss_spectral_ops",
        [
            helper.make_tensor_value_info("prob", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [2, 3, 2]),
            helper.make_tensor_value_info("labels", TensorProto.INT64, [2, 2]),
            helper.make_tensor_value_info("class_weights", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("labels_1d", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("num_mel", TensorProto.INT64, []),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
            helper.make_tensor_value_info("sample_rate", TensorProto.INT64, []),
            helper.make_tensor_value_info("lower", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("upper", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("signal", TensorProto.FLOAT, [1, 4, 1]),
            helper.make_tensor_value_info("frame_step", TensorProto.INT64, []),
            helper.make_tensor_value_info("window", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("frame_length", TensorProto.INT64, []),
            helper.make_tensor_value_info("window_size", TensorProto.INT64, []),
        ],
        [
            helper.make_tensor_value_info("sample", TensorProto.INT64, [2, 2]),
            helper.make_tensor_value_info("nll", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("sce", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("logp", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("mel", TensorProto.FLOAT, [5, 3]),
            helper.make_tensor_value_info("dft", TensorProto.FLOAT, [1, 3, 2]),
            helper.make_tensor_value_info("stft", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("hann", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("hamming", TensorProto.DOUBLE, [5]),
            helper.make_tensor_value_info("blackman", TensorProto.INT32, [5]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "Multinomial", "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss", "MelWeightMatrix",
        "DFT", "STFT", "HannWindow", "HammingWindow", "BlackmanWindow"
    ]


def test_spectral_bfloat16_python_fallback_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    signal_values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    signal = Tensor(*signal_values.shape, dtype="bfloat16", data=_bf16_bits(signal_values))
    dft_length = Tensor(dtype="int64", data=np.array(4, dtype=np.int64))

    dft = DFT(["x", "dft_len"], ["y"], axis=1, onesided=1, dtype="bfloat16").forward(
        signal, dft_length
    )["tensor"]
    expected_fft = np.fft.fft(signal_values.squeeze(-1), n=4, axis=1)[:, :3]
    expected_dft = np.stack([expected_fft.real, expected_fft.imag], axis=-1).astype(np.float32)
    np.testing.assert_array_equal(dft.data, _bf16_bits(expected_dft))

    stft = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="bfloat16").forward(
        signal,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
        Tensor(2, dtype="bfloat16", data=_bf16_bits(np.ones((2,), dtype=np.float32))),
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
    )["tensor"]
    expected_frames = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    expected_stft_complex = np.fft.fft(expected_frames.squeeze(-1), n=2, axis=-1)[..., :2]
    expected_stft = np.stack([expected_stft_complex.real, expected_stft_complex.imag], axis=-1).astype(np.float32)
    np.testing.assert_array_equal(stft.data, _bf16_bits(expected_stft))


def test_onnx17_recurrent_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    x = Tensor(2, 1, 1, dtype="float32", data=np.array([[[1.0]], [[2.0]]], dtype=np.float32))
    rnn_w = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    rnn_r = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    rnn_y, rnn_h = RNN(["x", "w", "r"], ["y", "yh"], hidden_size=1, dtype="float32").forward(x, rnn_w, rnn_r)["tensor"]
    h0 = np.tanh(1.0)
    h1 = np.tanh(2.0 + h0)
    np.testing.assert_allclose(rnn_y.data, np.array([[[[h0]]], [[[h1]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(rnn_h.data, np.array([[[h1]]], dtype=np.float32), rtol=1e-6)
    assert RNN(["x", "w", "r"], ["y", "yh"], hidden_size=1).forward_(
        Tensor_(2, 1, 1, dtype="float32"), Tensor_(1, 1, 1, dtype="float32"), Tensor_(1, 1, 1, dtype="float32")
    )["tensor"][0].size == (2, 1, 1, 1)

    reverse_x_data = np.arange(8, dtype=np.float32).reshape(4, 2, 1) / 10.0
    reverse_x = Tensor(*reverse_x_data.shape, dtype="float32", data=reverse_x_data)
    reverse_w = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    reverse_r = Tensor(1, 1, 1, dtype="float32", data=np.full((1, 1, 1), 0.5, dtype=np.float32))
    sequence_lens = Tensor(2, dtype="int64", data=np.array([2, 4], dtype=np.int64))
    reverse_y, reverse_h = RNN(
        ["x", "w", "r", "", "sequence_lens"],
        ["y", "yh"],
        hidden_size=1,
        direction="reverse",
        dtype="float32",
    ).forward(reverse_x, reverse_w, reverse_r, None, sequence_lens)["tensor"]
    expected_reverse = np.zeros((4, 1, 2, 1), dtype=np.float32)
    b0_t1 = np.tanh(reverse_x_data[1, 0, 0])
    b0_t0 = np.tanh(reverse_x_data[0, 0, 0] + 0.5 * b0_t1)
    b1_t3 = np.tanh(reverse_x_data[3, 1, 0])
    b1_t2 = np.tanh(reverse_x_data[2, 1, 0] + 0.5 * b1_t3)
    b1_t1 = np.tanh(reverse_x_data[1, 1, 0] + 0.5 * b1_t2)
    b1_t0 = np.tanh(reverse_x_data[0, 1, 0] + 0.5 * b1_t1)
    expected_reverse[1, 0, 0, 0] = b0_t1
    expected_reverse[0, 0, 0, 0] = b0_t0
    expected_reverse[3, 0, 1, 0] = b1_t3
    expected_reverse[2, 0, 1, 0] = b1_t2
    expected_reverse[1, 0, 1, 0] = b1_t1
    expected_reverse[0, 0, 1, 0] = b1_t0
    np.testing.assert_allclose(reverse_y.data, expected_reverse, rtol=1e-6)
    np.testing.assert_allclose(reverse_h.data, np.array([[[b0_t0], [b1_t0]]], dtype=np.float32), rtol=1e-6)

    one_step = Tensor(1, 1, 1, dtype="float32", data=np.array([[[1.0]]], dtype=np.float32))
    gru_w = Tensor(1, 3, 1, dtype="float32", data=np.array([[[0.0], [0.0], [1.0]]], dtype=np.float32))
    gru_r = Tensor(1, 3, 1, dtype="float32", data=np.zeros((1, 3, 1), dtype=np.float32))
    gru_y, gru_h = GRU(["x", "w", "r"], ["y", "yh"], hidden_size=1, dtype="float32").forward(one_step, gru_w, gru_r)["tensor"]
    expected_gru = 0.5 * np.tanh(1.0)
    np.testing.assert_allclose(gru_y.data, np.array([[[[expected_gru]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(gru_h.data, np.array([[[expected_gru]]], dtype=np.float32), rtol=1e-6)

    lstm_w = Tensor(1, 4, 1, dtype="float32", data=np.array([[[0.0], [0.0], [0.0], [1.0]]], dtype=np.float32))
    lstm_r = Tensor(1, 4, 1, dtype="float32", data=np.zeros((1, 4, 1), dtype=np.float32))
    lstm_y, lstm_h, lstm_c = LSTM(["x", "w", "r"], ["y", "yh", "yc"], hidden_size=1, dtype="float32").forward(
        one_step, lstm_w, lstm_r
    )["tensor"]
    expected_c = 0.5 * np.tanh(1.0)
    expected_h = 0.5 * np.tanh(expected_c)
    np.testing.assert_allclose(lstm_y.data, np.array([[[[expected_h]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(lstm_h.data, np.array([[[expected_h]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(lstm_c.data, np.array([[[expected_c]]], dtype=np.float32), rtol=1e-6)

    model_path = tmp_path / "onnx17_recurrent_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("RNN", ["rnn_x", "rnn_w", "rnn_r"], ["rnn_y", "rnn_h"], hidden_size=1),
            helper.make_node("GRU", ["gru_x", "gru_w", "gru_r"], ["gru_y", "gru_h"], hidden_size=1),
            helper.make_node("LSTM", ["lstm_x", "lstm_w", "lstm_r"], ["lstm_y", "lstm_h", "lstm_c"], hidden_size=1),
        ],
        "onnx17_recurrent_ops",
        [
            helper.make_tensor_value_info("rnn_x", TensorProto.FLOAT, [2, 1, 1]),
            helper.make_tensor_value_info("rnn_w", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("rnn_r", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_x", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_w", TensorProto.FLOAT, [1, 3, 1]),
            helper.make_tensor_value_info("gru_r", TensorProto.FLOAT, [1, 3, 1]),
            helper.make_tensor_value_info("lstm_x", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_w", TensorProto.FLOAT, [1, 4, 1]),
            helper.make_tensor_value_info("lstm_r", TensorProto.FLOAT, [1, 4, 1]),
        ],
        [
            helper.make_tensor_value_info("rnn_y", TensorProto.FLOAT, [2, 1, 1, 1]),
            helper.make_tensor_value_info("rnn_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_y", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("gru_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_y", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("lstm_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_c", TensorProto.FLOAT, [1, 1, 1]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["RNN", "GRU", "LSTM"]

def test_onnx17_unpool_and_string_normalizer_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    image = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    grid = Tensor(
        1, 2, 2, 2, dtype="float32",
        data=np.array([[[[-1.0, -1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]]]], dtype=np.float32),
    )
    sampled = GridSample(["x", "grid"], ["y"], mode="bilinear", align_corners=1, dtype="float32").forward(image, grid)["tensor"]
    np.testing.assert_allclose(sampled.data, np.array([[[[1.0, 4.0], [2.5, 2.0]]]], dtype=np.float32))
    assert GridSample(["x", "grid"], ["y"]).forward_(Tensor_(1, 3, 4, 5, dtype="float32"), Tensor_(1, 6, 7, 2, dtype="float32"))["tensor"].size == (1, 3, 6, 7)

    roi_input = Tensor(1, 1, 4, 4, dtype="float32", data=np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    max_roi = MaxRoiPool(["x", "rois"], ["y"], pooled_shape=[2, 2], spatial_scale=1.0, dtype="float32").forward(
        roi_input, Tensor(1, 5, dtype="float32", data=np.array([[0, 0, 0, 3, 3]], dtype=np.float32))
    )["tensor"]
    np.testing.assert_array_equal(max_roi.data, np.array([[[[5.0, 7.0], [13.0, 15.0]]]], dtype=np.float32))

    aligned = RoiAlign(
        ["x", "rois", "batch"], ["y"], output_height=1, output_width=1, sampling_ratio=1, dtype="float32"
    ).forward(
        roi_input,
        Tensor(1, 4, dtype="float32", data=np.array([[0, 0, 3, 3]], dtype=np.float32)),
        Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_allclose(aligned.data, np.array([[[[5.0]]]], dtype=np.float32), rtol=1e-6)
    aligned_max = RoiAlign(
        ["x", "rois", "batch"], ["y"], output_height=1, output_width=1, sampling_ratio=2, mode="max", dtype="float32"
    ).forward(
        roi_input,
        Tensor(1, 4, dtype="float32", data=np.array([[0, 0, 3, 3]], dtype=np.float32)),
        Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_allclose(aligned_max.data, np.array([[[[5.625]]]], dtype=np.float32), rtol=1e-6)

    pooled = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    indices = Tensor(1, 1, 2, 2, dtype="int64", data=np.array([[[[5, 7], [13, 15]]]], dtype=np.int64))
    unpooled = MaxUnpool(["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2], dtype="float32").forward(
        pooled, indices
    )["tensor"]
    expected_unpool = np.zeros((1, 1, 4, 4), dtype=np.float32)
    expected_unpool.reshape(-1)[[5, 7, 13, 15]] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(unpooled.data, expected_unpool)
    assert MaxUnpool(["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2]).forward_(
        Tensor_(1, 1, 2, 2, dtype="float32"), Tensor_(1, 1, 2, 2, dtype="int64")
    )["tensor"].size == (1, 1, 4, 4)

    strings = Tensor(3, dtype="string", data=np.array(["The Café", "stop WORD", ""], dtype=np.str_))
    normalized = StringNormalizer(
        ["x"], ["y"], case_change_action="LOWER", is_case_sensitive=0, stopwords=["the", "stop"]
    ).forward(strings)["tensor"]
    np.testing.assert_array_equal(normalized.data, np.array(["cafe", "word"], dtype=np.str_))

    matrix_strings = Tensor(1, 3, dtype="string", data=np.array([["Keep", "THE item", ""]], dtype=np.str_))
    normalized_matrix = StringNormalizer(
        ["x"], ["y"], case_change_action="LOWER", stopwords=["the"]
    ).forward(matrix_strings)["tensor"]
    assert normalized_matrix.size == (1, 2)
    np.testing.assert_array_equal(normalized_matrix.data, np.array([["keep", "item"]], dtype=np.str_))

    tfidf = TfIdfVectorizer(
        ["tokens"],
        ["features"],
        mode="TFIDF",
        ngram_counts=[0, 0],
        ngram_indexes=[1, 0],
        max_skip_count=0,
        min_gram_length=2,
        max_gram_length=2,
        pool_int64s=[94, 17, 17, 36],
        weights=[0.5, 2.0],
    ).forward(Tensor(3, dtype="int64", data=np.array([94, 17, 36], dtype=np.int64)))["tensor"]
    np.testing.assert_array_equal(tfidf.data, np.array([0.5, 2.0], dtype=np.float32))

    string_tfidf = TfIdfVectorizer(
        ["tokens"],
        ["features"],
        mode="TF",
        ngram_counts=[0],
        ngram_indexes=[0, 1],
        max_skip_count=0,
        min_gram_length=1,
        max_gram_length=1,
        pool_strings=["a", "b"],
    ).forward(Tensor(2, 2, dtype="string", data=np.array([["a", "x"], ["b", "a"]], dtype=np.str_)))["tensor"]
    np.testing.assert_array_equal(string_tfidf.data, np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))

    model_path = tmp_path / "onnx17_unpool_string_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("GridSample", ["image", "grid"], ["sampled"], mode="bilinear", align_corners=1),
            helper.make_node("MaxRoiPool", ["roi_input", "max_rois"], ["max_roi"], pooled_shape=[2, 2]),
            helper.make_node("RoiAlign", ["roi_input", "align_rois", "batch_indices"], ["aligned"], output_height=1, output_width=1, sampling_ratio=1),
            helper.make_node("MaxUnpool", ["pooled", "indices"], ["unpooled"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("StringNormalizer", ["tokens"], ["normalized"], case_change_action="LOWER", stopwords=["the", "stop"]),
            helper.make_node(
                "TfIdfVectorizer",
                ["ids"],
                ["tfidf"],
                mode="TF",
                ngram_counts=[0, 0],
                ngram_indexes=[1, 0],
                max_skip_count=0,
                min_gram_length=2,
                max_gram_length=2,
                pool_int64s=[94, 17, 17, 36],
            ),
        ],
        "onnx17_unpool_string_ops",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("roi_input", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("max_rois", TensorProto.FLOAT, [1, 5]),
            helper.make_tensor_value_info("align_rois", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1, 2, 2]),
            helper.make_tensor_value_info("tokens", TensorProto.STRING, [3]),
            helper.make_tensor_value_info("ids", TensorProto.INT64, [3]),
        ],
        [
            helper.make_tensor_value_info("sampled", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("max_roi", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("aligned", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("unpooled", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("normalized", TensorProto.STRING, [2]),
            helper.make_tensor_value_info("tfidf", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "GridSample", "MaxRoiPool", "RoiAlign", "MaxUnpool", "StringNormalizer", "TfIdfVectorizer"
    ]

def test_roi_pool_ops_use_c_backend_against_reference():
    x_data = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    roi_input = Tensor(*x_data.shape, dtype="float32", data=x_data)

    max_rois_data = np.array(
        [
            [0, 0, 0, 3, 3],
            [1, 1, 1, 3, 3],
        ],
        dtype=np.float32,
    )
    max_rois = Tensor(*max_rois_data.shape, dtype="float32", data=max_rois_data)
    pooled = MaxRoiPool(["x", "rois"], ["y"], pooled_shape=[2, 2], dtype="float32").forward(
        roi_input, max_rois
    )["tensor"]
    np.testing.assert_array_equal(
        pooled.data,
        np.array(
            [
                [[[5.0, 7.0], [13.0, 15.0]]],
                [[[26.0, 27.0], [30.0, 31.0]]],
            ],
            dtype=np.float32,
        ),
    )

    align_rois_data = np.array([[0, 0, 3, 3], [1, 1, 3, 3]], dtype=np.float32)
    batch_indices_data = np.array([0, 1], dtype=np.int64)
    align_rois = Tensor(*align_rois_data.shape, dtype="float32", data=align_rois_data)
    batch_indices = Tensor(*batch_indices_data.shape, dtype="int64", data=batch_indices_data)

    aligned = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=2,
        output_width=2,
        sampling_ratio=2,
        mode="avg",
        dtype="float32",
    ).forward(roi_input, align_rois, batch_indices)["tensor"]

    from onnx.reference import ReferenceEvaluator

    graph = helper.make_graph(
        [
            helper.make_node(
                "RoiAlign",
                ["x", "rois", "batch"],
                ["y"],
                output_height=2,
                output_width=2,
                sampling_ratio=2,
                mode="avg",
            )
        ],
        "roi_align_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_data.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(align_rois_data.shape)),
            helper.make_tensor_value_info("batch", TensorProto.INT64, list(batch_indices_data.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 2, 2])],
    )
    ref = ReferenceEvaluator(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    expected = ref.run(None, {"x": x_data, "rois": align_rois_data, "batch": batch_indices_data})[0]
    np.testing.assert_allclose(aligned.data, expected, rtol=1e-6, atol=1e-6)

    aligned_max = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=1,
        output_width=1,
        sampling_ratio=2,
        mode="max",
        coordinate_transformation_mode="output_half_pixel",
        dtype="float32",
    ).forward(roi_input, align_rois, batch_indices)["tensor"]
    graph_max = helper.make_graph(
        [
            helper.make_node(
                "RoiAlign",
                ["x", "rois", "batch"],
                ["y"],
                output_height=1,
                output_width=1,
                sampling_ratio=2,
                mode="max",
                coordinate_transformation_mode="output_half_pixel",
            )
        ],
        "roi_align_max_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_data.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(align_rois_data.shape)),
            helper.make_tensor_value_info("batch", TensorProto.INT64, list(batch_indices_data.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, 1])],
    )
    ref_max = ReferenceEvaluator(helper.make_model(graph_max, opset_imports=[helper.make_opsetid("", 17)]))
    expected_max = ref_max.run(None, {"x": x_data, "rois": align_rois_data, "batch": batch_indices_data})[0]
    np.testing.assert_allclose(aligned_max.data, expected_max, rtol=1e-6, atol=1e-6)
