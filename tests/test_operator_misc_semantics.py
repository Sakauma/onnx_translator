# /**
#   ******************************************************************************
#   * @file        test_operator_misc_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       验证未进入 CUDA 数值计划的普通算子、窗口算子和仓库扩展算子的官方语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import math

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    Binarizer,
    BitShift,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    BlackmanWindow,
    Cast,
    CastLike,
    Ceil,
    Det,
    Dropout,
    DynamicQuantizeLinear,
    Gelu,
    GridSample,
    GroupNormalization,
    HammingWindow,
    HannWindow,
    Hardmax,
    IsInf,
    LogSoftmax,
    MelWeightMatrix,
    Mean,
    Mish,
    NegativeLogLikelihoodLoss,
    NonMaxSuppression,
    PRelu,
    Reciprocal,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceSumSquare,
    ScatterElements,
    SoftmaxCrossEntropyLoss,
    Tril,
    Triu,
    Unique,
)


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，返回指定节点的官方参考输出。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos=None, opset=17):
    output_protos = output_protos or [protos[0]] * len(output_shapes)
    input_names = [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, proto, list(np.asarray(value).shape))
            for name, proto, value in zip(input_names, protos, inputs)
        ],
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 比较 Tensor 输出和参考数组，浮点类型使用容差，整数和布尔类型使用精确比较。
def _assert_tensor_matches(actual, expected, rtol=1e-4, atol=1e-4):
    if np.issubdtype(np.asarray(expected).dtype, np.floating):
        np.testing.assert_allclose(actual.data, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual.data, expected)


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


# 将 bfloat16 的 uint16 位模式解码成 float32，用于按数值容差比较输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 按 ONNX schema 公式独立计算 reduce 系列输出，覆盖本地 reference 的旧版 axes 签名问题。
def _reduce_formula(op_name, x, axes):
    data = x.astype(np.float32)
    axis = tuple(axes.tolist())
    if op_name == "ReduceL1":
        return np.sum(np.abs(data), axis=axis)
    if op_name == "ReduceL2":
        return np.sqrt(np.sum(data * data, axis=axis))
    if op_name == "ReduceLogSum":
        return np.log(np.sum(data, axis=axis))
    if op_name == "ReduceLogSumExp":
        return np.log(np.sum(np.exp(data), axis=axis))
    if op_name == "ReduceSumSquare":
        return np.sum(data * data, axis=axis)
    raise AssertionError(f"unsupported reduce op {op_name}")


# 按 GroupNormalization 定义独立计算分组均值、方差和仿射变换。
def _group_norm_formula(x, scale, bias, num_groups, epsilon):
    batch, channels, *spatial = x.shape
    grouped = x.reshape(batch, num_groups, channels // num_groups, *spatial)
    axes = tuple(range(2, grouped.ndim))
    mean = np.mean(grouped, axis=axes, keepdims=True)
    var = np.mean((grouped - mean) ** 2, axis=axes, keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(var + epsilon)).reshape(x.shape)
    affine_shape = (1, channels) + (1,) * len(spatial)
    return normalized * scale.reshape(affine_shape) + bias.reshape(affine_shape)


# 按 ScatterElements schema 独立计算 none/add/mul 三种更新语义。
def _scatter_elements_formula(data, indices, updates, axis, reduction):
    output = data.copy()
    axis = axis if axis >= 0 else axis + data.ndim
    for update_index in np.ndindex(updates.shape):
        target_index = list(update_index)
        idx = int(indices[update_index])
        if idx < 0:
            idx += data.shape[axis]
        target_index[axis] = idx
        target_index = tuple(target_index)
        if reduction == "none":
            output[target_index] = updates[update_index]
        elif reduction == "add":
            output[target_index] += updates[update_index]
        elif reduction == "mul":
            output[target_index] *= updates[update_index]
        else:
            raise AssertionError(f"unsupported ScatterElements reduction {reduction}")
    return output


# 按 Unique(axis=None) 的 ONNX 输出约定计算 values/indices/inverse/counts。
def _unique_flat_formula(data, sorted_output):
    flat = data.reshape(-1)
    values, indices, inverse, counts = np.unique(flat, return_index=True, return_inverse=True, return_counts=True)
    if not sorted_output:
        order = np.argsort(indices)
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        values = values[order]
        indices = indices[order]
        inverse = remap[inverse]
        counts = counts[order]
    return values, indices.astype(np.int64), inverse.astype(np.int64), counts.astype(np.int64)


# 验证一批普通官方算子在低精度、整数转换和布尔输出场景下与 ONNX reference 对齐。
def test_c_backend_misc_official_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    prelu_x = np.array([[-2.0, -0.5, 0.0, 1.5]], dtype=np.float16)
    prelu_slope = np.array([0.25], dtype=np.float16)
    _assert_tensor_matches(
        PRelu(["x", "slope"], ["y"], dtype="float16").forward(_tensor(prelu_x, "float16"), _tensor(prelu_slope, "float16"))["tensor"],
        _onnx_reference("PRelu", [prelu_x, prelu_slope], [TensorProto.FLOAT16, TensorProto.FLOAT16], {}, [prelu_x.shape])[0],
        rtol=2e-3,
        atol=2e-3,
    )

    unary_x = np.array([-0.2, 0.2, 2.5], dtype=np.float64)
    _assert_tensor_matches(
        Reciprocal(["x"], ["y"], dtype="float64").forward(_tensor(unary_x, "float64"))["tensor"],
        _onnx_reference("Reciprocal", [unary_x], [TensorProto.DOUBLE], {}, [unary_x.shape])[0],
        rtol=1e-12,
        atol=1e-12,
    )
    _assert_tensor_matches(
        Ceil(["x"], ["y"], dtype="float64").forward(_tensor(unary_x, "float64"))["tensor"],
        _onnx_reference("Ceil", [unary_x], [TensorProto.DOUBLE], {}, [unary_x.shape])[0],
        rtol=1e-12,
        atol=1e-12,
    )

    logits = np.array([[1.0, 3.0, 2.0], [4.0, 4.0, 1.0]], dtype=np.float32)
    _assert_tensor_matches(
        Hardmax(["x"], ["y"], axis=1, dtype="float32").forward(_tensor(logits, "float32"))["tensor"],
        _onnx_reference("Hardmax", [logits], [TensorProto.FLOAT], {"axis": 1}, [logits.shape])[0],
    )
    _assert_tensor_matches(
        LogSoftmax(["x"], ["y"], axis=1, dtype="float32").forward(_tensor(logits, "float32"))["tensor"],
        _onnx_reference("LogSoftmax", [logits], [TensorProto.FLOAT], {"axis": 1}, [logits.shape])[0],
    )

    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    right = np.array([[4.0, 1.0, 0.0], [5.0, 2.0, 1.0]], dtype=np.float16)
    _assert_tensor_matches(
        Mean(["a", "b"], ["y"], dtype="float16").forward(_tensor(left, "float16"), _tensor(right, "float16"))["tensor"],
        _onnx_reference("Mean", [left, right], [TensorProto.FLOAT16, TensorProto.FLOAT16], {}, [left.shape])[0],
        rtol=2e-3,
        atol=2e-3,
    )
    _assert_tensor_matches(
        Sum(["a", "b"], ["y"], dtype="float16").forward(_tensor(left, "float16"), _tensor(right, "float16"))["tensor"],
        _onnx_reference("Sum", [left, right], [TensorProto.FLOAT16, TensorProto.FLOAT16], {}, [left.shape])[0],
        rtol=2e-3,
        atol=2e-3,
    )

    ints = np.array([-2, 0, 3], dtype=np.int32)
    _assert_tensor_matches(
        Cast(["x"], ["y"], dtype="float16").forward(_tensor(ints, "int32"))["tensor"],
        _onnx_reference("Cast", [ints], [TensorProto.INT32], {"to": TensorProto.FLOAT16}, [ints.shape], [TensorProto.FLOAT16])[0],
    )
    cast_target = np.array([1.0], dtype=np.float16)
    _assert_tensor_matches(
        CastLike(["x", "target"], ["y"]).forward(_tensor(ints, "int32"), _tensor(cast_target, "float16"))["tensor"],
        _onnx_reference("CastLike", [ints, cast_target], [TensorProto.INT32, TensorProto.FLOAT16], {}, [ints.shape], [TensorProto.FLOAT16])[0],
    )

    inf_values = np.array([-np.inf, -1.0, 0.0, np.inf], dtype=np.float32)
    _assert_tensor_matches(
        IsInf(["x"], ["y"], detect_negative=0, detect_positive=1).forward(_tensor(inf_values, "float32"))["tensor"],
        _onnx_reference("IsInf", [inf_values], [TensorProto.FLOAT], {"detect_negative": 0, "detect_positive": 1}, [inf_values.shape], [TensorProto.BOOL])[0],
    )

    image = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    grid = np.array([[[[-1.0, -1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]]]], dtype=np.float32)
    _assert_tensor_matches(
        GridSample(["x", "grid"], ["y"], mode="linear", padding_mode="zeros", align_corners=1, dtype="float32").forward(
            _tensor(image, "float32"), _tensor(grid, "float32")
        )["tensor"],
        _onnx_reference(
            "GridSample",
            [image, grid],
            [TensorProto.FLOAT, TensorProto.FLOAT],
            {"mode": "linear", "padding_mode": "zeros", "align_corners": 1},
            [(1, 1, 2, 2)],
        )[0],
    )

    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int64)
    updates = np.array([[9, 8], [7, 6]], dtype=np.float32)
    for reduction in ["none", "add", "mul"]:
        actual = ScatterElements(["data", "indices", "updates"], ["y"], axis=1, reduction=reduction, dtype="float32").forward(
            _tensor(data, "float32"), _tensor(indices, "int64"), _tensor(updates, "float32")
        )["tensor"]
        _assert_tensor_matches(actual, _scatter_elements_formula(data, indices, updates, axis=1, reduction=reduction))


# 验证 DynamicQuantizeLinear 三个输出的数值和标量形状均与 ONNX reference 一致。
def test_c_backend_dynamic_quantize_linear_outputs_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    actual = DynamicQuantizeLinear(["x"], ["y", "scale", "zero_point"]).forward(_tensor(x, "float32"))["tensor"]
    expected = _onnx_reference(
        "DynamicQuantizeLinear",
        [x],
        [TensorProto.FLOAT],
        {},
        [x.shape, (), ()],
        [TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8],
    )
    for actual_tensor, expected_value in zip(actual, expected):
        assert actual_tensor.data.shape == np.asarray(expected_value).shape
        _assert_tensor_matches(actual_tensor, expected_value, rtol=1e-7, atol=1e-7)


# 验证 Cast/CastLike 的 bfloat16 位存储和无符号整型 fallback 行为。
def test_cast_ops_cover_bfloat16_bit_storage_and_unsigned_dtypes():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([-2.25, -0.5, 0.0, 1.5, 3.75], dtype=np.float32)
    cast_to_bf16 = Cast(["x"], ["y"], dtype="bfloat16").forward(_tensor(values, "float32"))["tensor"]
    np.testing.assert_array_equal(cast_to_bf16.data, _bf16_bits(values))

    cast_back = Cast(["x"], ["y"], dtype="float32").forward(cast_to_bf16)["tensor"]
    np.testing.assert_allclose(cast_back.data, _bf16_to_float32(_bf16_bits(values)), rtol=0.0, atol=0.0)

    target = _tensor(_bf16_bits(np.array([0.0], dtype=np.float32)), "bfloat16")
    cast_like = CastLike(["x", "target"], ["y"]).forward(_tensor(values, "float32"), target)["tensor"]
    assert cast_like.dtype == "bfloat16"
    np.testing.assert_array_equal(cast_like.data, _bf16_bits(values))

    uint_values = np.array([0, 3, 4294967295], dtype=np.uint32)
    cast_uint32 = Cast(["x"], ["y"], dtype="uint32").forward(_tensor(uint_values.astype(np.int64), "int64"))["tensor"]
    assert cast_uint32.dtype == "uint32"
    np.testing.assert_array_equal(cast_uint32.data, uint_values)


# 验证 Dropout 推理模式的输出和 mask 与 ONNX reference 对齐。
def test_c_backend_dropout_inference_mode_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    ratio = np.array(0.25, dtype=np.float32)
    training_mode = np.array(False, dtype=np.bool_)
    actual = Dropout(["x", "ratio", "training_mode"], ["y", "mask"]).forward(
        _tensor(x, "float32"), _tensor(ratio, "float32"), _tensor(training_mode, "bool")
    )["tensor"]
    expected = _onnx_reference(
        "Dropout",
        [x, ratio, training_mode],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.BOOL],
        {},
        [x.shape, x.shape],
        [TensorProto.FLOAT, TensorProto.BOOL],
    )
    _assert_tensor_matches(actual[0], expected[0])
    _assert_tensor_matches(actual[1], expected[1])


# 验证窗口算子使用 ONNX17 官方公式和输出 dtype。
def test_c_backend_window_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    size = np.array(5, dtype=np.int64)
    for periodic in [0, 1]:
        for op_name, op_cls in [("HannWindow", HannWindow), ("HammingWindow", HammingWindow), ("BlackmanWindow", BlackmanWindow)]:
            actual = op_cls(["size"], ["y"], periodic=periodic, output_datatype=TensorProto.DOUBLE).forward(_tensor(size, "int64"))["tensor"]
            expected = _onnx_reference(
                op_name,
                [size],
                [TensorProto.INT64],
                {"periodic": periodic, "output_datatype": TensorProto.DOUBLE},
                [(5,)],
                [TensorProto.DOUBLE],
            )[0]
            assert actual.dtype == "float64"
            _assert_tensor_matches(actual, expected, rtol=1e-12, atol=1e-12)


# 验证 reduce 系列在 float16 下按 ONNX schema 公式归约运行时 axes 输入。
@pytest.mark.parametrize(
    "op_cls,op_name",
    [
        (ReduceL1, "ReduceL1"),
        (ReduceL2, "ReduceL2"),
        (ReduceLogSum, "ReduceLogSum"),
        (ReduceLogSumExp, "ReduceLogSumExp"),
        (ReduceSumSquare, "ReduceSumSquare"),
    ],
)
def test_c_backend_reduce_formula_ops_float16_match_onnx_schema(op_cls, op_name):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10.0 + 0.1).astype(np.float16)
    axes = np.array([1, 2], dtype=np.int64)
    actual = op_cls(["x", "axes"], ["y"], keepdims=0, dtype="float16").forward(_tensor(x, "float16"), _tensor(axes, "int64"))["tensor"]
    expected = _reduce_formula(op_name, x, axes).astype(np.float16)
    _assert_tensor_matches(actual, expected, rtol=2e-3, atol=2e-3)


# 验证 bitwise/bitshift 整数语义与 NumPy 等价公式一致。
def test_c_backend_bitwise_ops_match_integer_formulas():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.array([[1, 2, 3]], dtype=np.int32)
    right = np.array([[3], [4]], dtype=np.int32)
    for op_cls, np_func in [(BitwiseAnd, np.bitwise_and), (BitwiseOr, np.bitwise_or), (BitwiseXor, np.bitwise_xor)]:
        actual = op_cls(["a", "b"], ["y"], dtype="int32").forward(_tensor(left, "int32"), _tensor(right, "int32"))["tensor"]
        _assert_tensor_matches(actual, np_func(left, right))

    actual_not = BitwiseNot(["x"], ["y"], dtype="int32").forward(_tensor(left, "int32"))["tensor"]
    _assert_tensor_matches(actual_not, np.bitwise_not(left))

    shift_left = np.array([1, 2], dtype=np.int32)
    shift_right = np.array([1, 2], dtype=np.int32)
    actual_shift = BitShift(["a", "b"], ["y"], direction="LEFT", dtype="int32").forward(
        _tensor(shift_left, "int32"), _tensor(shift_right, "int32")
    )["tensor"]
    _assert_tensor_matches(actual_shift, np.left_shift(shift_left, shift_right))
    actual_shift = BitShift(["a", "b"], ["y"], direction="RIGHT", dtype="int32").forward(
        _tensor(np.array([8, 16], dtype=np.int32), "int32"), _tensor(shift_right, "int32")
    )["tensor"]
    _assert_tensor_matches(actual_shift, np.right_shift(np.array([8, 16], dtype=np.int32), shift_right))


# 验证仓库扩展激活和 GroupNormalization 的独立公式语义，包括 bfloat16 位存储路径。
def test_c_backend_extra_formula_ops_match_expected_mixed_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
    bf16_input = Tensor(*values.shape, dtype="bfloat16", data=_bf16_bits(values))
    gelu_expected = 0.5 * values * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))
    mish_expected = values * np.tanh(np.log1p(np.exp(values)))
    gelu = Gelu(["x"], ["y"], dtype="bfloat16").forward(bf16_input)["tensor"]
    mish = Mish(["x"], ["y"], dtype="bfloat16").forward(bf16_input)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(gelu.data), gelu_expected, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(_bf16_to_float32(mish.data), mish_expected, rtol=2e-2, atol=2e-2)

    binarizer = Binarizer(["x"], ["y"], threshold=0.25, dtype="float32").forward(_tensor(values, "float32"))["tensor"]
    _assert_tensor_matches(binarizer, (values > 0.25).astype(np.float32))

    x = (np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2) / 5.0) - 1.0
    scale = np.array([1.0, 0.5, 1.5, -1.0], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    group = GroupNormalization(["x", "scale", "bias"], ["y"], num_groups=2, epsilon=1e-4, dtype="float32").forward(
        _tensor(x, "float32"), _tensor(scale, "float32"), _tensor(bias, "float32")
    )["tensor"]
    np.testing.assert_allclose(group.data, _group_norm_formula(x, scale, bias, 2, 1e-4), rtol=1e-5, atol=1e-5)


# 验证 Det、Tril/Triu 和 Unique 的确定性张量语义。
def test_c_backend_matrix_unique_and_triangular_ops_match_reference_formulas():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    matrices = np.array([[[1.0, 2.0], [3.0, 5.0]], [[2.0, 0.0], [1.0, 4.0]]], dtype=np.float32)
    det = Det(["x"], ["y"], dtype="float32").forward(_tensor(matrices, "float32"))["tensor"]
    _assert_tensor_matches(det, _onnx_reference("Det", [matrices], [TensorProto.FLOAT], {}, [(2,)])[0])

    triangular = np.arange(12, dtype=np.float32).reshape(3, 4)
    k = np.array(-1, dtype=np.int64)
    tril = Tril(["x", "k"], ["y"], dtype="float32").forward(_tensor(triangular, "float32"), _tensor(k, "int64"))["tensor"]
    triu = Triu(["x", "k"], ["y"], dtype="float32").forward(_tensor(triangular, "float32"), _tensor(k, "int64"))["tensor"]
    _assert_tensor_matches(tril, np.tril(triangular, k=-1))
    _assert_tensor_matches(triu, np.triu(triangular, k=-1))

    unique_input = np.array([3, 1, 3, 2, 1, 3], dtype=np.int64)
    actual = Unique(["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64").forward(_tensor(unique_input, "int64"))["tensor"]
    expected = _unique_flat_formula(unique_input, sorted_output=False)
    for actual_tensor, expected_value in zip(actual, expected):
        _assert_tensor_matches(actual_tensor, expected_value)


# 验证 NonMaxSuppression、loss 和 MelWeightMatrix 与 ONNX reference 对齐。
def test_c_backend_ranking_loss_and_mel_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 10.0, 1.0, 11.0]]], dtype=np.float32)
    scores = np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32)
    max_output = np.array([2], dtype=np.int64)
    iou = np.array([0.5], dtype=np.float32)
    nms = NonMaxSuppression(["boxes", "scores", "max", "iou"], ["selected"], center_point_box=0).forward(
        _tensor(boxes, "float32"),
        _tensor(scores, "float32"),
        _tensor(max_output, "int64"),
        _tensor(iou, "float32"),
    )["tensor"]
    _assert_tensor_matches(
        nms,
        _onnx_reference(
            "NonMaxSuppression",
            [boxes, scores, max_output, iou],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            {"center_point_box": 0},
            [(2, 3)],
            [TensorProto.INT64],
        )[0],
    )

    log_probs = np.array(
        [[[-0.1, -0.2], [-1.0, -1.1], [-2.0, -2.1]], [[-0.3, -0.4], [-1.2, -1.3], [-2.2, -2.3]]],
        dtype=np.float32,
    )
    labels = np.array([[0, 2], [1, -1]], dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    nll = NegativeLogLikelihoodLoss(["x", "target", "weights"], ["loss"], reduction="mean", ignore_index=-1, dtype="float32").forward(
        _tensor(log_probs, "float32"), _tensor(labels, "int64"), _tensor(weights, "float32")
    )["tensor"]
    _assert_tensor_matches(
        nll,
        _onnx_reference(
            "NegativeLogLikelihoodLoss",
            [log_probs, labels, weights],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            {"reduction": "mean", "ignore_index": -1},
            [()],
            [TensorProto.FLOAT],
        )[0],
    )

    sce_scores = np.array([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=np.float32)
    sce_labels = np.array([2, 0], dtype=np.int64)
    sce_loss, log_prob = SoftmaxCrossEntropyLoss(["scores", "labels"], ["loss", "log_prob"], reduction="none", dtype="float32").forward(
        _tensor(sce_scores, "float32"), _tensor(sce_labels, "int64")
    )["tensor"]
    sce_expected = _onnx_reference(
        "SoftmaxCrossEntropyLoss",
        [sce_scores, sce_labels],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"reduction": "none"},
        [(2,), sce_scores.shape],
        [TensorProto.FLOAT, TensorProto.FLOAT],
    )
    _assert_tensor_matches(sce_loss, sce_expected[0])
    _assert_tensor_matches(log_prob, sce_expected[1])

    mel_inputs = [
        np.array(3, dtype=np.int64),
        np.array(8, dtype=np.int64),
        np.array(16000, dtype=np.int64),
        np.array(0.0, dtype=np.float32),
        np.array(8000.0, dtype=np.float32),
    ]
    mel = MelWeightMatrix([], ["mel"], output_datatype=TensorProto.FLOAT).forward(
        _tensor(mel_inputs[0], "int64"),
        _tensor(mel_inputs[1], "int64"),
        _tensor(mel_inputs[2], "int64"),
        _tensor(mel_inputs[3], "float32"),
        _tensor(mel_inputs[4], "float32"),
    )["tensor"]
    _assert_tensor_matches(
        mel,
        _onnx_reference(
            "MelWeightMatrix",
            mel_inputs,
            [TensorProto.INT64, TensorProto.INT64, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT],
            {"output_datatype": TensorProto.FLOAT},
            [(5, 3)],
            [TensorProto.FLOAT],
        )[0],
    )
