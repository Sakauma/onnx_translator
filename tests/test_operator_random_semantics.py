# /**
#   ******************************************************************************
#   * @file        test_operator_random_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       验证随机、概率采样和 Dropout 算子的 ONNX schema 语义与混合精度 dtype 行为。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    Bernoulli,
    Dropout,
    Multinomial,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
)


# 构造 Tensor，避免每个测试重复 shape/dtype 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


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


# 调用 ONNX reference evaluator，用于验证可精确复现的随机算子路径。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos):
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 验证 RandomUniform/RandomUniformLike 的 shape、dtype、范围和 seed 复现语义。
def test_c_backend_random_uniform_ops_respect_schema_bounds_dtype_and_seed():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    first = RandomUniform([], ["y"], low=-2.0, high=-1.0, seed=7.0, dtype="float64", shape=[2, 3]).forward()["tensor"]
    second = RandomUniform([], ["y"], low=-2.0, high=-1.0, seed=7.0, dtype="float64", shape=[2, 3]).forward()["tensor"]
    third = RandomUniform([], ["y"], low=-2.0, high=-1.0, seed=8.0, dtype="float64", shape=[2, 3]).forward()["tensor"]
    assert first.dtype == "float64"
    assert first.data.dtype == np.float64
    assert first.size == (2, 3)
    assert np.all(first.data >= -2.0)
    assert np.all(first.data < -1.0)
    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, third.data)

    scalar = RandomUniform([], ["y"], low=0.0, high=1.0, seed=3.0, dtype=TensorProto.FLOAT, shape=[]).forward()["tensor"]
    assert scalar.size == ()
    assert scalar.data.shape == ()
    assert 0.0 <= float(scalar.data) < 1.0

    like_input = np.ones((2, 2), dtype=np.float16)
    inherited = RandomUniformLike(["x"], ["y"], low=0.25, high=0.75, seed=11.0).forward(_tensor(like_input, "float16"))["tensor"]
    assert inherited.dtype == "float16"
    assert inherited.data.dtype == np.float16
    assert np.all(inherited.data >= np.float16(0.25))
    assert np.all(inherited.data < np.float16(0.75))

    explicit = RandomUniformLike(["x"], ["y"], low=0.0, high=1.0, seed=11.0, dtype=TensorProto.DOUBLE).forward(
        _tensor(like_input, "float16")
    )["tensor"]
    assert explicit.dtype == "float64"
    assert explicit.data.dtype == np.float64


# 验证 RandomNormal/RandomNormalLike 的 seed 复现、dtype 解析和基础分布统计。
def test_c_backend_random_normal_ops_respect_schema_dtype_seed_and_distribution():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    first = RandomNormal([], ["y"], mean=3.0, scale=0.5, seed=13.0, dtype="float64", shape=[4096]).forward()["tensor"]
    second = RandomNormal([], ["y"], mean=3.0, scale=0.5, seed=13.0, dtype="float64", shape=[4096]).forward()["tensor"]
    third = RandomNormal([], ["y"], mean=3.0, scale=0.5, seed=14.0, dtype="float64", shape=[4096]).forward()["tensor"]
    assert first.dtype == "float64"
    assert first.data.dtype == np.float64
    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, third.data)
    assert abs(float(np.mean(first.data)) - 3.0) < 0.08
    assert abs(float(np.std(first.data)) - 0.5) < 0.08

    like_input = np.zeros((3, 4), dtype=np.float32)
    like = RandomNormalLike(["x"], ["y"], mean=-1.0, scale=2.0, seed=5.0, dtype="float64").forward(
        _tensor(like_input, "float32")
    )["tensor"]
    assert like.dtype == "float64"
    assert like.data.dtype == np.float64
    assert like.size == (3, 4)
    assert np.all(np.isfinite(like.data))


# 验证 Bernoulli 的概率极值、输出 dtype 和 seed 复现。
def test_c_backend_bernoulli_respects_probability_extremes_dtype_and_seed():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    extremes = Bernoulli(["p"], ["y"], dtype=TensorProto.UINT32, seed=5.0).forward(
        _tensor(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32), "float32")
    )["tensor"]
    assert extremes.dtype == "uint32"
    np.testing.assert_array_equal(extremes.data, np.array([0, 1, 1, 0], dtype=np.uint32))

    probabilities = _tensor(np.array([0.2, 0.8, 0.4, 0.6, 0.5], dtype=np.float32), "float32")
    first = Bernoulli(["p"], ["y"], seed=17.0, dtype="float32").forward(probabilities)["tensor"]
    second = Bernoulli(["p"], ["y"], seed=17.0, dtype="float32").forward(probabilities)["tensor"]
    np.testing.assert_array_equal(first.data, second.data)
    assert set(np.unique(first.data)).issubset({0.0, 1.0})


# 验证 Multinomial 的 one-hot 概率、输出 dtype/shape 和 seed 复现。
def test_c_backend_multinomial_respects_probability_rows_dtype_and_seed():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    one_hot = _tensor(np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 0.0]], dtype=np.float32), "float32")
    exact = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=4, seed=7.0).forward(one_hot)["tensor"]
    assert exact.dtype == "int64"
    assert exact.size == (2, 4)
    np.testing.assert_array_equal(exact.data, np.array([[2, 2, 2, 2], [0, 0, 0, 0]], dtype=np.int64))

    probabilities = _tensor(np.array([[0.1, 0.2, 0.7], [2.0, 1.0, 1.0]], dtype=np.float32), "float32")
    first = Multinomial(["p"], ["y"], dtype="int32", sample_size=6, seed=9.0).forward(probabilities)["tensor"]
    second = Multinomial(["p"], ["y"], dtype="int32", sample_size=6, seed=9.0).forward(probabilities)["tensor"]
    np.testing.assert_array_equal(first.data, second.data)
    assert first.dtype == "int32"
    assert first.size == (2, 6)
    assert np.all((first.data >= 0) & (first.data < 3))


# 验证 Python fallback 在 bfloat16 下正确按位写回随机输出，而不是写入普通 uint16 数值。
def test_python_random_fallback_bfloat16_outputs_use_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    uniform = RandomUniform([], ["y"], low=0.25, high=0.75, seed=19.0, dtype="bfloat16", shape=[2, 3]).forward()["tensor"]
    assert uniform.dtype == "bfloat16"
    assert uniform.data.dtype == np.uint16
    uniform_values = _bf16_to_float32(uniform.data)
    assert np.all(np.isfinite(uniform_values))
    assert np.all(uniform_values >= np.float32(0.25))
    assert np.all(uniform_values <= np.float32(0.75))

    like_input = _tensor(_bf16_bits(np.ones((2, 2), dtype=np.float32)), "bfloat16")
    uniform_like = RandomUniformLike(["x"], ["y"], low=-0.5, high=0.5, seed=23.0).forward(like_input)["tensor"]
    assert uniform_like.dtype == "bfloat16"
    assert uniform_like.data.dtype == np.uint16
    uniform_like_values = _bf16_to_float32(uniform_like.data)
    assert np.all(uniform_like_values >= np.float32(-0.5))
    assert np.all(uniform_like_values <= np.float32(0.5))

    normal = RandomNormal([], ["y"], mean=2.0, scale=0.25, seed=29.0, dtype="bfloat16", shape=[2048]).forward()["tensor"]
    assert normal.dtype == "bfloat16"
    assert normal.data.dtype == np.uint16
    normal_values = _bf16_to_float32(normal.data)
    assert np.all(np.isfinite(normal_values))
    assert abs(float(np.mean(normal_values)) - 2.0) < 0.03

    normal_like = RandomNormalLike(["x"], ["y"], mean=-1.0, scale=0.5, seed=31.0).forward(like_input)["tensor"]
    assert normal_like.dtype == "bfloat16"
    assert normal_like.data.dtype == np.uint16
    assert np.all(np.isfinite(_bf16_to_float32(normal_like.data)))


# 验证 Bernoulli/Multinomial fallback 会先解码 bfloat16 概率，再按目标 dtype 写回输出。
def test_python_probability_fallback_bfloat16_decodes_probabilities(monkeypatch):
    _disable_c_backend(monkeypatch)

    extreme_prob = _tensor(_bf16_bits(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)), "bfloat16")
    bernoulli = Bernoulli(["p"], ["y"], dtype="bfloat16", seed=37.0).forward(extreme_prob)["tensor"]
    assert bernoulli.dtype == "bfloat16"
    np.testing.assert_array_equal(bernoulli.data, _bf16_bits(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)))

    skewed_prob = _tensor(_bf16_bits(np.array([[1.0e-8, 1.0, 0.0]], dtype=np.float32)), "bfloat16")
    samples = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=64, seed=41.0).forward(skewed_prob)["tensor"]
    assert samples.dtype == "int64"
    np.testing.assert_array_equal(samples.data, np.ones((1, 64), dtype=np.int64))


# 验证 Dropout 推理和训练模式与 ONNX reference 在固定 seed 下完全一致。
def test_dropout_inference_and_training_match_onnx_reference_with_seed():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    ratio = np.array(0.5, dtype=np.float32)

    inference_mode = np.array(False, dtype=np.bool_)
    inference_actual = Dropout(["x", "ratio", "training_mode"], ["y", "mask"], seed=123).forward(
        _tensor(x, "float32"), _tensor(ratio, "float32"), _tensor(inference_mode, "bool")
    )["tensor"]
    inference_expected = _onnx_reference(
        "Dropout",
        [x, ratio, inference_mode],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.BOOL],
        {"seed": 123},
        [x.shape, x.shape],
        [TensorProto.FLOAT, TensorProto.BOOL],
    )
    np.testing.assert_array_equal(inference_actual[0].data, inference_expected[0])
    np.testing.assert_array_equal(inference_actual[1].data, inference_expected[1])

    training_mode = np.array(True, dtype=np.bool_)
    training_actual = Dropout(["x", "ratio", "training_mode"], ["y", "mask"], seed=123).forward(
        _tensor(x, "float32"), _tensor(ratio, "float32"), _tensor(training_mode, "bool")
    )["tensor"]
    training_expected = _onnx_reference(
        "Dropout",
        [x, ratio, training_mode],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.BOOL],
        {"seed": 123},
        [x.shape, x.shape],
        [TensorProto.FLOAT, TensorProto.BOOL],
    )
    np.testing.assert_array_equal(training_actual[1].data, training_expected[1])
    np.testing.assert_allclose(training_actual[0].data, training_expected[0], rtol=1e-7, atol=1e-7)


# 验证 Dropout 训练模式在 bfloat16 下先解码输入再缩放，并把输出重新编码为位模式。
def test_dropout_training_bfloat16_decodes_and_encodes_bit_storage():
    x_values = np.array([[1.0, 2.0, -3.0], [0.5, -0.25, 4.0]], dtype=np.float32)
    ratio = np.array(0.25, dtype=np.float32)
    training_mode = np.array(True, dtype=np.bool_)
    actual_y, actual_mask = Dropout(["x", "ratio", "training_mode"], ["y", "mask"], seed=53).forward(
        _tensor(_bf16_bits(x_values), "bfloat16"),
        _tensor(_bf16_bits(ratio), "bfloat16"),
        _tensor(training_mode, "bool"),
    )["tensor"]

    mask = np.random.RandomState(53).uniform(0.0, 1.0, x_values.shape) >= float(ratio)
    expected = _bf16_bits(x_values * mask.astype(np.float32) / (1.0 - float(ratio)))
    assert actual_y.dtype == "bfloat16"
    assert actual_y.data.dtype == np.uint16
    np.testing.assert_array_equal(actual_mask.data, mask)
    np.testing.assert_array_equal(actual_y.data, expected)
