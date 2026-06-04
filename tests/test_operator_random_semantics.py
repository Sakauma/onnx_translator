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
