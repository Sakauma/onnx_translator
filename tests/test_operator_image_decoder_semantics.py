# /**
#   ******************************************************************************
#   * @file        test_operator_image_decoder_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证 ImageDecoder 算子的官方图像解码语义。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import io

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import ImageDecoder


# 构造一张小尺寸 RGB PNG 的 uint8 字节流，保证测试无需依赖外部图片文件。
def _encoded_png():
    import PIL.Image

    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[16, 32, 48], [64, 80, 96], [112, 128, 144]],
        ],
        dtype=np.uint8,
    )
    with io.BytesIO() as buffer:
        PIL.Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


# 构造 uint8 Tensor，避免重复 shape 和 dtype 样板。
def _uint8_tensor(data):
    data = np.asarray(data, dtype=np.uint8)
    return Tensor(*data.shape, dtype="uint8", data=data)


# 调用 ONNX reference evaluator，获得 ImageDecoder 的官方输出。
def _onnx_image_decoder_reference(encoded, pixel_format="RGB"):
    graph = helper.make_graph(
        [helper.make_node("ImageDecoder", ["encoded"], ["image"], pixel_format=pixel_format)],
        "image_decoder_reference",
        [helper.make_tensor_value_info("encoded", TensorProto.UINT8, list(encoded.shape))],
        [helper.make_tensor_value_info("image", TensorProto.UINT8, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    return ReferenceEvaluator(model).run(None, {"encoded": encoded})[0]


# 验证 RGB 输出与 ONNX 官方 reference 一致。
def test_image_decoder_rgb_matches_onnx_reference():
    encoded = _encoded_png()
    expected = _onnx_image_decoder_reference(encoded, "RGB")
    actual = ImageDecoder(["encoded"], ["image"], pixel_format="RGB").forward(_uint8_tensor(encoded))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)
    assert actual.dtype == "uint8"


# 验证 BGR 输出按官方 reference 翻转 RGB 通道顺序。
def test_image_decoder_bgr_matches_onnx_reference():
    encoded = _encoded_png()
    expected = _onnx_image_decoder_reference(encoded, "BGR")
    actual = ImageDecoder(["encoded"], ["image"], pixel_format="BGR").forward(_uint8_tensor(encoded))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)


# 验证 Grayscale 输出转为单通道 channel-last 布局。
def test_image_decoder_grayscale_matches_onnx_reference():
    encoded = _encoded_png()
    expected = _onnx_image_decoder_reference(encoded, "Grayscale")
    actual = ImageDecoder(["encoded"], ["image"], pixel_format="Grayscale").forward(_uint8_tensor(encoded))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)
    assert actual.size[-1] == 1


# 验证无法解码的字节流按 ONNX schema 文档返回空图像矩阵。
def test_image_decoder_invalid_stream_returns_empty_matrix():
    encoded = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    actual = ImageDecoder(["encoded"], ["image"]).forward(_uint8_tensor(encoded))["tensor"]
    assert actual.dtype == "uint8"
    assert actual.size == (0, 0, 0)
    assert actual.data.size == 0


# 验证 ONNX 导入时保留 pixel_format 属性。
def test_onnx_import_image_decoder_preserves_pixel_format(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("ImageDecoder", ["encoded"], ["image"], pixel_format="BGR")],
        "image_decoder_import",
        [helper.make_tensor_value_info("encoded", TensorProto.UINT8, [None])],
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [None, None, None])],
    )
    model_path = tmp_path / "image_decoder.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, ImageDecoder)]
    assert len(imported) == 1
    assert imported[0].pixel_format == "BGR"
    assert imported[0].version == "20"
