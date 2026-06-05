# /**
#   ******************************************************************************
#   * @file        test_operator_text_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证高版本字符串算子的官方语义。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import RegexFullMatch, StringConcat, StringSplit


# 构造字符串 Tensor，避免测试里重复 shape 和 dtype 样板。
def _string_tensor(data):
    data = np.asarray(data, dtype=np.str_)
    return Tensor(*data.shape, dtype="string", data=data)


# 调用 ONNX reference evaluator，返回单节点模型的官方输出列表。
def _onnx_reference(op_name, inputs, attrs, output_shapes, output_protos, opset=20):
    input_names = [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, TensorProto.STRING, list(np.asarray(value).shape))
            for name, value in zip(input_names, inputs)
        ],
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 验证 RegexFullMatch 对每个字符串元素执行 fullmatch，而不是 search/partial match。
def test_regex_full_match_matches_onnx_reference():
    data = np.array(
        [["account@gmail.com", "prefix account@gmail.com suffix"], ["not email", "account2@yahoo.com"]],
        dtype=object,
    )
    pattern = r"[\w.\-]{0,25}@(yahoo|gmail)\.com"
    expected = _onnx_reference(
        "RegexFullMatch",
        [data],
        {"pattern": pattern},
        [data.shape],
        [TensorProto.BOOL],
    )[0]
    actual = RegexFullMatch(["x"], ["y"], pattern=pattern).forward(_string_tensor(data))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)


# 验证 RegexFullMatch 空输入保持空输出 shape。
def test_regex_full_match_empty_input_matches_onnx_reference():
    data = np.array([[], []], dtype=object)
    pattern = r"abc"
    expected = _onnx_reference(
        "RegexFullMatch",
        [data],
        {"pattern": pattern},
        [data.shape],
        [TensorProto.BOOL],
    )[0]
    actual = RegexFullMatch(["x"], ["y"], pattern=pattern).forward(_string_tensor(data))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)
    assert actual.size == expected.shape


# 验证 StringConcat 逐元素拼接并遵循 NumPy-style broadcasting。
def test_string_concat_broadcast_matches_onnx_reference():
    left = np.array([["pre-", "x-"], ["mid-", "z-"]], dtype=object)
    right = np.array(["A", "B"], dtype=object)
    expected = _onnx_reference(
        "StringConcat",
        [left, right],
        {},
        [(2, 2)],
        [TensorProto.STRING],
    )[0]
    actual = StringConcat(["x", "y"], ["z"]).forward(_string_tensor(left), _string_tensor(right))["tensor"]
    np.testing.assert_array_equal(actual.data.astype(np.str_), expected.astype(np.str_))


# 验证 StringSplit 使用普通 delimiter 时保留连续分隔符生成的空字符串。
def test_string_split_delimiter_and_padding_match_onnx_reference():
    data = np.array([["a,,b", "c,d,e"]], dtype=object)
    expected_strings, expected_counts = _onnx_reference(
        "StringSplit",
        [data],
        {"delimiter": ",", "maxsplit": 1},
        [(1, 2, 2), (1, 2)],
        [TensorProto.STRING, TensorProto.INT64],
    )
    actual_strings, actual_counts = StringSplit(["x"], ["y", "z"], delimiter=",", maxsplit=1).forward(
        _string_tensor(data)
    )["tensor"]
    np.testing.assert_array_equal(actual_strings.data.astype(np.str_), expected_strings.astype(np.str_))
    np.testing.assert_array_equal(actual_counts.data, expected_counts)


# 验证 StringSplit 缺省 delimiter 时按连续空白拆分并裁剪首尾空白。
def test_string_split_whitespace_default_matches_onnx_reference():
    data = np.array(["  a   b  ", "c"], dtype=object)
    expected_strings, expected_counts = _onnx_reference(
        "StringSplit",
        [data],
        {},
        [(2, 2), (2,)],
        [TensorProto.STRING, TensorProto.INT64],
    )
    actual_strings, actual_counts = StringSplit(["x"], ["y", "z"]).forward(_string_tensor(data))["tensor"]
    np.testing.assert_array_equal(actual_strings.data.astype(np.str_), expected_strings.astype(np.str_))
    np.testing.assert_array_equal(actual_counts.data, expected_counts)


# 验证 StringSplit 空输入时最后一维为 0，计数输出保持输入 shape。
def test_string_split_empty_input_matches_onnx_reference():
    data = np.array([[], []], dtype=object)
    expected_strings, expected_counts = _onnx_reference(
        "StringSplit",
        [data],
        {"delimiter": ","},
        [(2, 0, 0), (2, 0)],
        [TensorProto.STRING, TensorProto.INT64],
    )
    actual_strings, actual_counts = StringSplit(["x"], ["y", "z"], delimiter=",").forward(_string_tensor(data))[
        "tensor"
    ]
    np.testing.assert_array_equal(actual_strings.data.astype(np.str_), expected_strings.astype(np.str_))
    np.testing.assert_array_equal(actual_counts.data, expected_counts)
    assert actual_strings.size == expected_strings.shape


# 验证三个字符串算子导入时保留官方属性。
def test_onnx_import_string_ops_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("StringConcat", ["left", "right"], ["joined"]),
            helper.make_node("RegexFullMatch", ["joined"], ["matched"], pattern=r"pre-[AB]"),
            helper.make_node("StringSplit", ["joined"], ["parts", "counts"], delimiter="-", maxsplit=1),
        ],
        "string_ops_import",
        [
            helper.make_tensor_value_info("left", TensorProto.STRING, [2]),
            helper.make_tensor_value_info("right", TensorProto.STRING, [2]),
        ],
        [
            helper.make_tensor_value_info("joined", TensorProto.STRING, [2]),
            helper.make_tensor_value_info("matched", TensorProto.BOOL, [2]),
            helper.make_tensor_value_info("parts", TensorProto.STRING, [2, 2]),
            helper.make_tensor_value_info("counts", TensorProto.INT64, [2]),
        ],
    )
    model_path = tmp_path / "string_ops.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), model_path)

    imported = ONNXImport(str(model_path), strict=True)
    assert [op.__class__.__name__ for op in imported] == ["StringConcat", "RegexFullMatch", "StringSplit"]
    assert imported[1].pattern == r"pre-[AB]"
    assert imported[2].delimiter == "-"
    assert imported[2].maxsplit == 1
    assert imported[0].version == "20"
    assert imported[1].version == "20"
    assert imported[2].version == "20"
