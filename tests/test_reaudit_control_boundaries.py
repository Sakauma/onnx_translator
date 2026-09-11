# /**
#   ******************************************************************************
#   * @file        test_reaudit_control_boundaries.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖控制流复审发现的空值、容器与轴边界。
#   * @details     2026.09.11  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


def _tensor(value, dtype):
    array = np.asarray(value)
    return Tensor(*array.shape, dtype=dtype, data=array)


def _import_runtime(tmp_path, graph, name):
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    ops = ONNXImport(str(path), strict=True)
    runtime = Graph(
        ops,
        [value.name for value in graph.input],
        [value.name for value in graph.output],
    )
    return ops, runtime


def _loop_tensor_body(scan_dim="width"):
    return helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["state_in"], ["state_out"]),
            helper.make_node("Identity", ["state_in"], ["scan_value"]),
        ],
        "loop_tensor_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, ["width"]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, ["width"]),
            helper.make_tensor_value_info("scan_value", TensorProto.FLOAT, [scan_dim]),
        ],
    )


@pytest.mark.parametrize("trip_count, expected_scan_shape", [(0, (0, 2)), (1, (1, 2))])
def test_loop_binds_empty_scan_symbol_from_runtime_state(
    monkeypatch, tmp_path, trip_count, expected_scan_shape
):
    _disable_c_backend(monkeypatch)
    graph = helper.make_graph(
        [helper.make_node(
            "Loop", ["m", "cond", "state"], ["final", "scan"],
            body=_loop_tensor_body(),
        )],
        "loop_symbol_runtime_binding",
        [
            helper.make_tensor_value_info("m", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, ["width"]),
        ],
        [
            helper.make_tensor_value_info("final", TensorProto.FLOAT, ["width"]),
            helper.make_tensor_value_info("scan", TensorProto.FLOAT, [None, "width"]),
        ],
    )
    _ops, runtime = _import_runtime(tmp_path, graph, f"loop_symbol_{trip_count}")
    state = np.array([3.0, 4.0], dtype=np.float32)
    final, scan = runtime.forward(
        _tensor(trip_count, "int64"), _tensor(True, "bool"),
        _tensor(state, "float32"),
    )
    np.testing.assert_array_equal(final.data, state)
    assert scan.data.shape == expected_scan_shape


def test_loop_empty_scan_rejects_unbound_symbol(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)
    graph = helper.make_graph(
        [helper.make_node(
            "Loop", ["m", "cond", "state"], ["final", "scan"],
            body=_loop_tensor_body("unbound"),
        )],
        "loop_unbound_scan_symbol",
        [
            helper.make_tensor_value_info("m", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("final", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan", TensorProto.FLOAT, [None, None]),
        ],
    )
    _ops, runtime = _import_runtime(tmp_path, graph, "loop_unbound_scan_symbol")
    with pytest.raises(ValueError, match="cannot determine dimension 0"):
        runtime.forward(
            _tensor(0, "int64"), _tensor(False, "bool"),
            _tensor(np.array([1.0, 2.0], dtype=np.float32), "float32"),
        )


@pytest.mark.parametrize("trip_count", [0, 1])
def test_loop_preserves_sequence_carried_state(monkeypatch, tmp_path, trip_count):
    _disable_c_backend(monkeypatch)
    seq_in = helper.make_tensor_sequence_value_info("seq_in", TensorProto.FLOAT, [2])
    seq_out = helper.make_tensor_sequence_value_info("seq_out", TensorProto.FLOAT, [2])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["seq_in"], ["seq_out"]),
        ],
        "loop_sequence_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            seq_in,
        ],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), seq_out],
    )
    graph = helper.make_graph(
        [
            helper.make_node("SequenceConstruct", ["x"], ["initial"]),
            helper.make_node(
                "Loop", ["m", "cond", "initial"], ["final_sequence"], body=body
            ),
            helper.make_node("SequenceLength", ["final_sequence"], ["length"]),
        ],
        "loop_sequence_state",
        [
            helper.make_tensor_value_info("m", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("length", TensorProto.INT64, [])],
    )
    _ops, runtime = _import_runtime(tmp_path, graph, f"loop_sequence_{trip_count}")
    length = runtime.forward(
        _tensor(trip_count, "int64"), _tensor(True, "bool"),
        _tensor(np.array([1.0, 2.0], dtype=np.float32), "float32"),
    )
    assert int(np.asarray(length.data)) == 1


@pytest.mark.parametrize("empty", [True, False])
def test_sequence_map_keeps_multi_output_arity(monkeypatch, tmp_path, empty):
    _disable_c_backend(monkeypatch)
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["item"], ["same"]),
            helper.make_node("Neg", ["item"], ["negative"]),
        ],
        "sequence_map_multi_body",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("same", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("negative", TensorProto.FLOAT, [2]),
        ],
    )
    nodes = []
    inputs = []
    if empty:
        nodes.append(helper.make_node("SequenceEmpty", [], ["sequence"], dtype=TensorProto.FLOAT))
    else:
        inputs.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]))
        nodes.append(helper.make_node("SequenceConstruct", ["x"], ["sequence"]))
    nodes.extend([
        helper.make_node(
            "SequenceMap", ["sequence"], ["mapped_a", "mapped_b"], body=body
        ),
        helper.make_node("SequenceLength", ["mapped_a"], ["len_a"]),
        helper.make_node("SequenceLength", ["mapped_b"], ["len_b"]),
    ])
    graph = helper.make_graph(
        nodes, "sequence_map_multi_output", inputs,
        [
            helper.make_tensor_value_info("len_a", TensorProto.INT64, []),
            helper.make_tensor_value_info("len_b", TensorProto.INT64, []),
        ],
    )
    _ops, runtime = _import_runtime(tmp_path, graph, f"sequence_map_{empty}")
    args = () if empty else (
        _tensor(np.array([2.0, -3.0], dtype=np.float32), "float32"),
    )
    len_a, len_b = runtime.forward(*args)
    expected = 0 if empty else 1
    assert int(np.asarray(len_a.data)) == expected
    assert int(np.asarray(len_b.data)) == expected


@pytest.mark.parametrize("output_axis, expected_shape", [(0, (4, 2, 3)), (-1, (2, 3, 4)), (-3, (4, 2, 3))])
def test_scan_output_axis_matches_runtime_and_shape_only(
    monkeypatch, tmp_path, output_axis, expected_shape
):
    _disable_c_backend(monkeypatch)
    body = helper.make_graph(
        [helper.make_node("Identity", ["item"], ["value"])],
        "scan_axis_body",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("value", TensorProto.FLOAT, [2, 3])],
    )
    graph = helper.make_graph(
        [helper.make_node(
            "Scan", ["x"], ["y"], body=body, num_scan_inputs=1,
            scan_output_axes=[output_axis],
        )],
        "scan_output_axis",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(expected_shape))],
    )
    ops, runtime = _import_runtime(tmp_path, graph, f"scan_axis_{output_axis}")
    x = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    actual = runtime.forward(_tensor(x, "float32"))
    shape_only = ops[0].forward_(Tensor_(4, 2, 3, dtype="float32"))["tensor"]
    assert actual.data.shape == expected_shape
    assert tuple(shape_only.size) == expected_shape


@pytest.mark.parametrize("output_axis", [-4, 3])
def test_scan_rejects_output_axis_outside_final_rank(monkeypatch, output_axis):
    _disable_c_backend(monkeypatch)
    body = helper.make_graph(
        [helper.make_node("Identity", ["item"], ["value"])],
        "scan_invalid_axis_body",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("value", TensorProto.FLOAT, [2, 3])],
    )
    scan = Scan(
        ["x"], ["y"], body=body, num_scan_inputs=1,
        scan_output_axes=[output_axis],
    )
    tensor = _tensor(np.arange(24, dtype=np.float32).reshape(4, 2, 3), "float32")
    with pytest.raises(ValueError, match="output axis"):
        scan.forward(tensor)
    with pytest.raises(ValueError, match="output axis"):
        scan.forward_(Tensor_(4, 2, 3, dtype="float32"))
