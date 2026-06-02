"""文件功能：覆盖序列、可选值、控制流和常量形状推断场景。
作者：Egor Izmaylov
时间：2026-06-02
"""

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


def test_onnx17_sequence_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    a = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(2, dtype="float32", data=np.array([3.0, 4.0], dtype=np.float32))
    c = Tensor(2, dtype="float32", data=np.array([5.0, 6.0], dtype=np.float32))
    seq = SequenceConstruct(["a", "b"], ["seq"], dtype="float32").forward(a, b)["tensor"]
    assert len(seq) == 2

    inserted = SequenceInsert(["seq", "c"], ["out"], dtype="float32").forward(seq, c)["tensor"]
    assert len(inserted) == 3
    picked = SequenceAt(["seq", "pos"], ["out"], dtype="float32").forward(
        inserted, Tensor(1, dtype="int64", data=np.array([-1], dtype=np.int64))
    )["tensor"]
    np.testing.assert_array_equal(picked.data, c.data)

    erased = SequenceErase(["seq"], ["out"], dtype="float32").forward(inserted)["tensor"]
    assert len(erased) == 2
    length = SequenceLength(["seq"], ["len"]).forward(erased)["tensor"]
    assert length.size == ()
    np.testing.assert_array_equal(length.data, np.array(2, dtype=np.int64))

    stacked = ConcatFromSequence(["seq"], ["out"], axis=0, new_axis=1, dtype="float32").forward(erased)["tensor"]
    np.testing.assert_array_equal(stacked.data, np.stack([a.data, b.data], axis=0))

    split_input = Tensor(2, 3, dtype="float32", data=np.arange(6, dtype=np.float32).reshape(2, 3))
    split = Tensor(2, dtype="int64", data=np.array([1, 2], dtype=np.int64))
    pieces = SplitToSequence(["x", "split"], ["seq"], axis=1, keepdims=1, dtype="float32").forward(split_input, split)["tensor"]
    assert [piece.size for piece in pieces] == [(2, 1), (2, 2)]
    np.testing.assert_array_equal(pieces[0].data, split_input.data[:, :1])
    squeezed_pieces = SplitToSequence(["x"], ["seq"], axis=1, keepdims=0, dtype="float32").forward(split_input)["tensor"]
    assert [piece.size for piece in squeezed_pieces] == [(2,), (2,), (2,)]

    empty = SequenceEmpty([], ["seq"], dtype="float32").forward()["tensor"]
    assert empty == []

    model_path = tmp_path / "onnx17_sequence_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("SequenceEmpty", [], ["empty"], dtype=TensorProto.FLOAT),
            helper.make_node("SequenceConstruct", ["a", "b"], ["seq"]),
            helper.make_node("SequenceLength", ["seq"], ["seq_len"]),
            helper.make_node("SequenceAt", ["seq", "pos"], ["at"]),
            helper.make_node("SequenceInsert", ["seq", "c"], ["seq_inserted"]),
            helper.make_node("SequenceErase", ["seq_inserted"], ["seq_erased"]),
            helper.make_node("ConcatFromSequence", ["seq_erased"], ["concat"], axis=0, new_axis=0),
            helper.make_node("SplitToSequence", ["matrix", "split"], ["split_seq"], axis=1),
        ],
        "onnx17_sequence_ops",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("pos", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("matrix", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("split", TensorProto.INT64, [2]),
        ],
        [
            helper.make_tensor_sequence_value_info("empty", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("seq_len", TensorProto.INT64, []),
            helper.make_tensor_value_info("at", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("concat", TensorProto.FLOAT, [4]),
            helper.make_tensor_sequence_value_info("split_seq", TensorProto.FLOAT, None),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "SequenceEmpty", "SequenceConstruct", "SequenceLength", "SequenceAt",
        "SequenceInsert", "SequenceErase", "ConcatFromSequence", "SplitToSequence"
    ]

def test_onnx17_optional_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    tensor = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))
    optional = Optional(["x"], ["opt"], dtype="float32").forward(tensor)["tensor"]
    np.testing.assert_array_equal(OptionalGetElement(["opt"], ["y"], dtype="float32").forward(optional)["tensor"].data, tensor.data)
    has = OptionalHasElement(["opt"], ["has"]).forward(optional)["tensor"]
    assert has.size == ()
    np.testing.assert_array_equal(has.data, np.array(True, dtype=np.bool_))
    empty = Optional([], ["opt"], dtype="float32").forward()["tensor"]
    empty_has = OptionalHasElement(["opt"], ["has"]).forward(empty)["tensor"]
    assert empty_has.size == ()
    np.testing.assert_array_equal(empty_has.data, np.array(False, dtype=np.bool_))
    with pytest.raises(ValueError, match="empty optional"):
        OptionalGetElement(["opt"], ["y"], dtype="float32").forward(empty)

    model_path = tmp_path / "onnx17_optional_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Optional", ["x"], ["opt"]),
            helper.make_node("OptionalHasElement", ["opt"], ["has"]),
            helper.make_node("OptionalGetElement", ["opt"], ["y"]),
        ],
        "onnx17_optional_ops",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("has", TensorProto.BOOL, []),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["Optional", "OptionalHasElement", "OptionalGetElement"]

def test_onnx17_control_flow_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    one_const = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    zero_const = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    true_const = helper.make_tensor("true", TensorProto.BOOL, [], [True])

    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["branch_y"], value=one_const)],
        "then_branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["branch_y"], value=zero_const)],
        "else_branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
    )
    if_out = If(["cond"], ["y"], then_branch=then_graph, else_branch=else_graph).forward(
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_))
    )["tensor"]
    np.testing.assert_array_equal(if_out.data, np.array(1.0, dtype=np.float32))

    loop_body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["cond_out"], value=true_const),
            helper.make_node("Constant", [], ["one"], value=one_const),
            helper.make_node("Add", ["v_in", "one"], ["v_out"]),
            helper.make_node("Identity", ["v_out"], ["scan_out"]),
        ],
        "loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, []),
        ],
    )
    loop_final, loop_scan = Loop(["m", "cond", "v"], ["v_final", "scan"], body=loop_body).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(loop_final.data, np.array(3.0, dtype=np.float32))
    np.testing.assert_array_equal(loop_scan.data, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    zero_loop_final, zero_loop_scan = Loop(["m", "cond", "v"], ["v_final", "scan"], body=loop_body).forward(
        Tensor(dtype="int64", data=np.array(0, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(5.0, dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(zero_loop_final.data, np.array(5.0, dtype=np.float32))
    assert zero_loop_scan.data.shape == (0,)

    scan_body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "x_in"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["scan_y"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_y", TensorProto.FLOAT, []),
        ],
    )
    scan_final, scan_y = Scan(["state", "x"], ["state_final", "scan_y"], body=scan_body, num_scan_inputs=1).forward(
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(3, dtype="float32", data=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(scan_final.data, np.array(6.0, dtype=np.float32))
    np.testing.assert_array_equal(scan_y.data, np.array([1.0, 3.0, 6.0], dtype=np.float32))

    seq_body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["one"], value=one_const),
            helper.make_node("Add", ["item", "one"], ["mapped"]),
        ],
        "sequence_map_body",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("mapped", TensorProto.FLOAT, [])],
    )
    mapped = SequenceMap(["seq"], ["out_seq"], body=seq_body).forward(
        [
            Tensor(dtype="float32", data=np.array(1.0, dtype=np.float32)),
            Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
        ]
    )["tensor"]
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[0].data, np.array(2.0, dtype=np.float32))
    np.testing.assert_array_equal(mapped[1].data, np.array(3.0, dtype=np.float32))

    model_path = tmp_path / "onnx17_control_flow_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("If", ["cond"], ["if_y"], then_branch=then_graph, else_branch=else_graph),
            helper.make_node("Loop", ["m", "loop_cond", "loop_v"], ["loop_final", "loop_scan"], body=loop_body),
            helper.make_node("Scan", ["scan_state", "scan_x"], ["scan_final", "scan_out"], body=scan_body, num_scan_inputs=1),
            helper.make_node("SequenceMap", ["seq"], ["mapped_seq"], body=seq_body),
        ],
        "onnx17_control_flow_ops",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("m", TensorProto.INT64, []),
            helper.make_tensor_value_info("loop_cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("loop_v", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_state", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_x", TensorProto.FLOAT, [3]),
            helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None),
        ],
        [
            helper.make_tensor_value_info("if_y", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("loop_final", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("loop_scan", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("scan_final", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3]),
            helper.make_tensor_sequence_value_info("mapped_seq", TensorProto.FLOAT, None),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["If", "Loop", "Scan", "SequenceMap"]

def test_control_flow_subgraphs_capture_outer_scope(monkeypatch):
    _disable_c_backend(monkeypatch)

    class PassThroughOp:
        # 初始化 `PassThroughOp` 的构造参数，保存后续运行、形状推断或验证所需的状态。
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
            self.name = None

        # 执行 `PassThroughOp` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
        def forward(self, x):
            return {"tensor": x, "parameters": None}

        # 执行 `PassThroughOp` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
        def forward_(self, x):
            return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}

    one_const = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    neg_one_const = helper.make_tensor("neg_one", TensorProto.FLOAT, [], [-1.0])

    then_graph = helper.make_graph(
        [helper.make_node("Add", ["outer_x", "one"], ["branch_y"])],
        "then_capture",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
        initializer=[one_const],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Add", ["outer_x", "neg_one"], ["branch_y"])],
        "else_capture",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
        initializer=[neg_one_const],
    )
    if_graph = Graph(
        [
            PassThroughOp(["cond"], ["cond2"]),
            If(["cond2"], ["y"], then_branch=then_graph, else_branch=else_graph),
        ],
        input_name=["cond", "outer_x"],
        output_name=["y"],
    )
    if_out = if_graph.forward(
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(if_out.data, np.array(3.0, dtype=np.float32))

    loop_body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Add", ["v_in", "bias"], ["v_out"]),
        ],
        "loop_capture",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, []),
        ],
    )
    loop_graph = Graph(
        [
            PassThroughOp(["cond"], ["cond2"]),
            Loop(["trip_count", "cond2", "v"], ["v_final"], body=loop_body),
        ],
        input_name=["trip_count", "cond", "v", "bias"],
        output_name=["v_final"],
    )
    loop_out = loop_graph.forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(loop_out.data, np.array(6.0, dtype=np.float32))

    scan_body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "x_in"], ["tmp"]),
            helper.make_node("Add", ["tmp", "bias"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["scan_y"]),
        ],
        "scan_capture",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_y", TensorProto.FLOAT, []),
        ],
    )
    scan_graph = Graph(
        [
            PassThroughOp(["state"], ["state2"]),
            Scan(["state2", "x"], ["state_final", "scan_y"], body=scan_body, num_scan_inputs=1),
        ],
        input_name=["state", "x", "bias"],
        output_name=["state_final", "scan_y"],
    )
    scan_final, scan_y = scan_graph.forward(
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(10.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(scan_final.data, np.array(23.0, dtype=np.float32))
    np.testing.assert_array_equal(scan_y.data, np.array([11.0, 23.0], dtype=np.float32))

    seq_body = helper.make_graph(
        [helper.make_node("Add", ["item", "bias"], ["mapped"])],
        "sequence_map_capture",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("mapped", TensorProto.FLOAT, [])],
    )
    seq_graph = Graph(
        [
            PassThroughOp(["seq"], ["seq2"]),
            SequenceMap(["seq2"], ["mapped_seq"], body=seq_body),
        ],
        input_name=["seq", "bias"],
        output_name=["mapped_seq"],
    )
    mapped = seq_graph.forward(
        [
            Tensor(dtype="float32", data=np.array(1.0, dtype=np.float32)),
            Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
        ],
        Tensor(dtype="float32", data=np.array(5.0, dtype=np.float32)),
    )
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[0].data, np.array(6.0, dtype=np.float32))
    np.testing.assert_array_equal(mapped[1].data, np.array(7.0, dtype=np.float32))

def test_onehot_and_compress_shape_inference(monkeypatch):
    _disable_c_backend(monkeypatch)

    depth = Tensor(1, dtype="int64", data=np.array([4], dtype=np.int64))
    values = Tensor(2, dtype="float32", data=np.array([0.0, 1.0], dtype=np.float32))
    onehot = OneHot(["indices", "depth", "values"], ["out"], axis=-1, dtype="float32").forward_(Tensor_(2, 3, dtype="int64"), depth, values)["tensor"]
    assert onehot.size == (2, 3, 4)

    condition = Tensor(5, dtype="bool", data=np.array([True, False, True, False, True]))
    compress_axis = Compress(["x", "cond"], ["out"], axis=1, dtype="float32").forward_(Tensor_(2, 5, dtype="float32"), condition)["tensor"]
    assert compress_axis.size == (2, 3)

    data = Tensor(2, 3, dtype="float32", data=np.arange(6, dtype=np.float32).reshape(2, 3))
    flat_cond = Tensor(6, dtype="bool", data=np.array([True, False, True, False, False, True]))
    flat = Compress(["x", "cond"], ["out"], axis=None, dtype="float32").forward(data, flat_cond)["tensor"]
    np.testing.assert_array_equal(flat.data, np.array([0.0, 2.0, 5.0], dtype=np.float32))
    assert flat.size == (3,)

def test_reduce_and_nonzero_constant_shape_inference(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    axes = Tensor(1, dtype="int64", data=np.array([1], dtype=np.int64))
    reduced = ReduceSum(["x", "axes"], ["out"], axes=None, keepdims=0, dtype="float32").forward_(
        Tensor_(2, 3, 4, dtype="float32"), axes
    )["tensor"]
    assert reduced.size == (2, 4)

    data_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = Tensor(2, 3, dtype="float32", data=data_arr)
    empty_axes = Tensor(0, dtype="int64", data=np.array([], dtype=np.int64))
    reduce_all = ReduceSum(["x", "axes"], ["out"], axes=None, keepdims=0, dtype="float32").forward(
        data, empty_axes
    )["tensor"]
    np.testing.assert_array_equal(reduce_all.data, np.sum(data_arr))
    assert reduce_all.size == ()

    no_op = ReduceSum(
        ["x", "axes"], ["out"], axes=None, keepdims=0, noop_with_empty_axes=1, dtype="float32"
    ).forward(data, empty_axes)["tensor"]
    np.testing.assert_array_equal(no_op.data, data_arr)
    assert no_op.size == (2, 3)

    axes_initializer = helper.make_tensor("axes_empty", TensorProto.INT64, [0], [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y0_info = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [])
    y1_info = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["x", "axes_empty"], ["y0"], keepdims=0),
            helper.make_node("ReduceSum", ["x", "axes_empty"], ["y1"], keepdims=0, noop_with_empty_axes=1),
        ],
        "reduce_sum_empty_axes",
        [x_info],
        [y0_info, y1_info],
        [axes_initializer],
    )
    model_path = tmp_path / "reduce_sum_empty_axes.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    imported = ONNXImport(str(model_path), strict=True)
    reduce_ops = [op for op in imported if op.__class__.__name__ == "ReduceSum"]
    assert reduce_ops[0].noop_with_empty_axes == 0
    assert reduce_ops[1].noop_with_empty_axes == 1

    data = Tensor(2, 3, dtype="float32", data=np.array([[1, 0, 2], [0, 0, 3]], dtype=np.float32))
    nonzero = NonZero(["x"], ["out"]).forward_(data)["tensor"]
    assert nonzero.size == (2, 3)
