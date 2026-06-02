import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Egor Izmaylov: Function `make_const` implements the make const step for the ONNX fixture generator, normalizing inputs and returning the exact data or metadata contract expected downstream.
def make_const(name: str, value: np.ndarray):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(value, name=name + "_value"),
    )

# Egor Izmaylov: Function `export_graph_ops_model` implements the export graph ops model step for the ONNX fixture generator, normalizing inputs and returning the exact data or metadata contract expected downstream.
def export_graph_ops_model(out_path="./onnx_model/model.onnx"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    nodes = []
    initializers = []

    # 一个固定 shape 的输入，便于 verify_graph 做 shape 推导
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 32, 32])

    # ---------- 1) Shape ----------
    nodes.append(helper.make_node("Shape", ["X"], ["shape_of_x"], name="Shape_X"))

    # ---------- 2) Unsqueeze / Squeeze ----------
    unsq_axes = np.array([0], dtype=np.int64)
    nodes.append(make_const("unsq_axes", unsq_axes))
    nodes.append(helper.make_node("Unsqueeze", ["X", "unsq_axes"], ["unsq"], name="Unsqueeze_X"))

    sq_axes = np.array([0], dtype=np.int64)
    nodes.append(make_const("sq_axes", sq_axes))
    nodes.append(helper.make_node("Squeeze", ["unsq", "sq_axes"], ["sq"], name="Squeeze_unsq"))

    # ---------- 3) Slice ----------
    # slice sq[:, :, 2:10, 3:19]
    starts = np.array([2, 3], dtype=np.int64)
    ends   = np.array([10, 19], dtype=np.int64)
    axes   = np.array([2, 3], dtype=np.int64)
    steps  = np.array([1, 1], dtype=np.int64)

    nodes.append(make_const("slice_starts", starts))
    nodes.append(make_const("slice_ends", ends))
    nodes.append(make_const("slice_axes", axes))
    nodes.append(make_const("slice_steps", steps))

    nodes.append(helper.make_node(
        "Slice",
        ["sq", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
        ["sl"],
        name="Slice_sq",
    ))

    # ---------- 4) Transpose ----------
    nodes.append(helper.make_node("Transpose", ["sl"], ["tr"], name="Transpose_sl", perm=[0, 1, 3, 2]))

    # ---------- 5) Concat ----------
    nodes.append(helper.make_node("Concat", ["tr", "tr"], ["cat"], name="Concat_tr", axis=1))

    # ---------- 6) Reshape ----------
    reshape_shape = np.array([1, -1], dtype=np.int64)
    nodes.append(make_const("reshape_shape", reshape_shape))
    nodes.append(helper.make_node("Reshape", ["cat", "reshape_shape"], ["rs"], name="Reshape_cat"))

    # ---------- 7) Expand ----------
    # a: [1,3] -> expand to [2,3]
    a_val = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expand_shape = np.array([2, 3], dtype=np.int64)
    nodes.append(make_const("a", a_val))
    nodes.append(make_const("expand_shape", expand_shape))
    nodes.append(helper.make_node("Expand", ["a", "expand_shape"], ["ex"], name="Expand_a"))

    # ---------- 8) Where ----------
    # cond(bool[2,3]), x(float[2,3]), y(float[2,3])
    cond_val = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    x_val = np.array([[1.0, -1.0, 2.0], [3.0, -2.0, 0.5]], dtype=np.float32)
    y_val = np.array([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    nodes.append(make_const("cond", cond_val))
    nodes.append(make_const("wx", x_val))
    nodes.append(make_const("wy", y_val))
    nodes.append(helper.make_node("Where", ["cond", "wx", "wy"], ["wh"], name="Where_demo"))

    # ---------- 9) Cast ----------
    # 把 where 输出 cast 成 int64
    nodes.append(helper.make_node("Cast", ["wh"], ["wh_i64"], name="Cast_wh", to=TensorProto.INT64))

    # ---------- 10) Range ----------
    # range(0, 4, 1) -> int64[4]
    nodes.append(make_const("r_start", np.array(0, dtype=np.int64)))
    nodes.append(make_const("r_limit", np.array(4, dtype=np.int64)))
    nodes.append(make_const("r_delta", np.array(1, dtype=np.int64)))
    nodes.append(helper.make_node("Range", ["r_start", "r_limit", "r_delta"], ["rg"], name="Range_demo"))

    # ---------- 11) ConstantOfShape ----------
    # shape=[2,3] -> ones tensor
    nodes.append(make_const("cos_shape", np.array([2, 3], dtype=np.int64)))
    # ConstantOfShape 的 value 是一个标量 tensor attribute
    cos_value = helper.make_tensor("cos_value", TensorProto.FLOAT, [1], [1.0])
    nodes.append(helper.make_node("ConstantOfShape", ["cos_shape"], ["cos"], name="ConstantOfShape_demo", value=cos_value))

    # ---------- Graph Outputs ----------
    # 多输出没问题，verify_graph 会逐节点推导
    Y0 = helper.make_tensor_value_info("wh_i64", TensorProto.INT64, [2, 3])      # Cast output
    Y1 = helper.make_tensor_value_info("shape_of_x", TensorProto.INT64, [4])     # Shape output
    Y2 = helper.make_tensor_value_info("rg", TensorProto.INT64, [4])             # Range output
    Y3 = helper.make_tensor_value_info("cos", TensorProto.FLOAT, [2, 3])         # ConstantOfShape output
    Y4 = helper.make_tensor_value_info("rs", TensorProto.FLOAT, [1, None])       # Reshape output (第二维未知)

    graph = helper.make_graph(
        nodes=nodes,
        name="graph_ops_cover",
        inputs=[X],
        outputs=[Y0, Y1, Y2, Y3, Y4],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="onnx_translator_graph_ops",
    )

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    export_graph_ops_model()
