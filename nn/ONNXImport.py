import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping, Tensor_
from onnx import shape_inference
import traceback

class GenericNode:
    """
    通用占位节点：用于承载尚未实现、解析失败或自定义的算子。
    """
    def __init__(self, op_type, inputs, outputs, name=None, attributes=None, error=None):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.name = name if name else f"{op_type}_{outputs[0] if outputs else 'unknown'}"
        self.attributes = attributes if attributes else {}
        self.error = error

    def forward(self, *args):
        # 运行时占位
        return {"tensor": [None] * len(self.outputs), "parameters": None}

    def forward_(self, *args):
        # 图推断占位
        out_tensors = []
        for _ in self.outputs:
            out_tensors.append(Tensor_(1, dtype="float32"))
        res = out_tensors[0] if len(out_tensors) == 1 else out_tensors
        return {"tensor": res, "parameters": None, "graph": None}

    @property
    def parameters(self):
        info = []
        if self.error:
            info.append(f"error={self.error}")
        for k, v in self.attributes.items():
            val_str = str(v)
            if len(val_str) > 20: val_str = val_str[:17] + "..."
            info.append(f"{k}={val_str}")
        return {"info": "\\n".join(info)}

def ONNXImport(file_path, strict=False):
    """
    [Optimized] 从ONNX模型文件导入计算图节点

    Args:
        file_path: ONNX 模型路径。
        strict: 为 True 时，任何不支持或解析失败的节点都会直接抛错；
            为 False 时保留 GenericNode 占位，但会记录错误原因。
    """
    onnx_graph_list = []
    generic_nodes = []
    
    print(f"   [ONNXImport] Loading model from {file_path}...")
    try:
        onnx_model = onnx.load(file_path, load_external_data=False)
    except Exception as e:
        print(f" Critical Error: Failed to load ONNX file. {e}")
        raise e

    # =========================================================================
    # 预计算类型映射表
    # =========================================================================
    print("   [ONNXImport] Optimizing: Running shape inference ONCE...")
    try:
        # 全局执行一次形状推断
        inferred_model = shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        print(f" Warning: Shape inference failed ({e}). Falling back to raw model.")
        inferred_model = onnx_model

    print("   [ONNXImport] Optimizing: Building Tensor DType Map...")
    # 构建 Hash Map 实现 O(1) 查找
    dtype_map = {}
    graph = inferred_model.graph
    
    # 收集所有来源的 dtype 信息
    # 优先级: Initializer -> ValueInfo -> Input -> Output
    for t in graph.initializer: dtype_map[t.name] = t.data_type
    for t in graph.input: dtype_map[t.name] = t.type.tensor_type.elem_type
    for t in graph.output: dtype_map[t.name] = t.type.tensor_type.elem_type
    for t in graph.value_info: dtype_map[t.name] = t.type.tensor_type.elem_type

    # 内部辅助函数：获取 dtype
    def get_dtype(name, default=onnx.TensorProto.FLOAT):
        return dtype_map.get(name, default)

    # =========================================================================
    # 解析 Initializers
    # =========================================================================
    print("   [ONNXImport] Parsing Initializers...")
    for init in onnx_model.graph.initializer:
        try:
            val = numpy_helper.to_array(init)
            dtype = onnx_dtype_mapping.get(init.data_type, "float32")
            const_op = nn.Operators.Constant([], [init.name], value=val, dtype=dtype, version="17")
            onnx_graph_list.append(const_op)
        except Exception as e:
            print(f" Warning: Failed to convert initializer {init.name}: {e}")

    # =========================================================================
    # 解析 Nodes
    # =========================================================================
    total_nodes = len(onnx_model.graph.node)
    print(f"   [ONNXImport] Parsing {total_nodes} Nodes...")
    
    for i, node in enumerate(onnx_model.graph.node):
        # 进度提示
        if i > 0 and i % 1000 == 0:
            print(f"      -> Processed {i}/{total_nodes} nodes...")

        try:
            op_upper = node.op_type.upper()
            
            if op_upper == "RELU":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.RELU(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "COS":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.COS(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "ABS":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ABS(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "ADD":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ADD(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "SUB":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SUB(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "MUL":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MUL(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "DIV":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.DIV(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Conv":
                pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
                for attr in node.attribute:
                    if attr.name == "pads": pads = list(attr.ints)
                    elif attr.name == "strides": strides = list(attr.ints)
                    elif attr.name == "dilations": dilations = list(attr.ints)
                    elif attr.name == "group": group = attr.i
                    elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Conv(node.input, node.output, pads=pads, strides=strides, dilations=dilations, group=group, kernel_shape=kernel_shape, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "ConvTranspose":
                attrs = {
                    "pads": None,
                    "strides": None,
                    "dilations": None,
                    "group": 1,
                    "kernel_shape": None,
                    "output_padding": None,
                    "output_shape": None,
                    "auto_pad": "NOTSET",
                }
                for attr in node.attribute:
                    if attr.name in {"pads", "strides", "dilations", "kernel_shape", "output_padding", "output_shape"}:
                        attrs[attr.name] = list(attr.ints)
                    elif attr.name == "group":
                        attrs["group"] = attr.i
                    elif attr.name == "auto_pad":
                        attrs["auto_pad"] = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ConvTranspose(
                    node.input,
                    node.output,
                    pads=attrs["pads"],
                    strides=attrs["strides"],
                    dilations=attrs["dilations"],
                    group=attrs["group"],
                    kernel_shape=attrs["kernel_shape"],
                    output_padding=attrs["output_padding"],
                    output_shape=attrs["output_shape"],
                    auto_pad=attrs["auto_pad"],
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                ))
            elif node.op_type == "ConvInteger":
                pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
                for attr in node.attribute:
                    if attr.name == "pads": pads = list(attr.ints)
                    elif attr.name == "strides": strides = list(attr.ints)
                    elif attr.name == "dilations": dilations = list(attr.ints)
                    elif attr.name == "group": group = attr.i
                    elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                onnx_graph_list.append(nn.Operators.ConvInteger(
                    node.input,
                    node.output,
                    pads=pads,
                    strides=strides,
                    dilations=dilations,
                    group=group,
                    kernel_shape=kernel_shape,
                    auto_pad=auto_pad,
                    version="17",
                ))
            elif node.op_type == "QLinearConv":
                pads, strides, dilations, group, kernel_shape, auto_pad = None, None, None, 1, None, "NOTSET"
                for attr in node.attribute:
                    if attr.name == "pads": pads = list(attr.ints)
                    elif attr.name == "strides": strides = list(attr.ints)
                    elif attr.name == "dilations": dilations = list(attr.ints)
                    elif attr.name == "group": group = attr.i
                    elif attr.name == "kernel_shape": kernel_shape = list(attr.ints)
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.QLinearConv(
                    node.input,
                    node.output,
                    pads=pads,
                    strides=strides,
                    dilations=dilations,
                    group=group,
                    kernel_shape=kernel_shape,
                    auto_pad=auto_pad,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                ))
            elif node.op_type in ["RNN", "GRU", "LSTM"]:
                hidden_size, direction, clip, layout = None, "forward", None, 0
                activations, activation_alpha, activation_beta = None, None, None
                linear_before_reset, input_forget = 0, 0
                for attr in node.attribute:
                    if attr.name == "hidden_size": hidden_size = attr.i
                    elif attr.name == "direction": direction = attr.s.decode("utf-8")
                    elif attr.name == "clip": clip = attr.f
                    elif attr.name == "layout": layout = attr.i
                    elif attr.name == "activations": activations = [item.decode("utf-8") for item in attr.strings]
                    elif attr.name == "activation_alpha": activation_alpha = list(attr.floats)
                    elif attr.name == "activation_beta": activation_beta = list(attr.floats)
                    elif attr.name == "linear_before_reset": linear_before_reset = attr.i
                    elif attr.name == "input_forget": input_forget = attr.i
                out_name = next((name for name in node.output if name), node.input[0])
                elem_type = get_dtype(out_name)
                common = dict(
                    hidden_size=hidden_size,
                    direction=direction,
                    activations=activations,
                    activation_alpha=activation_alpha,
                    activation_beta=activation_beta,
                    clip=clip,
                    layout=layout,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                )
                if node.op_type == "RNN":
                    onnx_graph_list.append(nn.Operators.RNN(node.input, node.output, **common))
                elif node.op_type == "GRU":
                    onnx_graph_list.append(nn.Operators.GRU(node.input, node.output, linear_before_reset=linear_before_reset, **common))
                else:
                    onnx_graph_list.append(nn.Operators.LSTM(node.input, node.output, input_forget=input_forget, **common))
            elif node.op_type == "MaxPool":
                kernel_shape, pads, strides, dilations, auto_pad = [1,1], [0]*4, [1,1], [1,1], "NOTSET"
                ceil_mode, storage_order = 0, 0
                for attr in node.attribute:
                    if attr.name == "kernel_shape": kernel_shape = attr.ints
                    elif attr.name == "pads": pads = attr.ints
                    elif attr.name == "strides": strides = attr.ints
                    elif attr.name == "dilations": dilations = attr.ints
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                    elif attr.name == "ceil_mode": ceil_mode = attr.i
                    elif attr.name == "storage_order": storage_order = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MaxPool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, ceil_mode=ceil_mode, storage_order=storage_order, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MaxUnpool":
                kernel_shape, pads, strides = None, None, None
                for attr in node.attribute:
                    if attr.name == "kernel_shape": kernel_shape = list(attr.ints)
                    elif attr.name == "pads": pads = list(attr.ints)
                    elif attr.name == "strides": strides = list(attr.ints)
                if kernel_shape is None:
                    raise ValueError("MaxUnpool requires kernel_shape attribute")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MaxUnpool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MaxRoiPool":
                pooled_shape, spatial_scale = None, 1.0
                for attr in node.attribute:
                    if attr.name == "pooled_shape": pooled_shape = list(attr.ints)
                    elif attr.name == "spatial_scale": spatial_scale = attr.f
                if pooled_shape is None:
                    raise ValueError("MaxRoiPool requires pooled_shape attribute")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MaxRoiPool(node.input, node.output, pooled_shape=pooled_shape, spatial_scale=spatial_scale, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "RoiAlign":
                output_height, output_width, sampling_ratio = 1, 1, 0
                spatial_scale, mode, coord_mode = 1.0, "avg", "half_pixel"
                for attr in node.attribute:
                    if attr.name == "output_height": output_height = attr.i
                    elif attr.name == "output_width": output_width = attr.i
                    elif attr.name == "sampling_ratio": sampling_ratio = attr.i
                    elif attr.name == "spatial_scale": spatial_scale = attr.f
                    elif attr.name == "mode": mode = attr.s.decode("utf-8")
                    elif attr.name == "coordinate_transformation_mode": coord_mode = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.RoiAlign(
                    node.input,
                    node.output,
                    output_height=output_height,
                    output_width=output_width,
                    spatial_scale=spatial_scale,
                    sampling_ratio=sampling_ratio,
                    mode=mode,
                    coordinate_transformation_mode=coord_mode,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                ))
            elif node.op_type == "Gemm":
                alpha, beta, transA, transB = 1.0, 1.0, 0, 0
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                    elif attr.name == "beta": beta = attr.f
                    elif attr.name == "transA": transA = attr.i
                    elif attr.name == "transB": transB = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Gemm(node.input, node.output, alpha=alpha, beta=beta, transA=transA, transB=transB, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Softmax":
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Softmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "QuantizeLinear":
                if len(node.input) >= 3 and node.input[2]:
                    zp_name = node.input[2]
                    if zp_name in dtype_map:
                        target_dtype = onnx_dtype_mapping[dtype_map[zp_name]]
                    else:
                        raise ValueError(f"Unknown dtype for ZeroPoint {zp_name}")
                else:
                    try:
                        target_dtype = onnx_dtype_mapping[get_dtype(node.output[0])]
                    except Exception:
                        target_dtype = "uint8"
                axis = 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                onnx_graph_list.append(nn.Operators.QuantizeLinear(node.input, node.output, axis=axis, dtype=target_dtype, version="17"))
            elif node.op_type == "DequantizeLinear":
                axis = 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.DequantizeLinear(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "EXP":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.EXP(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "LOG":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LOG(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "SQRT":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SQRT(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "SIGMOID":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SIGMOID(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif op_upper == "TANH":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.TANH(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Flatten":
                axis = 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Flatten(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Reshape":
                allowzero = 0
                for attr in node.attribute:
                    if attr.name == "allowzero": allowzero = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Reshape(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17", allowzero=allowzero))
            elif node.op_type == "Transpose":
                perm = []
                for attr in node.attribute:
                    if attr.name == "perm": perm = attr.ints
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Transpose(node.input, node.output, perm=perm, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Pow":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Pow(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Max":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Max(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Min":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Min(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Squeeze":
                axes = None
                for attr in node.attribute:
                    if attr.name == "axes": axes = attr.ints
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Squeeze(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Unsqueeze":
                axes = None
                for attr in node.attribute:
                    if attr.name == "axes": axes = attr.ints
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Unsqueeze(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Concat":
                axis = 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Concat(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Slice":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Slice(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Neg":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Neg(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Reciprocal":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Reciprocal(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Ceil":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Ceil(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Floor":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Floor(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Cast":
                to = 1
                for attr in node.attribute:
                    if attr.name == "to": to = attr.i
                target_dtype = onnx_dtype_mapping.get(to, "float32")
                onnx_graph_list.append(nn.Operators.Cast(node.input, node.output, dtype=target_dtype, version="17"))
            elif node.op_type == "CastLike":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.CastLike(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Sum":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Sum(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "PRelu":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.PRelu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Det":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Det(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MatMulInteger":
                onnx_graph_list.append(nn.Operators.MatMulInteger(node.input, node.output, dtype="int32", version="17"))
            elif node.op_type == "QLinearMatMul":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.QLinearMatMul(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LRN":
                alpha, beta, bias, size = 0.0001, 0.75, 1.0, None
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                    elif attr.name == "beta": beta = attr.f
                    elif attr.name == "bias": bias = attr.f
                    elif attr.name == "size": size = attr.i
                if size is None:
                    raise ValueError("LRN requires size attribute")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LRN(node.input, node.output, size=size, alpha=alpha, beta=beta, bias=bias, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MeanVarianceNormalization":
                axes = [0, 2, 3]
                for attr in node.attribute:
                    if attr.name == "axes": axes = list(attr.ints)
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MeanVarianceNormalization(node.input, node.output, axes=axes, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "EyeLike":
                k, dtype = 0, None
                for attr in node.attribute:
                    if attr.name == "k": k = attr.i
                    elif attr.name == "dtype": dtype = attr.i
                elem_type = dtype if dtype is not None else get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.EyeLike(node.input, node.output, k=k, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Clip":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Clip(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MatMul":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.MatMul(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Gather":
                axis = 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Gather(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Expand":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Expand(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Shape":
                start, end = 0, None
                for attr in node.attribute:
                    if attr.name == "start": start = attr.i
                    elif attr.name == "end": end = attr.i
                onnx_graph_list.append(nn.Operators.Shape(node.input, node.output, start=start, end=end, dtype="int64", version="17"))
            elif node.op_type == "Constant":
                value, dtype = None, "float32"
                for attr in node.attribute:
                    if attr.name == "value":
                        value = numpy_helper.to_array(attr.t)
                        dtype = onnx_dtype_mapping[attr.t.data_type]
                    elif attr.name == "sparse_value":
                        value = numpy_helper.to_array(attr.sparse_tensor)
                        dtype = onnx_dtype_mapping[attr.sparse_tensor.values.data_type]
                    elif attr.name == "value_float":
                        value = np.asarray(attr.f, dtype=np.float32)
                        dtype = "float32"
                    elif attr.name == "value_floats":
                        value = np.asarray(list(attr.floats), dtype=np.float32)
                        dtype = "float32"
                    elif attr.name == "value_int":
                        value = np.asarray(attr.i, dtype=np.int64)
                        dtype = "int64"
                    elif attr.name == "value_ints":
                        value = np.asarray(list(attr.ints), dtype=np.int64)
                        dtype = "int64"
                    elif attr.name == "value_string":
                        value = np.asarray(attr.s.decode("utf-8", errors="ignore"), dtype=np.str_)
                        dtype = "string"
                    elif attr.name == "value_strings":
                        value = np.asarray([item.decode("utf-8", errors="ignore") for item in attr.strings], dtype=np.str_)
                        dtype = "string"
                onnx_graph_list.append(nn.Operators.Constant(node.input, node.output, value=value, dtype=dtype, version="17"))
            elif node.op_type in ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Xor", "IsNaN"]:
                cls_map = {
                    "Equal": nn.Operators.Equal, "Greater": nn.Operators.Greater, "Less": nn.Operators.Less,
                    "GreaterOrEqual": nn.Operators.GreaterOrEqual, "LessOrEqual": nn.Operators.LessOrEqual,
                    "Not": nn.Operators.Not, "And": nn.Operators.And, "Or": nn.Operators.Or, "Xor": nn.Operators.Xor, "IsNaN": nn.Operators.IsNaN
                }
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype="bool", version="17"))
            elif node.op_type in ["Sin", "Tan", "Atan", "Sign", "Identity", "Round", "Erf", "Softplus", "Softsign", "HardSwish", "Acos", "Asin", "Cosh", "Sinh", "Asinh", "Acosh", "Atanh", "Gelu", "Mish"]:
                elem_type = get_dtype(node.output[0])
                cls_map = {
                    "Sin": nn.Operators.Sin, "Tan": nn.Operators.Tan, "Atan": nn.Operators.Atan, "Sign": nn.Operators.Sign, "Identity": nn.Operators.Identity,
                    "Round": nn.Operators.Round, "Erf": nn.Operators.Erf, "Softplus": nn.Operators.Softplus, "Softsign": nn.Operators.Softsign, "HardSwish": nn.Operators.HardSwish,
                    "Acos": nn.Operators.Acos, "Asin": nn.Operators.Asin, "Cosh": nn.Operators.Cosh, "Sinh": nn.Operators.Sinh, "Asinh": nn.Operators.Asinh, "Acosh": nn.Operators.Acosh, "Atanh": nn.Operators.Atanh,
                    "Gelu": nn.Operators.Gelu, "Mish": nn.Operators.Mish
                }
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Mod":
                fmod = 0
                for attr in node.attribute:
                    if attr.name == "fmod": fmod = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Mod(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], fmod=fmod, version="17"))
            elif node.op_type == "Where":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Where(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "ConstantOfShape":
                value, target_dtype = None, "float32"
                for attr in node.attribute:
                    if attr.name == "value": value = numpy_helper.to_array(attr.t)
                if value is not None:
                    target_dtype = nn.NUMPY_TO_DTYPE.get(value.dtype.type, target_dtype)
                onnx_graph_list.append(nn.Operators.ConstantOfShape(node.input, node.output, value=value, dtype=target_dtype, version="17"))
            elif node.op_type == "Range":
                elem_type = get_dtype(node.input[0])
                onnx_graph_list.append(nn.Operators.Range(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Tile":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Tile(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Pad":
                mode = "constant"
                for attr in node.attribute:
                    if attr.name == "mode": mode = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Pad(node.input, node.output, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Split":
                axis = 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Split(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "SequenceEmpty":
                dtype = 1
                for attr in node.attribute:
                    if attr.name == "dtype": dtype = attr.i
                onnx_graph_list.append(nn.Operators.SequenceEmpty(node.input, node.output, dtype=onnx_dtype_mapping.get(dtype, "float32"), version="17"))
            elif node.op_type == "SequenceConstruct":
                elem_type = get_dtype(node.input[0]) if node.input else get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SequenceConstruct(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "SequenceAt":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SequenceAt(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "SequenceInsert":
                elem_type = get_dtype(node.input[1]) if len(node.input) > 1 else get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SequenceInsert(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "SequenceErase":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SequenceErase(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "SequenceLength":
                onnx_graph_list.append(nn.Operators.SequenceLength(node.input, node.output, version="17"))
            elif node.op_type == "ConcatFromSequence":
                axis, new_axis = 0, 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "new_axis": new_axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ConcatFromSequence(node.input, node.output, axis=axis, new_axis=new_axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "SplitToSequence":
                axis, keepdims = 0, 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "keepdims": keepdims = attr.i
                elem_type = get_dtype(node.input[0])
                onnx_graph_list.append(nn.Operators.SplitToSequence(node.input, node.output, axis=axis, keepdims=keepdims, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Optional":
                elem_type = get_dtype(node.input[0]) if node.input else get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Optional(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "OptionalGetElement":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.OptionalGetElement(node.input, node.output, dtype=onnx_dtype_mapping.get(elem_type, "float32"), version="17"))
            elif node.op_type == "OptionalHasElement":
                onnx_graph_list.append(nn.Operators.OptionalHasElement(node.input, node.output, version="17"))
            elif node.op_type == "If":
                then_branch, else_branch = None, None
                for attr in node.attribute:
                    if attr.name == "then_branch": then_branch = attr.g
                    elif attr.name == "else_branch": else_branch = attr.g
                if then_branch is None or else_branch is None:
                    raise ValueError("If requires then_branch and else_branch graphs")
                onnx_graph_list.append(nn.Operators.If(node.input, node.output, then_branch=then_branch, else_branch=else_branch, version="17"))
            elif node.op_type == "Loop":
                body = None
                for attr in node.attribute:
                    if attr.name == "body": body = attr.g
                if body is None:
                    raise ValueError("Loop requires body graph")
                onnx_graph_list.append(nn.Operators.Loop(node.input, node.output, body=body, version="17"))
            elif node.op_type == "Scan":
                body, num_scan_inputs = None, None
                scan_input_axes = scan_input_directions = scan_output_axes = scan_output_directions = None
                for attr in node.attribute:
                    if attr.name == "body": body = attr.g
                    elif attr.name == "num_scan_inputs": num_scan_inputs = attr.i
                    elif attr.name == "scan_input_axes": scan_input_axes = list(attr.ints)
                    elif attr.name == "scan_input_directions": scan_input_directions = list(attr.ints)
                    elif attr.name == "scan_output_axes": scan_output_axes = list(attr.ints)
                    elif attr.name == "scan_output_directions": scan_output_directions = list(attr.ints)
                if body is None or num_scan_inputs is None:
                    raise ValueError("Scan requires body graph and num_scan_inputs")
                onnx_graph_list.append(nn.Operators.Scan(
                    node.input,
                    node.output,
                    body=body,
                    num_scan_inputs=num_scan_inputs,
                    scan_input_axes=scan_input_axes,
                    scan_input_directions=scan_input_directions,
                    scan_output_axes=scan_output_axes,
                    scan_output_directions=scan_output_directions,
                    version="17",
                ))
            elif node.op_type == "SequenceMap":
                body = None
                for attr in node.attribute:
                    if attr.name == "body": body = attr.g
                if body is None:
                    raise ValueError("SequenceMap requires body graph")
                onnx_graph_list.append(nn.Operators.SequenceMap(node.input, node.output, body=body, version="17"))
            elif node.op_type in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"]:
                axes, keepdims, noop_with_empty_axes = None, 1, 0
                for attr in node.attribute:
                    if attr.name == "axes": axes = attr.ints
                    elif attr.name == "keepdims": keepdims = attr.i
                    elif attr.name == "noop_with_empty_axes": noop_with_empty_axes = attr.i
                elem_type = get_dtype(node.output[0])
                cls_map = {
                    "ReduceMean": nn.Operators.ReduceMean, "ReduceSum": nn.Operators.ReduceSum, "ReduceMax": nn.Operators.ReduceMax, "ReduceMin": nn.Operators.ReduceMin, "ReduceProd": nn.Operators.ReduceProd,
                    "ReduceL1": nn.Operators.ReduceL1, "ReduceL2": nn.Operators.ReduceL2, "ReduceLogSum": nn.Operators.ReduceLogSum, "ReduceLogSumExp": nn.Operators.ReduceLogSumExp, "ReduceSumSquare": nn.Operators.ReduceSumSquare
                }
                kwargs = {}
                if node.op_type == "ReduceSum":
                    kwargs["noop_with_empty_axes"] = noop_with_empty_axes
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, axes=axes, keepdims=keepdims, dtype=onnx_dtype_mapping[elem_type], version="17", **kwargs))
            elif node.op_type in ["ArgMax", "ArgMin"]:
                axis, keepdims, select_last_index = 0, 1, 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "keepdims": keepdims = attr.i
                    elif attr.name == "select_last_index": select_last_index = attr.i
                cls_map = {"ArgMax": nn.Operators.ArgMax, "ArgMin": nn.Operators.ArgMin}
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, axis=axis, keepdims=keepdims, select_last_index=select_last_index, dtype="int64", version="17"))
            elif node.op_type == "ScatterND":
                reduction = "none"
                for attr in node.attribute:
                    if attr.name == "reduction": reduction = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ScatterND(node.input, node.output, reduction=reduction, dtype=onnx_dtype_mapping[elem_type]))
            elif node.op_type == "GatherND":
                batch_dims = 0
                for attr in node.attribute:
                    if attr.name == "batch_dims": batch_dims = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GatherND(node.input, node.output, batch_dims=batch_dims, dtype=onnx_dtype_mapping[elem_type]))
            elif node.op_type == "GatherElements":
                axis = 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GatherElements(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type]))
            elif node.op_type == "NonZero":
                onnx_graph_list.append(nn.Operators.NonZero(node.input, node.output))
            elif node.op_type == "NonMaxSuppression":
                center_point_box = 0
                for attr in node.attribute:
                    if attr.name == "center_point_box": center_point_box = attr.i
                onnx_graph_list.append(nn.Operators.NonMaxSuppression(node.input, node.output, center_point_box=center_point_box, version="17"))
            elif node.op_type == "Resize":
                mode, coord_mode, nearest_mode = "nearest", "asymmetric", "round_prefer_floor"
                cubic_coeff_a, exclude_outside, extrapolation_value = -0.75, 0, 0.0
                for attr in node.attribute:
                    if attr.name == "mode": mode = attr.s.decode('utf-8')
                    elif attr.name == "coordinate_transformation_mode": coord_mode = attr.s.decode('utf-8')
                    elif attr.name == "nearest_mode": nearest_mode = attr.s.decode('utf-8')
                    elif attr.name == "cubic_coeff_a": cubic_coeff_a = attr.f
                    elif attr.name == "exclude_outside": exclude_outside = attr.i
                    elif attr.name == "extrapolation_value": extrapolation_value = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Resize(
                    node.input,
                    node.output,
                    mode=mode,
                    coord_mode=coord_mode,
                    nearest_mode=nearest_mode,
                    cubic_coeff_a=cubic_coeff_a,
                    exclude_outside=exclude_outside,
                    extrapolation_value=extrapolation_value,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                ))
            elif node.op_type == "GridSample":
                mode, padding_mode, align_corners = "bilinear", "zeros", 0
                for attr in node.attribute:
                    if attr.name == "mode": mode = attr.s.decode("utf-8")
                    elif attr.name == "padding_mode": padding_mode = attr.s.decode("utf-8")
                    elif attr.name == "align_corners": align_corners = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GridSample(
                    node.input,
                    node.output,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17",
                ))
            elif node.op_type == "TopK":
                axis, largest, sorted_ = -1, 1, 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "largest": largest = attr.i
                    elif attr.name == "sorted": sorted_ = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.TopK(node.input, node.output, axis=axis, largest=largest, sorted=sorted_, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "CumSum":
                exclusive, reverse = 0, 0
                for attr in node.attribute:
                    if attr.name == "exclusive": exclusive = attr.i
                    elif attr.name == "reverse": reverse = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.CumSum(node.input, node.output, exclusive=exclusive, reverse=reverse, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "RandomUniformLike":
                high, low, seed, dtype_attr = 1.0, 0.0, 0.0, None
                for attr in node.attribute:
                    if attr.name == "high": high = attr.f
                    elif attr.name == "low": low = attr.f
                    elif attr.name == "seed": seed = attr.f
                    elif attr.name == "dtype": dtype_attr = attr.i
                target_dtype = onnx_dtype_mapping.get(dtype_attr, "float32") if dtype_attr is not None else None
                onnx_graph_list.append(nn.Operators.RandomUniformLike(node.input, node.output, high=high, low=low, seed=seed, dtype=target_dtype, version="17"))
            elif node.op_type == "RandomUniform":
                high, low, seed, dtype_attr, shape = 1.0, 0.0, 0.0, 1, []
                for attr in node.attribute:
                    if attr.name == "high": high = attr.f
                    elif attr.name == "low": low = attr.f
                    elif attr.name == "seed": seed = attr.f
                    elif attr.name == "dtype": dtype_attr = attr.i
                    elif attr.name == "shape": shape = attr.ints
                onnx_graph_list.append(nn.Operators.RandomUniform(node.input, node.output, high=high, low=low, seed=seed, dtype=dtype_attr, shape=shape, version="17"))
            elif node.op_type == "Multinomial":
                dtype_attr, sample_size, seed = onnx.TensorProto.INT32, 1, None
                for attr in node.attribute:
                    if attr.name == "dtype": dtype_attr = attr.i
                    elif attr.name == "sample_size": sample_size = attr.i
                    elif attr.name == "seed": seed = attr.f
                onnx_graph_list.append(nn.Operators.Multinomial(node.input, node.output, dtype=dtype_attr, sample_size=sample_size, seed=seed, version="17"))
            elif node.op_type == "NegativeLogLikelihoodLoss":
                reduction, ignore_index = "mean", None
                for attr in node.attribute:
                    if attr.name == "reduction": reduction = attr.s.decode("utf-8")
                    elif attr.name == "ignore_index": ignore_index = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.NegativeLogLikelihoodLoss(node.input, node.output, reduction=reduction, ignore_index=ignore_index, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "SoftmaxCrossEntropyLoss":
                reduction, ignore_index = "mean", None
                for attr in node.attribute:
                    if attr.name == "reduction": reduction = attr.s.decode("utf-8")
                    elif attr.name == "ignore_index": ignore_index = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SoftmaxCrossEntropyLoss(node.input, node.output, reduction=reduction, ignore_index=ignore_index, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "MelWeightMatrix":
                output_datatype = onnx.TensorProto.FLOAT
                for attr in node.attribute:
                    if attr.name == "output_datatype": output_datatype = attr.i
                onnx_graph_list.append(nn.Operators.MelWeightMatrix(node.input, node.output, output_datatype=output_datatype, version="17"))
            elif node.op_type == "DFT":
                axis, inverse, onesided = 1, 0, 0
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "inverse": inverse = attr.i
                    elif attr.name == "onesided": onesided = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.DFT(node.input, node.output, axis=axis, inverse=inverse, onesided=onesided, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "STFT":
                onesided = 1
                for attr in node.attribute:
                    if attr.name == "onesided": onesided = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.STFT(node.input, node.output, onesided=onesided, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Unique":
                axis, sorted_ = None, 1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "sorted": sorted_ = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Unique(node.input, node.output, axis=axis, sorted=sorted_, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Einsum":
                equation = ""
                for attr in node.attribute:
                    if attr.name == "equation": equation = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Einsum(node.input, node.output, equation=equation, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Upsample":
                mode = "nearest"
                for attr in node.attribute:
                    if attr.name == "mode": mode = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Resize(node.input, node.output, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Elu":
                alpha = 1.0
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Elu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Selu":
                alpha, gamma = 1.67326, 1.0507
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                    elif attr.name == "gamma": gamma = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Selu(node.input, node.output, alpha=alpha, gamma=gamma, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LeakyRelu":
                alpha = 0.01
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LeakyRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "ThresholdedRelu":
                alpha = 1.0
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ThresholdedRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "HardSigmoid":
                alpha, beta = 0.2, 0.5
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                    elif attr.name == "beta": beta = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.HardSigmoid(node.input, node.output, alpha=alpha, beta=beta, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Celu":
                alpha = 1.0
                for attr in node.attribute:
                    if attr.name == "alpha": alpha = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Celu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Shrink":
                bias, lambd = 0.0, 0.5
                for attr in node.attribute:
                    if attr.name == "bias": bias = attr.f
                    elif attr.name == "lambd": lambd = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Shrink(node.input, node.output, bias=bias, lambd=lambd, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type in ["BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot"]:
                elem_type = get_dtype(node.output[0])
                cls_map = {"BitwiseAnd": nn.Operators.BitwiseAnd, "BitwiseOr": nn.Operators.BitwiseOr, "BitwiseXor": nn.Operators.BitwiseXor, "BitwiseNot": nn.Operators.BitwiseNot}
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "BitShift":
                direction = "LEFT"
                for attr in node.attribute:
                    if attr.name == "direction": direction = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.BitShift(node.input, node.output, direction=direction, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "AveragePool":
                kernel_shape, pads, strides, dilations, count_include_pad, ceil_mode, auto_pad = [1,1], [0]*4, [1,1], [1,1], 0, 0, "NOTSET"
                for attr in node.attribute:
                    if attr.name == "kernel_shape": kernel_shape = attr.ints
                    elif attr.name == "pads": pads = attr.ints
                    elif attr.name == "strides": strides = attr.ints
                    elif attr.name == "dilations": dilations = attr.ints
                    elif attr.name == "count_include_pad": count_include_pad = attr.i
                    elif attr.name == "ceil_mode": ceil_mode = attr.i
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.AveragePool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, count_include_pad=count_include_pad, ceil_mode=ceil_mode, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LpPool":
                kernel_shape, pads, strides, dilations, p, ceil_mode, auto_pad = [1,1], [0]*4, [1,1], [1,1], 2, 0, "NOTSET"
                for attr in node.attribute:
                    if attr.name == "kernel_shape": kernel_shape = attr.ints
                    elif attr.name == "pads": pads = attr.ints
                    elif attr.name == "strides": strides = attr.ints
                    elif attr.name == "dilations": dilations = attr.ints
                    elif attr.name == "p": p = attr.i
                    elif attr.name == "ceil_mode": ceil_mode = attr.i
                    elif attr.name == "auto_pad": auto_pad = attr.s.decode("utf-8")
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LpPool(node.input, node.output, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations, p=p, ceil_mode=ceil_mode, auto_pad=auto_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "GlobalAveragePool":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GlobalAveragePool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "GlobalMaxPool":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GlobalMaxPool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "GlobalLpPool":
                p = 2
                for attr in node.attribute:
                    if attr.name == "p": p = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GlobalLpPool(node.input, node.output, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Mean":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Mean(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Size":
                onnx_graph_list.append(nn.Operators.Size(node.input, node.output, dtype="int64", version="17"))
            elif node.op_type == "IsInf":
                detect_neg, detect_pos = 1, 1
                for attr in node.attribute:
                    if attr.name == "detect_negative": detect_neg = attr.i
                    elif attr.name == "detect_positive": detect_pos = attr.i
                onnx_graph_list.append(nn.Operators.IsInf(node.input, node.output, detect_negative=detect_neg, detect_positive=detect_pos, version="17"))
            elif node.op_type == "OneHot":
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.input[2]) # Values dtype
                onnx_graph_list.append(nn.Operators.OneHot(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Tril":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Tril(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Triu":
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Triu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Trilu":
                upper = 1
                for attr in node.attribute:
                    if attr.name == "upper": upper = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Trilu(node.input, node.output, upper=upper, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "BatchNormalization":
                epsilon, momentum, training_mode = 1e-5, 0.9, 0
                for attr in node.attribute:
                    if attr.name == "epsilon": epsilon = attr.f
                    elif attr.name == "momentum": momentum = attr.f
                    elif attr.name == "training_mode": training_mode = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.BatchNormalization(node.input, node.output, epsilon=epsilon, momentum=momentum, training_mode=training_mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "InstanceNormalization":
                epsilon = 1e-5
                for attr in node.attribute:
                    if attr.name == "epsilon": epsilon = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.InstanceNormalization(node.input, node.output, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LayerNormalization":
                epsilon, axis, stash_type = 1e-5, -1, 1
                for attr in node.attribute:
                    if attr.name == "epsilon": epsilon = attr.f
                    elif attr.name == "axis": axis = attr.i
                    elif attr.name == "stash_type": stash_type = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LayerNormalization(node.input, node.output, axis=axis, epsilon=epsilon, stash_type=stash_type, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type in ["HannWindow", "HammingWindow", "BlackmanWindow"]:
                periodic, output_datatype = 1, 1
                for attr in node.attribute:
                    if attr.name == "periodic": periodic = attr.i
                    elif attr.name == "output_datatype": output_datatype = attr.i
                cls_map = {"HannWindow": nn.Operators.HannWindow, "HammingWindow": nn.Operators.HammingWindow, "BlackmanWindow": nn.Operators.BlackmanWindow}
                onnx_graph_list.append(cls_map[node.op_type](node.input, node.output, periodic=periodic, output_datatype=output_datatype, version="17"))
            elif node.op_type == "RandomNormal":
                mean, scale, seed, dtype, shape = 0.0, 1.0, 0.0, 1, []
                for attr in node.attribute:
                    if attr.name == "mean": mean = attr.f
                    elif attr.name == "scale": scale = attr.f
                    elif attr.name == "seed": seed = attr.f
                    elif attr.name == "dtype": dtype = attr.i
                    elif attr.name == "shape": shape = attr.ints
                onnx_graph_list.append(nn.Operators.RandomNormal(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, shape=shape, version="17"))
            elif node.op_type == "RandomNormalLike":
                mean, scale, seed, dtype = 0.0, 1.0, 0.0, None
                for attr in node.attribute:
                    if attr.name == "mean": mean = attr.f
                    elif attr.name == "scale": scale = attr.f
                    elif attr.name == "seed": seed = attr.f
                    elif attr.name == "dtype": dtype = attr.i
                onnx_graph_list.append(nn.Operators.RandomNormalLike(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, version="17"))
            elif node.op_type == "Bernoulli":
                seed, dtype = 0.0, None
                for attr in node.attribute:
                    if attr.name == "seed": seed = attr.f
                    elif attr.name == "dtype": dtype = attr.i
                onnx_graph_list.append(nn.Operators.Bernoulli(node.input, node.output, seed=seed, dtype=dtype, version="17"))
            elif node.op_type == "Dropout":
                seed, ratio, training_mode = None, 0.5, 0
                for attr in node.attribute:
                    if attr.name == "seed": seed = attr.i if attr.type == onnx.AttributeProto.INT else attr.f
                    elif attr.name == "ratio": ratio = attr.f
                    elif attr.name == "training_mode": training_mode = attr.i
                onnx_graph_list.append(nn.Operators.Dropout(node.input, node.output, seed=seed, ratio=ratio, training_mode=training_mode, version="17"))
            elif node.op_type == "Hardmax":
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Hardmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LogSoftmax":
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LogSoftmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "LpNormalization":
                axis, p = -1, 2
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "p": p = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.LpNormalization(node.input, node.output, axis=axis, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "DepthToSpace":
                blocksize, mode = 1, "DCR"
                for attr in node.attribute:
                    if attr.name == "blocksize": blocksize = attr.i
                    elif attr.name == "mode": mode = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.DepthToSpace(node.input, node.output, blocksize=blocksize, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "SpaceToDepth":
                blocksize = 1
                for attr in node.attribute:
                    if attr.name == "blocksize": blocksize = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.SpaceToDepth(node.input, node.output, blocksize=blocksize, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "ReverseSequence":
                time_axis, batch_axis = 0, 1
                for attr in node.attribute:
                    if attr.name == "time_axis": time_axis = attr.i
                    elif attr.name == "batch_axis": batch_axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ReverseSequence(node.input, node.output, time_axis=time_axis, batch_axis=batch_axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "Compress":
                axis = None
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Compress(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type in ["ScatterElements", "Scatter"]:
                axis, reduction = 0, "none"
                for attr in node.attribute:
                    if attr.name == "axis": axis = attr.i
                    elif attr.name == "reduction": reduction = attr.s.decode('utf-8')
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.ScatterElements(node.input, node.output, axis=axis, reduction=reduction, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "GroupNormalization":
                num_groups, epsilon = 1, 1e-5
                for attr in node.attribute:
                    if attr.name == "num_groups": num_groups = attr.i
                    elif attr.name == "epsilon": epsilon = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.GroupNormalization(node.input, node.output, num_groups=num_groups, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "StringNormalizer":
                case_change_action, is_case_sensitive, locale, stopwords = "NONE", 0, "", []
                for attr in node.attribute:
                    if attr.name == "case_change_action": case_change_action = attr.s.decode("utf-8")
                    elif attr.name == "is_case_sensitive": is_case_sensitive = attr.i
                    elif attr.name == "locale": locale = attr.s.decode("utf-8")
                    elif attr.name == "stopwords": stopwords = [item.decode("utf-8") for item in attr.strings]
                onnx_graph_list.append(nn.Operators.StringNormalizer(
                    node.input,
                    node.output,
                    case_change_action=case_change_action,
                    is_case_sensitive=is_case_sensitive,
                    locale=locale,
                    stopwords=stopwords,
                    version="17",
                ))
            elif node.op_type == "TfIdfVectorizer":
                attrs = {
                    "mode": None,
                    "ngram_counts": None,
                    "ngram_indexes": None,
                    "max_skip_count": None,
                    "min_gram_length": None,
                    "max_gram_length": None,
                    "pool_int64s": [],
                    "pool_strings": [],
                    "weights": [],
                }
                for attr in node.attribute:
                    if attr.name == "mode": attrs["mode"] = attr.s.decode("utf-8")
                    elif attr.name == "ngram_counts": attrs["ngram_counts"] = list(attr.ints)
                    elif attr.name == "ngram_indexes": attrs["ngram_indexes"] = list(attr.ints)
                    elif attr.name == "max_skip_count": attrs["max_skip_count"] = attr.i
                    elif attr.name == "min_gram_length": attrs["min_gram_length"] = attr.i
                    elif attr.name == "max_gram_length": attrs["max_gram_length"] = attr.i
                    elif attr.name == "pool_int64s": attrs["pool_int64s"] = list(attr.ints)
                    elif attr.name == "pool_strings": attrs["pool_strings"] = [item.decode("utf-8") for item in attr.strings]
                    elif attr.name == "weights": attrs["weights"] = list(attr.floats)
                required = ["mode", "ngram_counts", "ngram_indexes", "max_skip_count", "min_gram_length", "max_gram_length"]
                missing = [name for name in required if attrs[name] is None]
                if missing:
                    raise ValueError(f"TfIdfVectorizer missing required attribute(s): {missing}")
                onnx_graph_list.append(nn.Operators.TfIdfVectorizer(
                    node.input,
                    node.output,
                    mode=attrs["mode"],
                    ngram_counts=attrs["ngram_counts"],
                    ngram_indexes=attrs["ngram_indexes"],
                    max_skip_count=attrs["max_skip_count"],
                    min_gram_length=attrs["min_gram_length"],
                    max_gram_length=attrs["max_gram_length"],
                    pool_int64s=attrs["pool_int64s"],
                    pool_strings=attrs["pool_strings"],
                    weights=attrs["weights"],
                    version="17",
                ))
            elif node.op_type == "Binarizer":
                threshold = 0.0
                for attr in node.attribute:
                    if attr.name == "threshold": threshold = attr.f
                elem_type = get_dtype(node.output[0])
                onnx_graph_list.append(nn.Operators.Binarizer(node.input, node.output, threshold=threshold, dtype=onnx_dtype_mapping[elem_type], version="17"))
            elif node.op_type == "DynamicQuantizeLinear":
                onnx_graph_list.append(nn.Operators.DynamicQuantizeLinear(node.input, node.output, version="17"))
            else:
                raise NotImplementedError(f"Operator {node.op_type} is not implemented.")
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if strict:
                raise RuntimeError(
                    f"Failed to import node #{i} ({node.op_type}, name={node.name or '<unnamed>'})"
                ) from e

            attrs = {}
            for attr in node.attribute:
                try:
                    if attr.type == onnx.AttributeProto.FLOAT: val = attr.f
                    elif attr.type == onnx.AttributeProto.INT: val = attr.i
                    elif attr.type == onnx.AttributeProto.STRING: val = attr.s.decode('utf-8', errors='ignore')
                    elif attr.type == onnx.AttributeProto.INTS: val = list(attr.ints)
                    elif attr.type == onnx.AttributeProto.FLOATS: val = list(attr.floats)
                    elif attr.type == onnx.AttributeProto.TENSOR: val = "<Tensor>"
                    elif attr.type == onnx.AttributeProto.GRAPH: val = "<Graph>"
                    else: val = f"<Type {attr.type}>"
                    attrs[attr.name] = val
                except Exception:
                    attrs[attr.name] = "?"

            generic_op = GenericNode(
                op_type=node.op_type,
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                attributes=attrs,
                error=error_msg
            )
            onnx_graph_list.append(generic_op)
            generic_nodes.append(generic_op)
            print(
                f" Warning: Node #{i} ({node.op_type}, name={node.name or '<unnamed>'}) "
                f"was downgraded to GenericNode: {error_msg}"
            )

    if generic_nodes:
        print(
            f"   [ONNXImport] Warning: {len(generic_nodes)} node(s) were imported as GenericNode. "
            "Use strict=True to fail on unsupported or invalid nodes."
        )
    return onnx_graph_list
