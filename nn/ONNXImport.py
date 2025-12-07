import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping
from onnx import shape_inference


def get_tensor_dtype(tensor_name, model):
    """
    获取张量的数据类型
    
    Args:
        tensor_name: 张量名称
        model: ONNX模型对象
        
    Returns:
        int: ONNX数据类型编码，如果未找到返回None
    """
    # 对模型进行形状推断
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph
    
    # 检查 Initializer (常量)
    for init_tensor in model.graph.initializer:
        if init_tensor.name == tensor_name:
            return init_tensor.data_type
        
    # 在图的输入中查找
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            return input_tensor.type.tensor_type.elem_type
            
    # 在图的输出中查找
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            return output_tensor.type.tensor_type.elem_type
            
    # 在图的value_info中查找
    for value_info_tensor in graph.value_info:
        if value_info_tensor.name == tensor_name:
            return value_info_tensor.type.tensor_type.elem_type
            
    return None


def ONNXImport(file_path):
    """
    从ONNX模型文件导入计算图节点
    
    Args:
        file_path: ONNX模型文件路径
        
    Returns:
        list: 包含操作节点的列表
    """
    onnx_graph_list = []
    
    # 加载ONNX模型
    onnx_model = onnx.load(file_path, load_external_data=False)
    
    # 提取初始化器的形状信息
    initializer_shapes = {}
    for init in onnx_model.graph.initializer:
        shape = [dim for dim in init.dims]
        initializer_shapes[init.name] = shape
        
    # 遍历图中的每个节点
    for node in onnx_model.graph.node:
        if node.op_type.upper() == "RELU":
            # 处理ReLU操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            # print("relu node data type", onnx_dtype_mapping[elem_type])
            onnx_graph_list.append(
                nn.Operators.__getattribute__("RELU")(node.input,
                                                      node.output,
                                                      dtype=onnx_dtype_mapping[elem_type],
                                                      version="17"))
        elif node.op_type.upper() == "COS":
            # 处理COS操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("COS")(node.input,
                                                     node.output,
                                                     dtype=onnx_dtype_mapping[elem_type],
                                                     version="17"))
        elif node.op_type.upper() == "ABS":
            # 处理ABS操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("ABS")(node.input,
                                                     node.output,
                                                     dtype=onnx_dtype_mapping[elem_type],
                                                     version="17"))  
        elif node.op_type.upper() == "ADD":
            # 处理ADD操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("ADD")(node.input,
                                                    node.output,
                                                    dtype=onnx_dtype_mapping[elem_type],
                                                    version="17"))
        elif node.op_type.upper() == "SUB":
            # 处理SUB操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("SUB")(node.input,
                                                    node.output,
                                                    dtype=onnx_dtype_mapping[elem_type],
                                                    version="17"))                                               
        elif node.op_type.upper() == "MUL":
            # 处理MUL操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("MUL")(node.input,
                                                    node.output,
                                                    dtype=onnx_dtype_mapping[elem_type],
                                                    version="17"))
        elif node.op_type.upper() == "DIV":
            # 处理DIV操作节点
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.__getattribute__("DIV")(node.input,
                                                    node.output,
                                                    dtype=onnx_dtype_mapping[elem_type],
                                                    version="17"))
        elif node.op_type == "Conv":
            pads = [0, 0, 0, 0]
            strides = [1, 1]
            dilations = [1, 1]
            group = 1
            for attr in node.attribute:
                if attr.name == "pads": pads = attr.ints
                elif attr.name == "strides": strides = attr.ints
                elif attr.name == "dilations": dilations = attr.ints
                elif attr.name == "group": group = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Conv(
                    node.input, node.output, 
                    pads=pads, strides=strides, dilations=dilations, group=group,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17"))
        elif node.op_type == "MaxPool":
            kernel_shape = [1, 1]
            pads = [0, 0, 0, 0]
            strides = [1, 1]
            dilations = [1, 1]
            for attr in node.attribute:
                if attr.name == "kernel_shape": kernel_shape = attr.ints
                elif attr.name == "pads": pads = attr.ints
                elif attr.name == "strides": strides = attr.ints
                elif attr.name == "dilations": dilations = attr.ints
                elif attr.name == "auto_pad": pass
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.MaxPool(
                    node.input, node.output, 
                    kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17")) 
        elif node.op_type == "Gemm":
            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
                elif attr.name == "beta": beta = attr.f
                elif attr.name == "transA": transA = attr.i
                elif attr.name == "transB": transB = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Gemm(
                    node.input, node.output, 
                    alpha=alpha, beta=beta, transA=transA, transB=transB,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17"))  
        elif node.op_type == "Softmax":
            axis = -1 # 默认最后一维
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Softmax(
                    node.input, node.output, axis=axis,
                    dtype=onnx_dtype_mapping[elem_type],
                    version="17"))  
        elif node.op_type == "QuantizeLinear":
            # 关键：通过 ZeroPoint (input[2]) 确定输出类型
            zp_name = node.input[2]
            elem_type = get_tensor_dtype(zp_name, onnx_model)
            if elem_type is None:
                # 尝试做一次 shape inference 作为最后手段
                print(f"⚠️ Warning: Inferring shapes for {zp_name}...")
                inferred_model = shape_inference.infer_shapes(onnx_model)
                elem_type = get_tensor_dtype(zp_name, inferred_model)
            if elem_type is None:
                raise ValueError(f"❌ Error: Could not determine dtype for ZeroPoint '{zp_name}' in node {node.name}. "
                                 "Cannot proceed with default, as it risks signed/unsigned mismatch.")
            target_dtype = onnx_dtype_mapping[elem_type]
            axis = 1 
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            onnx_graph_list.append(
                nn.Operators.QuantizeLinear(node.input, node.output, axis=axis, dtype=target_dtype, version="17"))
        elif node.op_type == "DequantizeLinear":
            # Dequantize 通常输出 float32，但也可能根据后续节点不同
            # 尝试推断 output[0] 的类型
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            # 如果找不到，Dequantize 默认为 float32 通常是安全的
            target_dtype = onnx_dtype_mapping[elem_type] if elem_type else "float32"
            onnx_graph_list.append(
                nn.Operators.DequantizeLinear(node.input, node.output, dtype=target_dtype, version="17"))
        elif node.op_type.upper() == "EXP":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.EXP(node.input, node.output, 
                                 dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type.upper() == "LOG":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.LOG(node.input, node.output, 
                                 dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type.upper() == "SQRT":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.SQRT(node.input, node.output, 
                                  dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type.upper() == "SIGMOID":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.SIGMOID(node.input, node.output, 
                                     dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type.upper() == "TANH":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.TANH(node.input, node.output, 
                                  dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Flatten":
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Flatten(node.input, node.output, axis=axis,
                                     dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Reshape":
            # Reshape 有两个输入: data, shape
            # shape 是 tensor，不是属性
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Reshape(node.input, node.output, 
                                     dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Transpose":
            perm = []
            for attr in node.attribute:
                if attr.name == "perm": perm = attr.ints
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Transpose(node.input, node.output, perm=perm,
                                       dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Pow":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Pow(node.input, node.output, 
                                 dtype=onnx_dtype_mapping[elem_type], version="17"))

        elif node.op_type == "Max":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Max(node.input, node.output, 
                                 dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Min":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Min(node.input, node.output, 
                                 dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Squeeze":
            axes = None
            for attr in node.attribute:
                if attr.name == "axes": axes = attr.ints
            
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Squeeze(node.input, node.output, axes=axes,
                                     dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Unsqueeze":
            axes = None
            for attr in node.attribute:
                if attr.name == "axes": axes = attr.ints
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Unsqueeze(node.input, node.output, axes=axes,
                                       dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Concat":
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Concat(node.input, node.output, axis=axis,
                                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Slice":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Slice(node.input, node.output,
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Neg":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Neg(node.input, node.output, 
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Reciprocal":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Reciprocal(node.input, node.output, 
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Ceil":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Ceil(node.input, node.output, 
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Floor":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Floor(node.input, node.output, 
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Cast":
            to = 1 # default float
            for attr in node.attribute:
                if attr.name == "to": to = attr.i
            # Cast 的输出类型由 'to' 属性决定，不一定等于输入类型
            target_dtype = onnx_dtype_mapping.get(to, "float32")
            onnx_graph_list.append(nn.Operators.Cast(node.input, node.output, 
                                   dtype=target_dtype, version="17"))
        elif node.op_type == "Clip":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Clip(node.input, node.output, 
                                   dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "MatMul":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.MatMul(node.input, node.output, 
                                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Gather":
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Gather(node.input, node.output, axis=axis,
                                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Expand":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Expand(node.input, node.output,
                                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Shape":
            start = 0
            end = None
            for attr in node.attribute:
                if attr.name == "start": start = attr.i
                if attr.name == "end": end = attr.i
            # Shape 输出一定是 int64
            onnx_graph_list.append(
                nn.Operators.Shape(node.input, node.output, start=start, end=end,
                                   dtype="int64", version="17"))
        elif node.op_type == "Constant":
            value = None
            dtype = "float32"
            for attr in node.attribute:
                if attr.name == "value":
                    # 解析 TensorProto
                    t = attr.t
                    np_dtype = onnx_dtype_mapping[t.data_type]
                    val_np = numpy_helper.to_array(t)
                    value = val_np
                    dtype = np_dtype                
            onnx_graph_list.append(
                nn.Operators.Constant(node.input, node.output, value=value,
                                      dtype=dtype, version="17"))
        elif node.op_type == "Equal":
            onnx_graph_list.append(nn.Operators.Equal(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Greater":
            onnx_graph_list.append(nn.Operators.Greater(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Less":
            onnx_graph_list.append(nn.Operators.Less(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "GreaterOrEqual":
            onnx_graph_list.append(nn.Operators.GreaterOrEqual(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "LessOrEqual":
            onnx_graph_list.append(nn.Operators.LessOrEqual(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Not":
            onnx_graph_list.append(nn.Operators.Not(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "And":
            onnx_graph_list.append(nn.Operators.And(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Or":
            onnx_graph_list.append(nn.Operators.Or(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Xor":
            onnx_graph_list.append(nn.Operators.Xor(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "IsNaN":
            onnx_graph_list.append(nn.Operators.IsNaN(node.input, node.output, dtype="bool", version="17"))
        elif node.op_type == "Sin":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Sin(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Tan":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Tan(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Atan":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Atan(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Sign":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Sign(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Identity":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Identity(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Mod":
            fmod = 0
            for attr in node.attribute:
                if attr.name == "fmod": fmod = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Mod(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], fmod=fmod, version="17"))
        elif node.op_type == "Where":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Where(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "ConstantOfShape":
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    value = numpy_helper.to_array(attr.t)
            # 输出类型由 value 决定，如果 value 为空默认 float32
            target_dtype = "float32"
            if value is not None:
                if value.dtype == np.float32: target_dtype = "float32"
                elif value.dtype == np.int64: target_dtype = "int64"
                elif value.dtype == np.int32: target_dtype = "int32"
                elif value.dtype == np.bool_: target_dtype = "bool"
            onnx_graph_list.append(
                nn.Operators.ConstantOfShape(node.input, node.output, value=value, dtype=target_dtype, version="17"))
        elif node.op_type == "Range":
            # Range 输出类型由 start 输入决定
            elem_type = get_tensor_dtype(node.input[0], onnx_model) 
            onnx_graph_list.append(
                nn.Operators.Range(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Tile":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Tile(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Pad":
            mode = "constant"
            for attr in node.attribute:
                if attr.name == "mode": mode = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Pad(node.input, node.output, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Split":
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Split(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"]:
            axes = None
            keepdims = 1
            for attr in node.attribute:
                if attr.name == "axes": axes = attr.ints
                elif attr.name == "keepdims": keepdims = attr.i
            # Reduce 操作通常保持输入类型
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            cls_map = {
                "ReduceMean": nn.Operators.ReduceMean,
                "ReduceSum": nn.Operators.ReduceSum,
                "ReduceMax": nn.Operators.ReduceMax,
                "ReduceMin": nn.Operators.ReduceMin,
                "ReduceProd": nn.Operators.ReduceProd
            }
            onnx_graph_list.append(
                cls_map[node.op_type](
                    node.input, node.output, 
                    axes=axes, keepdims=keepdims, 
                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type in ["ArgMax", "ArgMin"]:
            axis = 0
            keepdims = 1
            select_last_index = 0
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
                elif attr.name == "keepdims": keepdims = attr.i
                elif attr.name == "select_last_index": select_last_index = attr.i
            # ArgMax/ArgMin 输出固定为 int64
            cls_map = {"ArgMax": nn.Operators.ArgMax, "ArgMin": nn.Operators.ArgMin}
            onnx_graph_list.append(
                cls_map[node.op_type](
                    node.input, node.output, 
                    axis=axis, keepdims=keepdims, select_last_index=select_last_index,
                    dtype="int64", version="17"))
        elif node.op_type == "ScatterND":
            reduction = "none"
            for attr in node.attribute:
                if attr.name == "reduction": reduction = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.ScatterND(node.input, node.output, reduction=reduction, dtype=onnx_dtype_mapping[elem_type]))
        elif node.op_type == "GatherND":
            batch_dims = 0
            for attr in node.attribute:
                if attr.name == "batch_dims": batch_dims = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GatherND(node.input, node.output, batch_dims=batch_dims, dtype=onnx_dtype_mapping[elem_type]))
        elif node.op_type == "GatherElements":
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GatherElements(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type]))
        elif node.op_type == "NonZero":
             onnx_graph_list.append(nn.Operators.NonZero(node.input, node.output))
        elif node.op_type == "Resize":
            mode = "nearest"
            coord_mode = "asymmetric"
            nearest_mode = "round_prefer_floor"
            for attr in node.attribute:
                if attr.name == "mode": mode = attr.s.decode('utf-8')
                elif attr.name == "coordinate_transformation_mode": coord_mode = attr.s.decode('utf-8')
                elif attr.name == "nearest_mode": nearest_mode = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.Resize(node.input, node.output, 
                                    mode=mode, coord_mode=coord_mode, nearest_mode=nearest_mode,
                                    dtype=onnx_dtype_mapping[elem_type]))
        elif node.op_type == "TopK":
            axis = -1
            largest = 1
            sorted_ = 1 
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
                elif attr.name == "largest": largest = attr.i
                elif attr.name == "sorted": sorted_ = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            # TopK 返回两个输出
            onnx_graph_list.append(nn.Operators.TopK(node.input, node.output, axis=axis, largest=largest, sorted=sorted_, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "CumSum":
            exclusive = 0
            reverse = 0
            for attr in node.attribute:
                if attr.name == "exclusive": exclusive = attr.i
                elif attr.name == "reverse": reverse = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.CumSum(node.input, node.output, exclusive=exclusive, reverse=reverse, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "RandomUniformLike":
            high = 1.0
            low = 0.0
            seed = 0.0
            dtype_attr = 1 # float
            for attr in node.attribute:
                if attr.name == "high": high = attr.f
                elif attr.name == "low": low = attr.f
                elif attr.name == "seed": seed = attr.f
                elif attr.name == "dtype": dtype_attr = attr.i
            # 输出类型优先使用 dtype 属性，如果没有则跟随 input
            target_dtype = onnx_dtype_mapping.get(dtype_attr, "float32")
            onnx_graph_list.append(nn.Operators.RandomUniformLike(node.input, node.output, high=high, low=low, seed=seed, dtype=target_dtype, version="17"))
        elif node.op_type == "Einsum":
            equation = ""
            for attr in node.attribute:
                if attr.name == "equation": equation = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Einsum(node.input, node.output, equation=equation, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Upsample":
            # Upsample (deprecated) -> Resize
            mode = "nearest"
            for attr in node.attribute:
                if attr.name == "mode": mode = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            # Upsample 参数通常通过 scales 输入传递
            onnx_graph_list.append(
                nn.Operators.Resize(node.input, node.output, 
                                    mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Elu":
            alpha = 1.0
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Elu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Selu":
            alpha = 1.67326
            gamma = 1.0507
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
                elif attr.name == "gamma": gamma = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Selu(node.input, node.output, alpha=alpha, gamma=gamma, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "LeakyRelu":
            alpha = 0.01
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.LeakyRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "ThresholdedRelu":
            alpha = 1.0
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.ThresholdedRelu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "HardSigmoid":
            alpha = 0.2
            beta = 0.5
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
                elif attr.name == "beta": beta = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.HardSigmoid(node.input, node.output, alpha=alpha, beta=beta, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Celu":
            alpha = 1.0
            for attr in node.attribute:
                if attr.name == "alpha": alpha = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Celu(node.input, node.output, alpha=alpha, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Shrink":
            bias = 0.0
            lambd = 0.5
            for attr in node.attribute:
                if attr.name == "bias": bias = attr.f
                elif attr.name == "lambd": lambd = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Shrink(node.input, node.output, bias=bias, lambd=lambd, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Softplus":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Softplus(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Softsign":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Softsign(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "HardSwish":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.HardSwish(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Acos":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Acos(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Asin":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Asin(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17")) 
        elif node.op_type == "Cosh":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Cosh(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Sinh":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Sinh(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Asinh":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Asinh(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Acosh":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Acosh(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Atanh":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Atanh(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BitwiseAnd":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BitwiseAnd(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BitwiseOr":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BitwiseOr(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BitwiseXor":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BitwiseXor(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BitwiseNot":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BitwiseNot(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BitShift":
            direction = "LEFT"
            for attr in node.attribute:
                if attr.name == "direction": direction = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BitShift(node.input, node.output, direction=direction, dtype=onnx_dtype_mapping[elem_type], version="17"))
        # 后面其他的归约也可以这么处理
        elif node.op_type in ["ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"]:
            axes = None
            keepdims = 1
            for attr in node.attribute:
                if attr.name == "axes": axes = attr.ints
                elif attr.name == "keepdims": keepdims = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            cls_map = {
                "ReduceL1": nn.Operators.ReduceL1,
                "ReduceL2": nn.Operators.ReduceL2,
                "ReduceLogSum": nn.Operators.ReduceLogSum,
                "ReduceLogSumExp": nn.Operators.ReduceLogSumExp,
                "ReduceSumSquare": nn.Operators.ReduceSumSquare
            }
            onnx_graph_list.append(
                cls_map[node.op_type](
                    node.input, node.output, 
                    axes=axes, keepdims=keepdims, 
                    dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "AveragePool":
            kernel_shape = [1, 1]
            pads = [0, 0, 0, 0]
            strides = [1, 1]
            dilations = [1, 1]
            count_include_pad = 0
            for attr in node.attribute:
                if attr.name == "kernel_shape": kernel_shape = attr.ints
                elif attr.name == "pads": pads = attr.ints
                elif attr.name == "strides": strides = attr.ints
                elif attr.name == "dilations": dilations = attr.ints
                elif attr.name == "count_include_pad": count_include_pad = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.AveragePool(
                    node.input, node.output, 
                    kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations,
                    count_include_pad=count_include_pad, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "LpPool":
            kernel_shape = [1, 1]
            pads = [0, 0, 0, 0]
            strides = [1, 1]
            dilations = [1, 1]
            p = 2
            for attr in node.attribute:
                if attr.name == "kernel_shape": kernel_shape = attr.ints
                elif attr.name == "pads": pads = attr.ints
                elif attr.name == "strides": strides = attr.ints
                elif attr.name == "dilations": dilations = attr.ints
                elif attr.name == "p": p = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(
                nn.Operators.LpPool(
                    node.input, node.output, 
                    kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations,
                    p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "GlobalAveragePool":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GlobalAveragePool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "GlobalMaxPool":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GlobalMaxPool(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "GlobalLpPool":
            p = 2
            for attr in node.attribute:
                if attr.name == "p": p = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GlobalLpPool(node.input, node.output, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Mean":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Mean(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Size":
            onnx_graph_list.append(nn.Operators.Size(node.input, node.output, dtype="int64", version="17"))
        elif node.op_type == "IsInf":
            detect_neg = 1
            detect_pos = 1
            for attr in node.attribute:
                if attr.name == "detect_negative": detect_neg = attr.i
                elif attr.name == "detect_positive": detect_pos = attr.i
            onnx_graph_list.append(nn.Operators.IsInf(node.input, node.output, detect_negative=detect_neg, detect_positive=detect_pos, version="17"))
        elif node.op_type == "OneHot":
            axis = -1
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.input[2], onnx_model)
            onnx_graph_list.append(nn.Operators.OneHot(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Tril":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Tril(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Triu":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Triu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Round":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Round(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))  
        elif node.op_type == "Erf":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Erf(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "BatchNormalization":
            epsilon = 1e-5
            momentum = 0.9
            for attr in node.attribute:
                if attr.name == "epsilon": epsilon = attr.f
                elif attr.name == "momentum": momentum = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.BatchNormalization(node.input, node.output, epsilon=epsilon, momentum=momentum, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "InstanceNormalization":
            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon": epsilon = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.InstanceNormalization(node.input, node.output, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "LayerNormalization":
            epsilon = 1e-5
            axis = -1
            for attr in node.attribute:
                if attr.name == "epsilon": epsilon = attr.f
                elif attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.LayerNormalization(node.input, node.output, axis=axis, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "HannWindow":
            periodic = 1
            output_datatype = 1
            for attr in node.attribute:
                if attr.name == "periodic": periodic = attr.i
                elif attr.name == "output_datatype": output_datatype = attr.i
            onnx_graph_list.append(nn.Operators.HannWindow(node.input, node.output, periodic=periodic, output_datatype=output_datatype, version="17"))
        elif node.op_type == "HammingWindow":
            periodic = 1
            output_datatype = 1
            for attr in node.attribute:
                if attr.name == "periodic": periodic = attr.i
                elif attr.name == "output_datatype": output_datatype = attr.i
            onnx_graph_list.append(nn.Operators.HammingWindow(node.input, node.output, periodic=periodic, output_datatype=output_datatype, version="17"))
        elif node.op_type == "BlackmanWindow":
            periodic = 1
            output_datatype = 1
            for attr in node.attribute:
                if attr.name == "periodic": periodic = attr.i
                elif attr.name == "output_datatype": output_datatype = attr.i
            onnx_graph_list.append(nn.Operators.BlackmanWindow(node.input, node.output, periodic=periodic, output_datatype=output_datatype, version="17"))
        elif node.op_type == "RandomNormal":
            mean = 0.0
            scale = 1.0
            seed = 0.0
            dtype = 1
            shape = []
            for attr in node.attribute:
                if attr.name == "mean": mean = attr.f
                elif attr.name == "scale": scale = attr.f
                elif attr.name == "seed": seed = attr.f
                elif attr.name == "dtype": dtype = attr.i
                elif attr.name == "shape": shape = attr.ints
            onnx_graph_list.append(nn.Operators.RandomNormal(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, shape=shape, version="17"))
        elif node.op_type == "RandomNormalLike":
            mean = 0.0
            scale = 1.0
            seed = 0.0
            dtype = None
            for attr in node.attribute:
                if attr.name == "mean": mean = attr.f
                elif attr.name == "scale": scale = attr.f
                elif attr.name == "seed": seed = attr.f
                elif attr.name == "dtype": dtype = attr.i
            onnx_graph_list.append(nn.Operators.RandomNormalLike(node.input, node.output, mean=mean, scale=scale, seed=seed, dtype=dtype, version="17"))
        elif node.op_type == "Bernoulli":
            seed = 0.0
            dtype = None
            for attr in node.attribute:
                if attr.name == "seed": seed = attr.f
                elif attr.name == "dtype": dtype = attr.i
            onnx_graph_list.append(nn.Operators.Bernoulli(node.input, node.output, seed=seed, dtype=dtype, version="17"))
        elif node.op_type == "Dropout":
            seed = 0.0
            ratio = 0.5
            training_mode = 0
            for attr in node.attribute:
                if attr.name == "seed": seed = attr.f
                elif attr.name == "ratio": ratio = attr.f
                elif attr.name == "training_mode": training_mode = attr.i
            onnx_graph_list.append(nn.Operators.Dropout(node.input, node.output, seed=seed, ratio=ratio, training_mode=training_mode, version="17"))
        elif node.op_type == "Gelu":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Gelu(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Mish":
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Mish(node.input, node.output, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Hardmax":
            axis = -1
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Hardmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "LogSoftmax":
            axis = -1
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.LogSoftmax(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "LpNormalization":
            axis = -1
            p = 2
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
                elif attr.name == "p": p = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.LpNormalization(node.input, node.output, axis=axis, p=p, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "DepthToSpace":
            blocksize = 1
            mode = "DCR"
            for attr in node.attribute:
                if attr.name == "blocksize": blocksize = attr.i
                elif attr.name == "mode": mode = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.DepthToSpace(node.input, node.output, blocksize=blocksize, mode=mode, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "SpaceToDepth":
            blocksize = 1
            for attr in node.attribute:
                if attr.name == "blocksize": blocksize = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.SpaceToDepth(node.input, node.output, blocksize=blocksize, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "ReverseSequence":
            time_axis = 0
            batch_axis = 1
            for attr in node.attribute:
                if attr.name == "time_axis": time_axis = attr.i
                elif attr.name == "batch_axis": batch_axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.ReverseSequence(node.input, node.output, time_axis=time_axis, batch_axis=batch_axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Compress":
            axis = None
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Compress(node.input, node.output, axis=axis, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "ScatterElements":
            axis = 0
            reduction = "none"
            for attr in node.attribute:
                if attr.name == "axis": axis = attr.i
                elif attr.name == "reduction": reduction = attr.s.decode('utf-8')
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.ScatterElements(node.input, node.output, axis=axis, reduction=reduction, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "GroupNormalization":
            num_groups = 1
            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == "num_groups": num_groups = attr.i
                elif attr.name == "epsilon": epsilon = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.GroupNormalization(node.input, node.output, num_groups=num_groups, epsilon=epsilon, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "Binarizer":
            threshold = 0.0
            for attr in node.attribute:
                if attr.name == "threshold": threshold = attr.f
            elem_type = get_tensor_dtype(node.output[0], onnx_model)
            onnx_graph_list.append(nn.Operators.Binarizer(node.input, node.output, threshold=threshold, dtype=onnx_dtype_mapping[elem_type], version="17"))
        elif node.op_type == "DynamicQuantizeLinear":
            # 输出通常是 UINT8
            onnx_graph_list.append(nn.Operators.DynamicQuantizeLinear(node.input, node.output, version="17"))
        else:
            # 忽略未支持的操作类型
            pass
            
    return onnx_graph_list