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
        else:
            # 忽略未支持的操作类型
            pass
            
    return onnx_graph_list