import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping, Operators
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

    # 在初始化器中查找
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return initializer.data_type

    return 1  # 默认float32


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
        # 获取输出张量的数据类型
        elem_type = get_tensor_dtype(node.output[0], onnx_model)
        dtype = onnx_dtype_mapping.get(elem_type, "float32")

        if node.op_type.upper() == "ADD":
            # 处理Add操作节点
            onnx_graph_list.append(
                Operators.ADD(node.input, node.output, dtype=dtype, version="17")
            )

        elif node.op_type.upper() == "MUL":
            # 处理Mul操作节点
            onnx_graph_list.append(
                Operators.MUL(node.input, node.output, dtype=dtype, version="17")
            )

        elif node.op_type.upper() == "SUB":
            # 处理Sub操作节点
            onnx_graph_list.append(
                Operators.SUB(node.input, node.output, dtype=dtype, version="17")
            )

        elif node.op_type.upper() == "DIV":
            # 处理Div操作节点
            onnx_graph_list.append(
                Operators.DIV(node.input, node.output, dtype=dtype, version="17")
            )

        elif node.op_type.upper() == "RESHAPE":
            # 处理Reshape操作节点
            # 提取reshape的目标形状
            new_shape = None
            if len(node.input) > 1:
                # 形状信息在第二个输入中
                shape_name = node.input[1]
                if shape_name in initializer_shapes:
                    new_shape = initializer_shapes[shape_name]
                else:
                    # 如果形状不在初始化器中，可能需要从其他来源获取
                    # 这里可以添加额外的逻辑来处理动态形状
                    print(f"Warning: Reshape shape {shape_name} not found in initializers")

            onnx_graph_list.append(
                Operators.RESHAPE(
                    node.input,
                    node.output,
                    dtype=dtype,
                    new_shape=new_shape,
                    version="17"
                )
            )

        elif node.op_type.upper() == "RELU":
            # 保留原有的ReLU支持
            onnx_graph_list.append(
                Operators.RELU(node.input, node.output, dtype=dtype, version="17")
            )

        elif node.op_type.upper() == "COS":
            # 保留原有的COS支持
            onnx_graph_list.append(
                Operators.COS(node.input, node.output, dtype=dtype, version="17")
            )

        else:
            # 忽略未支持的操作类型，但打印警告
            print(f"Warning: Unsupported operator type {node.op_type} will be skipped")

    return onnx_graph_list