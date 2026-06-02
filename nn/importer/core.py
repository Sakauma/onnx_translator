# /**
#   ******************************************************************************
#   * @file        core.py
#   * @author      Egor Izmaylov
#   * @brief       实现 ONNX 模型导入主流程，负责加载模型、构建 dtype 映射、解析 initializer 和节点。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import onnx
import numpy as np
from onnx import numpy_helper
import nn.Operators
from nn import onnx_dtype_mapping, Tensor_
from .context import GenericNode, ImportContext
from .registry import OP_FACTORY_REGISTRY
from . import node_factories_01  # noqa: F401
from . import node_factories_02  # noqa: F401
from . import node_factories_03  # noqa: F401
from . import node_factories_04  # noqa: F401
from onnx import shape_inference
import traceback

# 实现 `ONNXImport` 步骤，规范化输入并返回下游期望的数据或元信息。
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
    import_context = ImportContext(dtype_map={}, strict=strict, generic_nodes=generic_nodes)
    
    print(f"   [ONNXImport] Loading model from {file_path}...")
    try:
        onnx_model = onnx.load(file_path, load_external_data=False)
    except Exception as e:
        print(f" Critical Error: Failed to load ONNX file. {e}")
        raise e

    # dtype 推断只在导入开始时执行一次；很多算子构造函数需要输出 dtype 来选择 C 张量类型或 Python fallback。
    # 如果每个节点都重新执行形状推断，不仅性能较差，对部分推断成功的图也更难保持行为稳定。
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
    # 实现 `get_dtype` 步骤，规范化输入并返回下游期望的数据或元信息。
    import_context.dtype_map = dtype_map

    def get_dtype(name, default=onnx.TensorProto.FLOAT):
        return import_context.get_dtype(name, default)

    # initializer 会先转换成 Constant 算子，这样后续运行时可以用同一套边数据逻辑处理权重和普通输入。
    # 这种做法能保持图遍历简单，也避免额外维护一套参数命名空间。
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

    # 下面每个分支都应遵循 ONNX opset 17 的属性默认值；遇到可选张量输入时保留空字符串占位。
    # 运行时会依赖输入位置还原 ONNX 语义，因此不能随意压缩 optional 输入列表。
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
            factory = OP_FACTORY_REGISTRY.get(node.op_type) or OP_FACTORY_REGISTRY.get(node.op_type.upper())
            if factory is None:
                raise NotImplementedError(f"Operator {node.op_type} is not implemented.")

            onnx_graph_list.append(factory(node, import_context))
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
