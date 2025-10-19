import numpy as np
import onnx
import onnx.helper as helper
import onnx.shape_inference as shape_inference
import os
import tempfile
import sys

# 添加路径以便导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def create_test_onnx_model():
    """创建一个包含五个算子的测试ONNX模型"""

    # 创建输入
    input1 = helper.make_tensor_value_info('input1', onnx.TensorProto.FLOAT, [4])
    input2 = helper.make_tensor_value_info('input2', onnx.TensorProto.FLOAT, [4])

    # 创建节点
    add_node = helper.make_node('Add', ['input1', 'input2'], ['add_output'], 'add_node')
    mul_node = helper.make_node('Mul', ['add_output', 'input2'], ['mul_output'], 'mul_node')
    sub_node = helper.make_node('Sub', ['mul_output', 'input1'], ['sub_output'], 'sub_node')
    div_node = helper.make_node('Div', ['sub_output', 'input2'], ['div_output'], 'div_node')

    # 添加reshape需要的形状常量
    shape_tensor = helper.make_tensor('shape', onnx.TensorProto.INT64, [2], [2, 2])
    shape_node = helper.make_node('Constant', [], ['shape'], value=shape_tensor, name='shape_const')

    reshape_node = helper.make_node('Reshape', ['div_output', 'shape'], ['reshape_output'], 'reshape_node')

    # 创建输出
    output = helper.make_tensor_value_info('reshape_output', onnx.TensorProto.FLOAT, [2, 2])

    # 创建图
    graph = helper.make_graph(
        [add_node, mul_node, sub_node, div_node, shape_node, reshape_node],
        'test_graph',
        [input1, input2],
        [output]
    )

    # 创建模型
    model = helper.make_model(graph, producer_name='test_producer')
    model = shape_inference.infer_shapes(model)

    return model


def test_onnx_import():
    """测试ONNX模型导入功能"""
    print("=== 测试ONNX模型导入 ===")

    try:
        # 创建测试模型
        model = create_test_onnx_model()

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model, f.name)
            temp_file = f.name

        # 从 nn 包中导入 ONNXImport
        from nn.ONNXImport import ONNXImport

        operators = ONNXImport(temp_file)

        print(f"成功导入 {len(operators)} 个算子:")
        for op in operators:
            print(f"  - {op.__class__.__name__}: 输入 {op.inputs} -> 输出 {op.outputs}")

        # 清理临时文件
        os.unlink(temp_file)

        print("ONNX导入测试: 通过")

    except Exception as e:
        print(f"ONNX导入测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_onnx_import()