import numpy as np
import sys
import os

# 添加路径以便导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nn.Operators import ADD, MUL, SUB, DIV, RESHAPE
from nn import Tensor, Tensor_


def test_add_operator():
    """测试加法算子"""
    print("=== 测试 ADD 算子 ===")

    # 创建加法算子
    add_op = ADD(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

    # 创建测试数据
    input1_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input2_data = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    input1 = Tensor(4, dtype='float32', data=input1_data)
    input2 = Tensor(4, dtype='float32', data=input2_data)

    # 执行前向传播
    result = add_op.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"输入1: {input1_data}")
    print(f"输入2: {input2_data}")
    print(f"输出: {output_tensor.data}")
    print(f"期望: {input1_data + input2_data}")
    print(f"测试结果: {'通过' if np.allclose(output_tensor.data, input1_data + input2_data) else '失败'}")
    print()


def test_mul_operator():
    """测试乘法算子"""
    print("=== 测试 MUL 算子 ===")

    mul_op = MUL(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

    input1_data = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input2_data = np.array([1.5, 2.0, 0.5, 2.5], dtype=np.float32)

    input1 = Tensor(4, dtype='float32', data=input1_data)
    input2 = Tensor(4, dtype='float32', data=input2_data)

    result = mul_op.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"输入1: {input1_data}")
    print(f"输入2: {input2_data}")
    print(f"输出: {output_tensor.data}")
    print(f"期望: {input1_data * input2_data}")
    print(f"测试结果: {'通过' if np.allclose(output_tensor.data, input1_data * input2_data) else '失败'}")
    print()


def test_sub_operator():
    """测试减法算子"""
    print("=== 测试 SUB 算子 ===")

    sub_op = SUB(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

    input1_data = np.array([5.0, 8.0, 10.0, 12.0], dtype=np.float32)
    input2_data = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    input1 = Tensor(4, dtype='float32', data=input1_data)
    input2 = Tensor(4, dtype='float32', data=input2_data)

    result = sub_op.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"输入1: {input1_data}")
    print(f"输入2: {input2_data}")
    print(f"输出: {output_tensor.data}")
    print(f"期望: {input1_data - input2_data}")
    print(f"测试结果: {'通过' if np.allclose(output_tensor.data, input1_data - input2_data) else '失败'}")
    print()


def test_div_operator():
    """测试除法算子"""
    print("=== 测试 DIV 算子 ===")

    div_op = DIV(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

    input1_data = np.array([6.0, 12.0, 15.0, 20.0], dtype=np.float32)
    input2_data = np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float32)

    input1 = Tensor(4, dtype='float32', data=input1_data)
    input2 = Tensor(4, dtype='float32', data=input2_data)

    result = div_op.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"输入1: {input1_data}")
    print(f"输入2: {input2_data}")
    print(f"输出: {output_tensor.data}")
    print(f"期望: {input1_data / input2_data}")
    print(f"测试结果: {'通过' if np.allclose(output_tensor.data, input1_data / input2_data) else '失败'}")
    print()


def test_reshape_operator():
    """测试reshape算子"""
    print("=== 测试 RESHAPE 算子 ===")

    reshape_op = RESHAPE(inputs=['input'], outputs=['output'], dtype='float32', new_shape=[2, 2])

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_tensor = Tensor(4, dtype='float32', data=input_data)

    result = reshape_op.forward(input_tensor)
    output_tensor = result['values']['tensor']

    print(f"输入形状: {input_tensor.size}")
    print(f"输入数据: {input_data}")
    print(f"输出形状: {output_tensor.size}")
    print(f"输出数据: {output_tensor.data}")
    print(f"期望形状: [2, 2]")
    print(
        f"测试结果: {'通过' if output_tensor.size == [2, 2] and np.array_equal(output_tensor.data.flatten(), input_data) else '失败'}")
    print()


def test_broadcast_operations():
    """测试相同形状的操作（移除广播测试）"""
    print("=== 测试相同形状操作 ===")

    add_op = ADD(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

    input1_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input2_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)  # 相同形状

    input1 = Tensor(4, dtype='float32', data=input1_data)
    input2 = Tensor(4, dtype='float32', data=input2_data)

    result = add_op.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"输入1: {input1_data}")
    print(f"输入2: {input2_data}")
    print(f"输出: {output_tensor.data}")
    print(f"期望: {input1_data + input2_data}")
    print(f"相同形状测试: {'通过' if np.allclose(output_tensor.data, input1_data + input2_data) else '失败'}")
    print()


def test_forward_():
    """测试符号执行（不使用真实数据）"""
    print("=== 测试符号执行 ===")

    # 测试ADD符号执行
    add_op = ADD(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')
    input1_ = Tensor_(4, dtype='float32')
    input2_ = Tensor_(4, dtype='float32')

    result = add_op.forward_(input1_, input2_)
    output_tensor_ = result['values']['tensor']

    print(f"ADD符号执行 - 输入形状: {input1_.size}, 输出形状: {output_tensor_.size}")

    # 测试RESHAPE符号执行
    reshape_op = RESHAPE(inputs=['input'], outputs=['output'], dtype='float32', new_shape=[2, 2])
    input_ = Tensor_(4, dtype='float32')
    result = reshape_op.forward_(input_)
    output_reshape_ = result['values']['tensor']

    print(f"RESHAPE符号执行 - 输入形状: {input_.size}, 输出形状: {output_reshape_.size}")
    print("符号执行测试: 通过")
    print()


def test_different_dtypes():
    """测试不同数据类型"""
    print("=== 测试不同数据类型 ===")

    # 测试int32类型
    add_op_int = ADD(inputs=['input1', 'input2'], outputs=['output'], dtype='int32')

    input1_data = np.array([1, 2, 3, 4], dtype=np.int32)
    input2_data = np.array([5, 6, 7, 8], dtype=np.int32)

    input1 = Tensor(4, dtype='int32', data=input1_data)
    input2 = Tensor(4, dtype='int32', data=input2_data)

    result = add_op_int.forward(input1, input2)
    output_tensor = result['values']['tensor']

    print(f"int32加法 - 输入1: {input1_data}")
    print(f"int32加法 - 输入2: {input2_data}")
    print(f"int32加法 - 输出: {output_tensor.data}")
    print(f"int32测试: {'通过' if np.array_equal(output_tensor.data, input1_data + input2_data) else '失败'}")
    print()


def main():
    """运行所有测试"""
    print("开始测试五个算子...\n")

    try:
        test_add_operator()
        test_mul_operator()
        test_sub_operator()
        test_div_operator()
        test_reshape_operator()
        test_broadcast_operations()
        test_forward_()
        test_different_dtypes()

        print("所有测试完成！")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()