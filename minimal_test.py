import numpy as np
import sys
import os

# 添加路径以便导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nn.Operators import ADD
from nn import Tensor


def minimal_test():
    """最小化测试"""
    print("=== 最小化测试 ===")

    try:
        # 只测试ADD算子
        add_op = ADD(inputs=['input1', 'input2'], outputs=['output'], dtype='float32')

        # 简单数据
        input1_data = np.array([1.0, 2.0], dtype=np.float32)
        input2_data = np.array([0.5, 1.5], dtype=np.float32)

        input1 = Tensor(2, dtype='float32', data=input1_data)
        input2 = Tensor(2, dtype='float32', data=input2_data)

        print("开始forward...")
        result = add_op.forward(input1, input2)
        output_tensor = result['values']['tensor']

        print(f"输入1: {input1_data}")
        print(f"输入2: {input2_data}")
        print(f"输出: {output_tensor.data}")
        print(f"期望: {input1_data + input2_data}")

        success = np.allclose(output_tensor.data, input1_data + input2_data)
        print(f"结果: {'✅ 通过' if success else '❌ 失败'}")

        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    minimal_test()