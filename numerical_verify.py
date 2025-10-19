import numpy as np
from nn import Tensor
from nn.Operators import MUL  # 假设存在ADD算子类

def run_numpy_verification(op_name="mul", nps_operator_class=MUL, shape=(1, 3, 128, 128), dtype="float32"):
    """
    针对ADD算子的NumPy基准验证函数，对比NPS加法与NumPy原生加法结果
    """
    print(f"\n--- NumPy Benchmark Verification for [{op_name.upper()}] Operator ---")

    # 1. 生成两组随机输入数据（加法需要两个输入，确保形状相同）
    np_input1 = np.random.randn(*shape).astype(dtype)
    np_input2 = np.random.randn(*shape).astype(dtype)
    # 转换为NPS Tensor
    tensor_input1 = Tensor(*shape, dtype=dtype, data=np_input1)
    tensor_input2 = Tensor(*shape, dtype=dtype, data=np_input2)
    print(f"Input shape: {shape}, dtype: {dtype} (two inputs for addition)")

    # 2. 获取NPS ADD算子计算结果
    print(f"[NPS] Running {op_name.upper()} operator...")
    # ADD算子需要两个输入，因此inputs参数为['x', 'y']
    nps_op = nps_operator_class(inputs=['x', 'y'], outputs=['z'], dtype=dtype)
    # 传入两个输入张量进行前向计算
    nps_result = nps_op.forward(tensor_input1, tensor_input2)["tensor"].data
    print("[NPS] Addition calculation complete.")

    # 3. 获取NumPy基准计算结果（原生加法）
    print(f"[NumPy] Running reference {op_name.upper()} calculation...")
    np_result = np_input1 * np_input2  # NumPy原生加法作为基准
    print("[NumPy] Addition calculation complete.")

    # 4. 对比结果（设置合理精度阈值）
    print("[COMPARE] Comparing NPS ADD result vs NumPy addition result...")
    if np.allclose(nps_result, np_result, atol=1e-6, rtol=1e-6):
        print(f"✅ SUCCESS: {op_name.upper()} operator matches NumPy benchmark!")
    else:
        print(f"❌ FAILED: {op_name.upper()} operator differs from NumPy benchmark!")
        # 计算差异统计量
        diff = np.abs(nps_result - np_result)
        print(f"  - Max absolute difference: {diff.max()}")
        print(f"  - Mean absolute difference: {diff.mean()}")
        # 可选项：打印第一个差异位置的具体值
        first_diff_idx = np.unravel_index(np.argmax(diff), shape)
        print(f"  - First major difference at index {first_diff_idx}:")
        print(f"    NPS: {nps_result[first_diff_idx]}, NumPy: {np_result[first_diff_idx]}")

if __name__ == "__main__":
    # 验证ADD算子（对比np.add，即+运算符）
    # 可调整shape测试不同维度的加法（如标量+张量、高维张量相加等）
    run_numpy_verification(op_name="relu", nps_operator_class=MUL)