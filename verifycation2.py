import numpy as np
import subprocess
import os
from nn import Tensor
from nn.Operators import DIV  # 导入DIV算子类（替换MUL）

def run_div_verification(op_name, nps_operator_class, shape=(1, 3, 128, 128), dtype="float32"):  # 函数名改为div相关
    """
    用于验证DIV算子的函数，适配双输入（x / y），对比NPS和CUDA的计算结果
    """
    print(f"\n--- Final Verification for [{op_name.upper()}] Operator ---")

    # 1. 准备通用资源（可执行文件名称改为div）
    executable_path = f"./verify_{op_name.lower()}.exe"  # 最终对应verify_div.exe
    input_file1 = "temp_input1.bin"  # 第一个输入（被除数x）
    input_file2 = "temp_input2.bin"  # 第二个输入（除数y）
    output_file = "temp_output.bin"

    if not os.path.exists(executable_path):
        print(f"❌ FAILED: CUDA executable '{executable_path}' not found. Please compile it first.")
        return

    # 2. 生成两个随机输入数据（div需要双输入：x / y，确保除数不为0）
    np_input1 = np.random.randn(*shape).astype(np.float32)  # 被除数x
    np_input2 = np.random.randn(*shape).astype(np.float32)  # 除数y
    # 处理除数，避免零值（防止除零错误）
    np_input2 = np_input2 + (1e-5 if np.any(np_input2 == 0) else 0)
    tensor_input1 = Tensor(*shape, dtype=dtype, data=np_input1)
    tensor_input2 = Tensor(*shape, dtype=dtype, data=np_input2)
    num_elements = np_input1.size  # 两个输入元素数量相同

    # 3. 获取NPS计算结果（使用DIV算子）
    print(f"[NPS] Running {op_name.upper()} operator...")
    nps_op = nps_operator_class(inputs=['x', 'y'], outputs=['z'], dtype=dtype)  # 双输入配置（x / y）
    nps_result = nps_op.forward(tensor_input1, tensor_input2)["tensor"].data  # 传入被除数和除数
    print("[NPS] Calculation complete.")

    # 4. 获取CUDA"黄金标准"计算结果（调用div的CUDA程序）
    print(f"[CUDA] Preparing data and running '{executable_path}'...")
    # 写入两个输入文件（x和y）
    np_input1.tofile(input_file1)
    np_input2.tofile(input_file2)

    # 调用CUDA程序（需支持双输入，计算x / y）
    subprocess.run([
        executable_path,
        str(num_elements),
        input_file1,  # 被除数x的路径
        input_file2,  # 除数y的路径
        output_file
    ], check=True)

    # 读取CUDA计算结果
    cuda_result = np.fromfile(output_file, dtype=np.float32).reshape(shape)
    print("[CUDA] Calculation complete.")

    # 5. 对比结果（验证x / y的一致性）
    print("[COMPARE] Comparing NPS result vs CUDA result...")
    # 除法对精度更敏感，适当放宽容差
    if np.allclose(nps_result, cuda_result, atol=1e-5, rtol=1e-5):
        print(f"✅ SUCCESS: {op_name.upper()} operator passed final verification against CUDA!")
    else:
        print(f"❌ FAILED: {op_name.upper()} operator failed final verification!")
        diff = np.abs(nps_result - cuda_result).max()
        print(f"  - Max difference: {diff}")

    # 6. 清理临时文件
    for f in [input_file1, input_file2, output_file]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    # 验证DIV算子（确保verify_div.cu已编译为verify_div.exe）
    run_div_verification(op_name="div", nps_operator_class=DIV)  # 传入div名称和DIV类