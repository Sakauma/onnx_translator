import numpy as np
import subprocess
import os
from nn import Tensor
from nn.Operators import RELU, COS

def run_final_verification(op_name, nps_operator_class, shape=(1, 3, 128, 128), dtype="float32"):
    """
    一个通用的最终验证函数，用于对比 NPS 和 CUDA 的计算结果。
    """
    print(f"\n--- Final Verification for [{op_name.upper()}] Operator ---")

    # 1. 准备通用资源
    executable_path = f"./verify_{op_name.lower()}"
    input_file = "temp_input.bin"
    output_file = "temp_output.bin"

    if not os.path.exists(executable_path):
        print(f"❌ FAILED: CUDA executable '{executable_path}' not found. Please compile it first.")
        return

    # 2. 生成随机输入数据
    np_input = np.random.randn(*shape).astype(np.float32)
    tensor_input = Tensor(*shape, dtype=dtype, data=np_input)
    num_elements = np_input.size

    # 3. 获取 NPS 计算结果
    print(f"[NPS] Running {op_name.upper()} operator...")
    nps_op = nps_operator_class(inputs=['x'], outputs=['y'], dtype=dtype)
    nps_result = nps_op.forward(tensor_input)["tensor"].data
    print("[NPS] Calculation complete.")

    # 4. 获取 CUDA "黄金标准" 计算结果
    print(f"[CUDA] Preparing data and running '{executable_path}'...")
    # 将输入数据写入二进制文件
    np_input.tofile(input_file)

    # 通过子进程调用编译好的 CUDA 程序
    subprocess.run([
        executable_path,
        str(num_elements),
        input_file,
        output_file
    ], check=True)

    # 从输出文件中读取 CUDA 的计算结果
    cuda_result = np.fromfile(output_file, dtype=np.float32).reshape(shape)
    print("[CUDA] Calculation complete.")

    # 5. 对比结果
    print("[COMPARE] Comparing NPS result vs CUDA result...")
    if np.allclose(nps_result, cuda_result, atol=1e-6):
        print(f"✅ SUCCESS: {op_name.upper()} operator passed final verification against CUDA!")
    else:
        print(f"❌ FAILED: {op_name.upper()} operator failed final verification!")
        diff = np.abs(nps_result - cuda_result).max()
        print(f"  - Max difference: {diff}")

    # 6. 清理临时文件
    os.remove(input_file)
    os.remove(output_file)

if __name__ == "__main__":
    # 验证 RELU
    run_final_verification(op_name="relu", nps_operator_class=RELU)

    # 验证 COS (确保 verify.cu 已修改并编译为 verify_cos)
    run_final_verification(op_name="cos", nps_operator_class=COS)