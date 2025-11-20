import numpy as np
import subprocess
import os
import nn
from nn import Tensor
from nn.Operators import RELU, COS, ABS, ADD, TANH, RESHAPE, UNSQUEEZE  # 导入所有已实现的算子

# --- 辅助函数：创建测试数据 ---


def create_test_data(shape, dtype):
    """根据形状和类型生成随机Numpy数据"""
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    if "float" in dtype:
        return np.random.randn(*shape).astype(np_dtype)
    else:
        return np.random.randint(-20, 20, shape).astype(np_dtype)

# =============================================================================
# 验证块二：对标 NumPy (功能/广播/类型提升 验证)
# =============================================================================


def run_numpy_verification(op_name, nps_op_class,
                           input_specs, fixed_inputs=None):
    """
    一个通用的验证函数，用于对比 NPS 和 NumPy (作为黄金标准) 的计算结果。
    它支持一元和二元操作，并能验证广播和类型提升。

    Args:
        op_name (str): 操作的Numpy名称 (例如 'add', 'abs')
        nps_op_class (Ops): 要测试的NPS算子类 (例如 nn.Operators.ADD)
        input_specs (list): 一个包含 (dtype, shape) 元组的列表
    """

    test_desc = " + ".join([f"{spec[0]}{spec[1]}" for spec in input_specs])
    print(f"\n--- [NumPy验证] 测试: {op_name.upper()} ({test_desc}) ---")

    # 1. 生成Numpy和NPS的输入数据
    np_inputs = []
    nps_inputs = []

    for i, (dtype, shape) in enumerate(input_specs):
        # 如果有固定输入值，使用固定值；否则生成随机数据
        if fixed_inputs and i < len(
                fixed_inputs) and fixed_inputs[i] is not None:
            np_data = fixed_inputs[i]
        else:
            np_data = create_test_data(shape, dtype)

        np_inputs.append(np_data)
        nps_inputs.append(Tensor(*shape, dtype=dtype, data=np_data))

    # 2. 获取 NPS 计算结果
    print(f"[NPS] ... 正在运行 NPS 算子...")
    nps_op = nps_op_class(inputs=['a', 'b'][:len(
        nps_inputs)], outputs=['y'], dtype="float32")
    nps_result_tensor = nps_op.forward(*nps_inputs)["tensor"]
    nps_result_data = nps_result_tensor.data

    # 3. 获取 NumPy "黄金标准" 计算结果
    print(f"[NumPy] ... 正在运行 NumPy 算子...")
    numpy_result = None
    try:
        if op_name == 'add':
            numpy_result = np_inputs[0] + np_inputs[1]
        elif op_name == 'abs':
            numpy_result = np.abs(np_inputs[0])
        elif op_name == 'cos':
            numpy_result = np.cos(np_inputs[0].astype(np.float64))
        elif op_name == 'relu':
            numpy_result = np.maximum(0, np_inputs[0])
        elif op_name == 'tanh':
            numpy_result = np.tanh(np_inputs[0])
        elif op_name == 'reshape':
            # Reshape 需要特殊处理，因为需要新的形状参数
            if len(np_inputs) > 1:
                new_shape = tuple(np_inputs[1].astype(np.int32))
                numpy_result = np_inputs[0].reshape(new_shape)
            else:
                print("警告: Reshape 需要形状参数")
                return
        elif op_name == 'unsqueeze':
            # Unsqueeze 需要轴参数
            if len(np_inputs) > 1:
                axis = int(np_inputs[1][0])
                numpy_result = np.expand_dims(np_inputs[0], axis=axis)
            else:
                print("警告: Unsqueeze 需要轴参数")
                return
        else:
            print(f"警告: {op_name} 的NumPy验证规则未定义。")
            return
    except Exception as e:
        print(f"✗ FAILED: NumPy 计算失败: {e}")
        return

    # 4. 对比结果
    print("[COMPARE] ... 正在对比 NPS 与 NumPy 结果...")

    # 检查形状
    if nps_result_data.shape != numpy_result.shape:
        print(f"✗ FAILED: 形状不匹配!")
        print(f"  NPS Shape: {nps_result_data.shape}")
        print(f"  NumPy Shape: {numpy_result.shape}")
        return

    # 检查数据类型
    nps_dtype_str = nps_result_tensor.dtype
    if numpy_result.dtype.type not in nn.NUMPY_TO_DTYPE:
        print(
            f"✗ FAILED: NumPy 结果类型 '{numpy_result.dtype}' 未在 nn.NUMPY_TO_DTYPE 中定义。")
        return
    numpy_dtype_str = nn.NUMPY_TO_DTYPE[numpy_result.dtype.type]

    if nps_dtype_str != numpy_dtype_str:
        print(f"✗ FAILED: 数据类型不匹配!")
        print(f"  NPS Dtype: {nps_dtype_str}")
        print(f"  NumPy Dtype: {numpy_dtype_str}")
        return

    # 检查数值（为 cos 放宽容差）
    tolerance = 1e-5 if op_name == 'cos' else 1e-6
    if np.allclose(nps_result_data, numpy_result, atol=tolerance, rtol=1e-5):
        print(f"✅ SUCCESS: {op_name.upper()} -> {nps_dtype_str} (通过 NumPy 验证)")
    else:
        print(f"✗ FAILED: 数值不匹配!")
        diff = np.abs(nps_result_data - numpy_result).max()
        print(f"  - 最大差异: {diff}")

# =============================================================================
# 验证块三：对标 CUDA (精度验证)
# =============================================================================


def run_cuda_unary_op_verification(
        op_name, nps_operator_class, shape=(1, 3, 128, 128), dtype="float32"):
    """
    (一元算子) 对比 NPS 和 CUDA 的计算结果。
    """
    print(f"\n--- [CUDA验证] 测试 (一元): {op_name.upper()} ({dtype}{shape}) ---")

    # 1. 准备资源
    executable_path = f"./verify_{op_name.lower()}"
    input_file = "temp_input.bin"
    output_file = "temp_output.bin"

    if not os.path.exists(executable_path):
        print(f"⚠️ SKIPPED: CUDA 可执行文件 '{executable_path}' 未找到。请先编译它。")
        return

    # 2. 生成随机输入数据
    # 注意：CUDA 验证程序目前只接受 float32 输入
    np_input = create_test_data(shape, dtype).astype(np.float32)
    nps_input_tensor = Tensor(*shape, dtype=dtype, data=np_input)
    num_elements = np_input.size

    # 3. 获取 NPS 计算结果
    print(f"[NPS] ... 正在运行 NPS 算子...")
    nps_op = nps_operator_class(inputs=['x'], outputs=['y'], dtype=dtype)
    nps_result = nps_op.forward(nps_input_tensor)["tensor"].data

    # 4. 获取 CUDA "黄金标准" 计算结果
    print(f"[CUDA] ... 正在运行 '{executable_path}'...")
    np_input.tofile(input_file)
    try:
        subprocess.run([
            executable_path,
            str(num_elements),
            input_file,
            output_file
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: CUDA 程序执行失败: {e.stderr}")
        return

    cuda_result = np.fromfile(output_file, dtype=np.float32).reshape(shape)

    # 5. 对比结果
    print("[COMPARE] ... 正在对比 NPS 与 CUDA 结果...")

    # (NPS 结果可能是 float64/int32 等, CUDA 结果是 float32, 我们统一转为 float32 比较)
    # Cos 使用 LUT，容差放宽
    tolerance = 1e-5 if op_name == 'cos' else 1e-6
    if np.allclose(nps_result.astype(np.float32), cuda_result, atol=tolerance):
        print(f"✅ SUCCESS: {op_name.upper()} (通过 CUDA 精度验证)")
    else:
        print(f"❌ FAILED: {op_name.upper()} (CUDA 精度验证失败)!")
        diff = np.abs(nps_result.astype(np.float32) - cuda_result).max()
        print(f"  - 最大差异: {diff}")

    # 6. 清理
    os.remove(input_file)
    os.remove(output_file)


def run_cuda_binary_op_verification(
        op_name, nps_op_class, shape_a, shape_b, dtype_a="float32", dtype_b="float32"):
    """
    (二元算子) 对比 NPS 和 CUDA 的计算结果。
    它在Python层处理广播，以匹配NPS架构。
    """
    print(
        f"\n--- [CUDA验证] 测试 (二元): {op_name.upper()} ({dtype_a}{shape_a} + {dtype_b}{shape_b}) ---")

    # 1. 准备资源
    executable_path = f"./verify_{op_name.lower()}"
    input_file_a = "temp_input_a.bin"
    input_file_b = "temp_input_b.bin"
    output_file = "temp_output.bin"

    if not os.path.exists(executable_path):
        print(f"⚠️ SKIPPED: CUDA 可执行文件 '{executable_path}' 未找到。请先编译它。")
        return

    # 2. 生成NPS和Numpy输入
    np_input_a = create_test_data(shape_a, dtype_a)
    np_input_b = create_test_data(shape_b, dtype_b)
    nps_input_a = Tensor(*shape_a, dtype=dtype_a, data=np_input_a)
    nps_input_b = Tensor(*shape_b, dtype=dtype_b, data=np_input_b)

    # 3. 获取 NPS 计算结果
    print(f"[NPS] ... 正在运行 NPS 算子 (带内部广播)...")
    nps_op = nps_op_class(inputs=['a', 'b'], outputs=['y'], dtype="float32")
    nps_result_tensor = nps_op.forward(nps_input_a, nps_input_b)["tensor"]
    nps_result_data = nps_result_tensor.data

    # 4. 准备 CUDA 输入数据 (在Python层手动广播)
    print(f"[CUDA] ... 正在 (Python端) 预广播数据...")
    # (注意：CUDA 验证程序目前只接受 float32 输入)
    np_a_bcast, np_b_bcast = np.broadcast_arrays(
        np_input_a.astype(
            np.float32), np_input_b.astype(
            np.float32))

    num_elements = np_a_bcast.size
    np_a_bcast.tofile(input_file_a)
    np_b_bcast.tofile(input_file_b)

    # 5. 获取 CUDA "黄金标准" 计算结果
    print(f"[CUDA] ... 正在运行 '{executable_path}' (元素级计算)...")
    try:
        subprocess.run([
            executable_path,
            str(num_elements),
            input_file_a,
            input_file_b,
            output_file
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: CUDA 程序执行失败: {e.stderr}")
        return

    cuda_result = np.fromfile(
        output_file,
        dtype=np.float32).reshape(
        np_a_bcast.shape)

    # 6. 对比结果
    print("[COMPARE] ... 正在对比 NPS 与 CUDA 结果...")

    # 检查形状 (NPS的广播 vs NumPy的广播)
    if nps_result_data.shape != cuda_result.shape:
        print(f"❌ FAILED: 形状不匹配!")
        print(f"  NPS Shape:   {nps_result_data.shape}")
        print(f"  CUDA Shape: {cuda_result.shape}")
        return

    # 检查数值 (NPS V2 C实现 vs CUDA Kernel)
    # 我们的NPS C代码和CUDA kernel都应该在float32上执行，精度应该非常高
    if np.allclose(nps_result_data.astype(np.float32), cuda_result, atol=1e-7):
        print(f"✅ SUCCESS: {op_name.upper()} (通过 CUDA 精度验证)")
    else:
        print(f"❌ FAILED: {op_name.upper()} (CUDA 精度验证失败)!")
        diff = np.abs(nps_result_data.astype(np.float32) - cuda_result).max()
        print(f"  - 最大差异: {diff}")

    # 7. 清理
    os.remove(input_file_a)
    os.remove(input_file_b)
    os.remove(output_file)

def run_cuda_shape_op_verification(op_name, nps_operator_class, input_shape, dtype="float32", 
                                  shape_param=None, axis_param=None):
    """ 
    对比 NPS 和 CUDA 的形状操作结果（Reshape, Unsqueeze等）。
    主要验证数据内容是否保持不变。
    """ 
    print(f"\n--- [CUDA验证] 测试（形状操作）：{op_name.upper()} ({dtype}{input_shape}) ---")

    # 1. 准备资源
    executable_path = f"./verify_{op_name.lower()}"
    input_file = "temp_input.bin"
    output_file = "temp_output.bin"

    if not os.path.exists(executable_path):
        print(f"▲ SKIPPED: CUDA 可执行文件 '{executable_path}' 未找到。请先编译它。")
        return

    # 2. 生成输入数据
    np_input = create_test_data(input_shape, dtype).astype(np.float32)
    num_elements = np_input.size

    # 3. 获取 NPS 计算结果
    print(f"[NPS] ... 正在运行 NPS 算子...")
    
    if op_name == 'reshape':
        # 创建形状参数张量
        shape_tensor = Tensor(len(shape_param), dtype="int32", data=np.array(shape_param, dtype=np.int32))
        nps_op = nps_operator_class(inputs=['x', 'shape'], outputs=['y'], dtype=dtype)
        nps_result = nps_op.forward(Tensor(*input_shape, dtype=dtype, data=np_input), shape_tensor)["tensor"]
    elif op_name == 'unsqueeze':
        # 创建轴参数张量
        axis_tensor = Tensor(1, dtype="int32", data=np.array([axis_param], dtype=np.int32))
        nps_op = nps_operator_class(inputs=['x', 'axes'], outputs=['y'], dtype=dtype)
        nps_result = nps_op.forward(Tensor(*input_shape, dtype=dtype, data=np_input), axis_tensor)["tensor"]
    else:
        print(f"▲ SKIPPED: 未知的形状操作 {op_name}")
        return

    # 4. 获取 CUDA "黄金标准" 计算结果
    print(f"[CUDA] ... 正在运行 '{executable_path}'...")
    
    # 写入输入数据
    np_input.tofile(input_file)
    
    try:
        if op_name == 'reshape':
            # 将形状参数转换为字符串格式
            shape_str = "_".join(map(str, shape_param))
            subprocess.run([
                executable_path,
                str(num_elements),
                input_file,
                output_file,
                shape_str
            ], check=True, capture_output=True, text=True)
        elif op_name == 'unsqueeze':
            subprocess.run([
                executable_path,
                str(num_elements),
                input_file,
                output_file,
                str(axis_param)
            ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: CUDA 程序执行失败: {e.stderr}")
        return

    # 读取CUDA结果 - 注意：CUDA结果保持原始形状，不改变形状
    cuda_result = np.fromfile(output_file, dtype=np.float32).reshape(np_input.shape)

    # 5. 对比结果 - 对于形状操作，我们只验证数据内容是否保持不变
    print("[COMPARE] ... 正在对比 NPS 与 CUDA 结果...")
    
    # 将NPS结果展平以便与CUDA结果比较（因为形状不同）
    nps_flat = nps_result.data.flatten().astype(np.float32)
    cuda_flat = cuda_result.flatten()
    
    # 检查数据内容是否一致（允许很小的浮点误差）
    if np.allclose(nps_flat, cuda_flat, atol=1e-7):
        print(f"✅ SUCCESS: {op_name.upper()} [通过 CUDA 数据一致性验证]")
        print(f"   - 形状从 {input_shape} 变为 {nps_result.data.shape}")
        print(f"   - 数据内容保持不变")
    else:
        print(f"✗ FAILED: {op_name.upper()} (CUDA 数据一致性验证失败)")
        diff = np.abs(nps_flat - cuda_flat).max()
        print(f"   - 最大差异: {diff}")

    # 6. 清理
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

# =============================================================================
# 执行验证
# =============================================================================


if __name__ == "__main__":

    print("=========================================================")
    print(" 验证块二: 对标 NumPy (功能/广播/类型提升 验证) ")
    print("=========================================================")

    # --- RELU ---
    run_numpy_verification('relu', RELU, [("float32", (10, 10))])
    run_numpy_verification('relu', RELU, [("int32", (5, 5))])

    # --- COS ---
    run_numpy_verification('cos', COS, [("float32", (10, 10))])

    # --- ABS ---
    run_numpy_verification('abs', ABS, [("float32", (10, 10))])
    run_numpy_verification('abs', ABS, [("int32", (5, 5))])

    # --- ADD ---
    run_numpy_verification(
        'add', ADD, [
            ("float32", (3, 32, 32)), ("float32", (3, 32, 32))])  # 相同形状
    run_numpy_verification(
        'add', ADD, [
            ("float32", (3, 32, 32)), ("float32", (32, 32))])    # 广播 B
    run_numpy_verification(
        'add', ADD, [
            ("float32", (32, 32)), ("float32", (3, 32, 32))])    # 广播 A
    run_numpy_verification(
        'add', ADD, [
            ("int32", (10, 10)), ("float32", (10, 10))])      # 混合精度 -> float32
    run_numpy_verification(
        'add', ADD, [
            ("float16", (10, 10)), ("float32", (10, 10))])      # 混合精度 -> float32
    run_numpy_verification(
        'add', ADD, [
            ("int32", (10, 10)), ("int32", (10, 10))])        # 相同 int -> int32
    run_numpy_verification(
        'add', ADD, [
            ("int64", (10, 10)), ("int32", (1, 10))])       # 混合 int + 广播 -> int64

    # --- TANH ---
    run_numpy_verification('tanh', TANH, [("float32", (10, 10))])
    run_numpy_verification('tanh', TANH, [("float32", (5, 5))])

    # --- RESHAPE ---
    # 使用固定形状值而不是随机数据
    run_numpy_verification('reshape', RESHAPE,
                           [("float32", (2, 3)), ("int32", (2,))],
                           fixed_inputs=[None, np.array([3, 2], dtype=np.int32)])  # (2,3) -> (3,2)

    run_numpy_verification('reshape', RESHAPE,
                           [("float32", (6,)), ("int32", (2,))],
                           fixed_inputs=[None, np.array([2, 3], dtype=np.int32)])  # (6,) -> (2,3)

    # --- UNSQUEEZE ---
    # 使用固定轴值而不是随机数据
    run_numpy_verification('unsqueeze', UNSQUEEZE,
                           [("float32", (2, 3)), ("int32", (1,))],
                           fixed_inputs=[None, np.array([1], dtype=np.int32)])  # 在axis=1处插入维度

    run_numpy_verification('unsqueeze', UNSQUEEZE,
                           [("float32", (2, 3)), ("int32", (1,))],
                           fixed_inputs=[None, np.array([0], dtype=np.int32)])  # 在axis=0处插入维度

    print("=========================================================")
    print(" 验证块三：对标 CUDA（精度验证） ")
    print("=========================================================")

    # --- 一元算子 ---
    run_cuda_unary_op_verification(
        'relu', RELU, shape=(
            10, 10), dtype="float32")
    run_cuda_unary_op_verification('cos', COS, shape=(10, 10), dtype="float32")
    run_cuda_unary_op_verification('abs', ABS, shape=(10, 10), dtype="float32")
    run_cuda_unary_op_verification(
        'tanh', TANH, shape=(
            10, 10), dtype="float32")  # 添加 Tanh

    # --- 二元算子 (ADD) ---
    # 测试1: 相同形状
    run_cuda_binary_op_verification(
        'add', ADD, shape_a=(
            10, 10), dtype_a="float32", shape_b=(
            10, 10), dtype_b="float32")

    # 测试2: 广播
    run_cuda_binary_op_verification(
        'add', ADD, shape_a=(
            5, 10), dtype_a="float32", shape_b=(
            10,), dtype_b="float32")

    # 测试3: 混合精度 (NPS(int+float->float) vs CUDA(float+float))
    # NPS 会将 int32 转为 float32 (在C层)，CUDA 也接收 float32
    run_cuda_binary_op_verification(
        'add', ADD, shape_a=(
            10, 10), dtype_a="int32", shape_b=(
            10, 10), dtype_b="float32")
    # --- 形状操作算子 ---
    # Reshape验证
    run_cuda_shape_op_verification('reshape', RESHAPE, 
                                  input_shape=(2, 3), dtype="float32",
                                  shape_param=[3, 2])  # (2,3) -> (3,2)
    
    run_cuda_shape_op_verification('reshape', RESHAPE,
                                  input_shape=(6,), dtype="float32", 
                                  shape_param=[2, 3])  # (6,) -> (2,3)

    # Unsqueeze验证
    run_cuda_shape_op_verification('unsqueeze', UNSQUEEZE,
                                  input_shape=(2, 3), dtype="float32",
                                  axis_param=1)  # (2,3) -> (2,1,3)
    
    run_cuda_shape_op_verification('unsqueeze', UNSQUEEZE,
                                  input_shape=(2, 3), dtype="float32", 
                                  axis_param=0)  # (2,3) -> (1,2,3)
