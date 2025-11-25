import numpy as np
import subprocess
import os
import nn
from nn import Tensor
from nn.Operators import RELU, COS, ABS, ADD # 导入所有已实现的算子

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

def run_numpy_verification(op_name, nps_op_class, input_specs):
    """
    一个通用的验证函数，用于对比 NPS 和 NumPy (作为标准) 的计算结果。
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
    for dtype, shape in input_specs:
        np_data = create_test_data(shape, dtype)
        np_inputs.append(np_data)
        nps_inputs.append(Tensor(*shape, dtype=dtype, data=np_data))

    # # 2. 获取 NPS 计算结果
    # print(f"[NPS] ... 正在运行 NPS 算子...")
    # nps_op = nps_op_class(inputs=['a', 'b'][:len(nps_inputs)], outputs=['y'], dtype="float32")
    # nps_result_tensor = nps_op.forward(*nps_inputs)["tensor"]
    # nps_result_data = nps_result_tensor.data

    # 2. 获取 NumPy 计算结果
    print(f"[NumPy] ... 正在运行 NumPy 算子...")
    numpy_result = None
    try:
        if op_name == 'add':
            numpy_result = np_inputs[0] + np_inputs[1]
        elif op_name == 'sub':
            numpy_result = np_inputs[0] - np_inputs[1]
        elif op_name == 'mul':
            numpy_result = np_inputs[0] * np_inputs[1]
        elif op_name == 'div':
            # 注意：Python/NumPy 的除法行为比较复杂，这里直接使用默认行为
            numpy_result = np_inputs[0] / np_inputs[1]
        elif op_name == 'abs':
            numpy_result = np.abs(np_inputs[0])
        elif op_name == 'cos':
            # Cos 在 NumPy 中默认输出 float64
            numpy_result = np.cos(np_inputs[0].astype(np.float64))
        elif op_name == 'relu':
            numpy_result = np.maximum(0, np_inputs[0])
        else:
            print(f"警告: {op_name} 的NumPy验证规则未定义。")
            return
    except Exception as e:
        print(f"❌ FAILED: NumPy 计算失败: {e}")
        return
    
    # 3. 智能推断 NPS 应使用的 dtype
    # - 如果 NumPy 算出来是整数 -> NPS 也应配置为输出对应的整数类型 (int32, int64 等)
    # - 如果 NumPy 算出来是浮点 -> NPS 统一配置为 float32 (为了兼容性和C后端默认行为)
    target_dtype = "float32" # 默认
    
    if numpy_result.dtype.type in nn.NUMPY_TO_DTYPE:
        np_type_str = nn.NUMPY_TO_DTYPE[numpy_result.dtype.type]
        if "int" in np_type_str:
            # 如果是整数结果 (如 int32 + int32 -> int32)，保持类型一致
            target_dtype = np_type_str
        elif "float" in np_type_str:
            # 如果是浮点结果，强制使用 float32 进行验证
            target_dtype = "float32"
    
    # 4. 运行 NPS 算子
    print(f"[NPS] ... 正在运行 NPS 算子 (预期输出: {target_dtype})...")
    try:
        nps_op = nps_op_class(inputs=['a', 'b'][:len(nps_inputs)], outputs=['y'], dtype=target_dtype)
        nps_result_tensor = nps_op.forward(*nps_inputs)["tensor"]
        nps_result_data = nps_result_tensor.data
    except Exception as e:
        print(f"❌ FAILED: NPS 算子执行失败: {e}")
        return

    # 5. 对比结果
    print("[COMPARE] ... 正在对比 NPS 与 NumPy 结果...")
    
    # # 检查形状
    # if nps_result_data.shape != numpy_result.shape:
    #     print(f"❌ FAILED: 形状不匹配!")
    #     print(f"  NPS Shape:   {nps_result_data.shape}")
    #     print(f"  NumPy Shape: {numpy_result.shape}")
    #     return

    # # 检查数据类型
    # nps_dtype_str = nps_result_tensor.dtype
    # if numpy_result.dtype.type not in nn.NUMPY_TO_DTYPE:
    #      print(f"❌ FAILED: NumPy 结果类型 '{numpy_result.dtype}' 未在 nn.NUMPY_TO_DTYPE 中定义。")
    #      return
    # numpy_dtype_str = nn.NUMPY_TO_DTYPE[numpy_result.dtype.type]

    # if nps_dtype_str != numpy_dtype_str:
    #      print(f"❌ FAILED: 数据类型不匹配!")
    #      print(f"  NPS Dtype:   {nps_dtype_str}")
    #      print(f"  NumPy Dtype: {numpy_dtype_str}")
    #      return
    # 数据准备：如果 NPS 是 float32 而 NumPy 是 float64，将 NumPy 降级为 float32 再对比
    if target_dtype == "float32" and numpy_result.dtype == np.float64:
        numpy_result = numpy_result.astype(np.float32)
    
    # 检查形状
    if nps_result_data.shape != numpy_result.shape:
        print(f"❌ FAILED: 形状不匹配!")
        print(f"  NPS:   {nps_result_data.shape}")
        print(f"  NumPy: {numpy_result.shape}")
        return

    # 检查数值 (为 cos 放宽容差)
    tolerance = 1e-4 if (op_name == 'cos' or target_dtype == 'float16') else 1e-5
    
    if np.allclose(nps_result_data, numpy_result, atol=tolerance, rtol=1e-5):
        print(f"✅ SUCCESS: {op_name.upper()} -> {target_dtype} (匹配)")
    else:
        print(f"❌ FAILED: 数值不匹配!")
        # 转换为 float64 计算差异，避免溢出
        diff = np.abs(nps_result_data.astype(np.float64) - numpy_result.astype(np.float64)).max()
        print(f"  - 最大差异: {diff}")
        print(f"  - NPS 样本: {nps_result_data.flatten()[:3]}")
        print(f"  - NumPy 样本: {numpy_result.flatten()[:3]}")


# =============================================================================
# 验证块三：对标 CUDA (精度验证)
# =============================================================================

def run_cuda_unary_op_verification(op_name, nps_operator_class, shape=(1, 3, 128, 128), dtype="float32"):
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
    
    # 4. 获取 CUDA 计算结果
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


def run_cuda_binary_op_verification(op_name, nps_op_class, shape_a, shape_b, dtype_a="float32", dtype_b="float32"):
    """
    (二元算子) 对比 NPS 和 CUDA 的计算结果。
    它在Python层处理广播，以匹配NPS架构。
    """
    print(f"\n--- [CUDA验证] 测试 (二元): {op_name.upper()} ({dtype_a}{shape_a} + {dtype_b}{shape_b}) ---")

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
    np_a_bcast, np_b_bcast = np.broadcast_arrays(np_input_a.astype(np.float32), np_input_b.astype(np.float32))
    
    num_elements = np_a_bcast.size
    np_a_bcast.tofile(input_file_a)
    np_b_bcast.tofile(input_file_b)

    # 5. 获取 CUDA 计算结果
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
        
    cuda_result = np.fromfile(output_file, dtype=np.float32).reshape(np_a_bcast.shape)

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
    
def run_mixed_precision_check():
    """
    专门验证边缘计算场景下的混合精度 (A8W4, A16W8 等)
    """
    print("\n=========================================================")
    print(" 验证块四: 边缘计算混合精度 (A8W4 / A16W8) ")
    print("=========================================================")

    # --- 测试案例 1: A8W4 (Activation Int8 + Weight Int4 -> Int32 Accum) ---
    print("\n[测试] A8W4 加法 (Int8 + Int4 -> Int32)")
    shape = (4, 4)
    
    # 构造数据
    # A (Activation): 完整 Int8 范围 [-128, 127]
    data_a = np.random.randint(-128, 127, shape).astype(np.int8)
    # B (Weight): Int4 范围 [-8, 7]，存储在 int8 中
    data_b = np.random.randint(-8, 7, shape).astype(np.int8)
    
    # NPS Tensor
    t_a = Tensor(*shape, dtype="int8", data=data_a)
    t_b = Tensor(*shape, dtype="int4", data=data_b)
    
    # 执行 NPS Add (指定输出为 int32)
    add_op = nn.Operators.ADD(inputs=[], outputs=[], dtype="int32")
    nps_res = add_op.forward(t_a, t_b)["tensor"].data
    
    # 执行真值计算 (NumPy 自动提升)
    gt_res = data_a.astype(np.int32) + data_b.astype(np.int32)
    
    if np.array_equal(nps_res, gt_res):
        print("✅ SUCCESS: A8W4 计算正确 (无溢出)")
    else:
        print("❌ FAILED: A8W4 计算错误")
        print("Diff:", nps_res - gt_res)

    # --- 测试案例 2: A8W4 饱和测试 (Int8 + Int4 -> Int8 Output) ---
    # 如果输出被限制为 Int8，则应该发生饱和截断，而不是回绕
    print("\n[测试] A8W4 饱和截断 (Int8 + Int4 -> Int8 Output)")
    
    # 构造必定溢出的数据: 125 + 5 = 130 -> 应该截断为 127 (而不是 -126)
    data_a_sat = np.full(shape, 125, dtype=np.int8)
    data_b_sat = np.full(shape, 5, dtype=np.int8)
    
    t_a_sat = Tensor(*shape, dtype="int8", data=data_a_sat)
    t_b_sat = Tensor(*shape, dtype="int4", data=data_b_sat)
    
    add_op_sat = nn.Operators.ADD(inputs=[], outputs=[], dtype="int8")
    nps_res_sat = add_op_sat.forward(t_a_sat, t_b_sat)["tensor"].data
    
    # 验证是否全部为 127
    if np.all(nps_res_sat == 127):
        print("✅ SUCCESS: 饱和截断逻辑正确 (125 + 5 -> 127)")
    else:
        print("❌ FAILED: 饱和截断失败")
        print("Result sample:", nps_res_sat[0,0])

    # --- 测试案例 3: A16W8 (Int16 + Int8 -> Int32) ---
    print("\n[测试] A16W8 加法 (Int16 + Int8 -> Int32)")
    # A (Activation): Int16
    data_a_16 = np.random.randint(-30000, 30000, shape).astype(np.int16)
    # B (Weight): Int8
    data_b_8 = np.random.randint(-128, 127, shape).astype(np.int8)
    
    t_a_16 = Tensor(*shape, dtype="int16", data=data_a_16)
    t_b_8 = Tensor(*shape, dtype="int8", data=data_b_8)
    
    add_op_16 = nn.Operators.ADD(inputs=[], outputs=[], dtype="int32")
    nps_res_16 = add_op_16.forward(t_a_16, t_b_8)["tensor"].data
    
    gt_res_16 = data_a_16.astype(np.int32) + data_b_8.astype(np.int32)
    
    if np.array_equal(nps_res_16, gt_res_16):
        print("✅ SUCCESS: A16W8 计算正确")
    else:
        print("❌ FAILED: A16W8 计算错误")


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
    run_numpy_verification('add', ADD, [("float32", (3, 32, 32)), ("float32", (3, 32, 32))]) # 相同形状
    run_numpy_verification('add', ADD, [("float32", (3, 32, 32)), ("float32", (32, 32))])    # 广播 B
    run_numpy_verification('add', ADD, [("float32", (32, 32)), ("float32", (3, 32, 32))])    # 广播 A
    run_numpy_verification('add', ADD, [("int32", (10, 10)), ("float32", (10, 10))])      # 混合精度 -> float32
    run_numpy_verification('add', ADD, [("float16", (10, 10)), ("float32", (10, 10))])      # 混合精度 -> float32
    run_numpy_verification('add', ADD, [("int32", (10, 10)), ("int32", (10, 10))])        # 相同 int -> int32
    run_numpy_verification('add', ADD, [("int64", (10, 10)), ("int32", (1, 10))])       # 混合 int + 广播 -> int64

    
    print("\n=========================================================")
    print(" 验证块三: 对标 CUDA (精度验证) ")
    print("=========================================================")
    
    # --- 一元算子 ---
    run_cuda_unary_op_verification('relu', RELU, shape=(10, 10), dtype="float32")
    run_cuda_unary_op_verification('cos', COS, shape=(10, 10), dtype="float32")
    run_cuda_unary_op_verification('abs', ABS, shape=(10, 10), dtype="float32")
    
    # --- 二元算子 (ADD) ---
    # 测试1: 相同形状
    run_cuda_binary_op_verification('add', ADD, 
                                    shape_a=(10, 10), dtype_a="float32",
                                    shape_b=(10, 10), dtype_b="float32")
                                    
    # 测试2: 广播
    run_cuda_binary_op_verification('add', ADD, 
                                    shape_a=(5, 10), dtype_a="float32",
                                    shape_b=(10,), dtype_b="float32")
                                    
    # 测试3: 混合精度 (NPS(int+float->float) vs CUDA(float+float))
    # NPS 会将 int32 转为 float32 (在C层)，CUDA 也接收 float32
    run_cuda_binary_op_verification('add', ADD, 
                                    shape_a=(10, 10), dtype_a="int32",
                                    shape_b=(10, 10), dtype_b="float32")
    
    run_mixed_precision_check()