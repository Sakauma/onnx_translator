import numpy as np
import subprocess
import os
import nn
from nn import Tensor
from nn.Operators import RELU, COS, ABS, ADD, SUB, MUL, DIV

# --- 辅助函数 ---
def decode_float8_e4m3(val_uint8):
    """Python 实现的 E4M3 到 float 的解码 (用于验证 C 后端结果)"""
    # 1位符号, 4位指数, 3位尾数, Bias 7
    s = (val_uint8 & 0x80) >> 7
    e = (val_uint8 & 0x78) >> 3
    m = (val_uint8 & 0x07)
    
    sign = -1.0 if s else 1.0
    
    if e == 0:
        if m == 0: return 0.0
        # 次正规数: 0.m * 2^(-6)
        return sign * (m / 8.0) * (2 ** -6)
    elif e == 0xF and m == 0x7:
        return np.nan
    else:
        # 正规数: 1.m * 2^(e-7)
        return sign * (1.0 + m / 8.0) * (2 ** (e - 7))

def decode_float8_e5m2(val_uint8):
    """Python 实现的 E5M2 到 float 的解码"""
    # 1位符号, 5位指数, 2位尾数, Bias 15
    s = (val_uint8 & 0x80) >> 7
    e = (val_uint8 & 0x7C) >> 2
    m = (val_uint8 & 0x03)
    
    sign = -1.0 if s else 1.0
    
    if e == 0:
        if m == 0: return 0.0
        # 次正规数: 0.m * 2^(-14)
        return sign * (m / 4.0) * (2 ** -14)
    elif e == 0x1F:
        return np.inf if m == 0 else np.nan
    else:
        # 正规数: 1.m * 2^(e-15)
        return sign * (1.0 + m / 4.0) * (2 ** (e - 15))

# 向量化解码函数，方便处理整个数组
vec_decode_e4m3 = np.vectorize(decode_float8_e4m3)
vec_decode_e5m2 = np.vectorize(decode_float8_e5m2)

def create_test_data(shape, dtype):
    """生成测试数据"""
    if dtype == "float8_e4m3" or dtype == "float8_e5m2":
        # 生成随机 uint8 位模式
        return np.random.randint(0, 255, shape).astype(np.uint8)
    elif dtype == "int4":
        return np.random.randint(-8, 7, shape).astype(np.int8)
    
    if dtype in nn.DTYPE_TO_NUMPY:
        np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    else:
        np_dtype = np.float32 # 默认兜底
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
            # 除法简化
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
    # - 如果 NumPy 算出来是浮点 -> NPS 统一配置为 float32  (可以更改)
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
    executable_path = f"./cache/verify_{op_name.lower()}"
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
    executable_path = f"./cache/verify_{op_name.lower()}"
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
    
# =============================================================================
# 验证块四: 边缘计算混合精度 (A8W4 / A16W8)
# =============================================================================
    
def run_mixed_precision_check():
    """
    专门验证边缘计算场景下的混合精度 (A8W4, A16W8 等)
    """
    shape = (4, 4)
    # --- Case 1: A16W8 (Float16 + Int8) ---
    print("\n[Case 1] A16W8: Float16 (Act) * Int8 (Weight) -> Float16")
    
    # 准备数据
    data_a = np.random.randn(*shape).astype(np.float16)
    data_w = np.random.randint(-10, 10, shape).astype(np.int8)
    
    t_a = Tensor(*shape, dtype="float16", data=data_a)
    t_w = Tensor(*shape, dtype="int8", data=data_w)
    
    # 定义算子: 输出指定为 float16
    mul_op = MUL(inputs=[], outputs=[], dtype="float16")
    nps_res = mul_op.forward(t_a, t_w)["tensor"].data
    
    # Ground Truth: 提升到 float32 计算，再转回 float16
    gt_res = (data_a.astype(np.float32) * data_w.astype(np.float32)).astype(np.float16)
    
    # 验证 (使用较大容差，因为中间计算精度不同)
    if np.allclose(nps_res, gt_res, atol=1e-3):
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
        print("NPS sample:", nps_res.flatten()[:3])
        print("GT  sample:", gt_res.flatten()[:3])
        
    # --- Case 2: A32W4 (Float32 + Int4) ---
    print("\n[Case 2] A32W4: Float32 (Act) + Int4 (Weight) -> Float16")
    
    data_a = np.random.randn(*shape).astype(np.float32)
    data_w = np.random.randint(-8, 7, shape).astype(np.int8) # Store as int8
    
    t_a = Tensor(*shape, dtype="float32", data=data_a)
    t_w = Tensor(*shape, dtype="int4", data=data_w)
    
    add_op = ADD(inputs=[], outputs=[], dtype="float16")
    nps_res = add_op.forward(t_a, t_w)["tensor"].data
    
    gt_res = (data_a + data_w.astype(np.float32)).astype(np.float16)
    
    if np.allclose(nps_res, gt_res, atol=1e-3):
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
        
    # --- Case 3: A8W8 (Float8_E4M3 + Int8) ---
    print("\n[Case 3] A8W8: Float8_E4M3 (Act) * Int8 (Weight) -> Float16")
    
    # 生成随机 Float8 位模式
    data_a_bits = np.random.randint(0, 255, shape).astype(np.uint8)
    # 解码为 float32 用于计算 GT
    data_a_float = vec_decode_e4m3(data_a_bits)
    # 过滤掉 NaN 用于测试
    mask = ~np.isnan(data_a_float)
    data_a_bits = data_a_bits * mask # 简单处理，NaN位置变0
    data_a_float = np.nan_to_num(data_a_float)

    data_w = np.random.randint(-5, 5, shape).astype(np.int8)
    
    t_a = Tensor(*shape, dtype="float8_e4m3", data=data_a_bits.astype(np.uint8))
    t_w = Tensor(*shape, dtype="int8", data=data_w)
    
    mul_op = MUL(inputs=[], outputs=[], dtype="float16")
    nps_res = mul_op.forward(t_a, t_w)["tensor"].data
    
    # GT
    gt_res = (data_a_float * data_w.astype(np.float32)).astype(np.float16)
    
    # 验证 (排除 GT 溢出导致 Inf 的情况)
    valid_mask = np.isfinite(gt_res)
    if np.allclose(nps_res[valid_mask], gt_res[valid_mask], atol=1e-2):
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
        diff = np.abs(nps_res - gt_res)
        max_diff_idx = np.unravel_index(np.argmax(diff * valid_mask), diff.shape)
        print(f"Max Diff at {max_diff_idx}: NPS={nps_res[max_diff_idx]}, GT={gt_res[max_diff_idx]}")
        
        
def run_float8_verification():
    """
    验证 Float8 (E4M3/E5M2) 的基本功能
    """
    print("\n=========================================================")
    print(" [Part 3] 验证: Float8 (E4M3 & E5M2) 功能测试 ")
    print("=========================================================")
    
    # --- 测试 E4M3 解析 ---
    print("\n[测试] Float8 E4M3 -> Float32 解析准确性")
    # 构造特定数值: 0x3C (1.5), 0xC0 (-2.0)
    vals = np.array([0x3C, 0xC0], dtype=np.uint8)
    t_in = Tensor(2, dtype="float8_e4m3", data=vals)
    
    # 使用 ABS 算子触发解析: Abs(1.5)=1.5, Abs(-2.0)=2.0
    abs_op = ABS(inputs=[], outputs=[], dtype="float32")
    res = abs_op.forward(t_in)["tensor"].data
    
    if res[0] == 1.5 and res[1] == 2.0:
        print("✅ SUCCESS: 特定数值解析正确")
    else:
        print(f"❌ FAILED: 预期 [1.5, 2.0], 实际 {res}")

    # --- 测试 E5M2 解析 ---
    print("\n[测试] Float8 E5M2 -> Float32 解析准确性")
    # 0x3E -> 1.5 (E5M2)
    vals = np.array([0x3E], dtype=np.uint8)
    t_in = Tensor(1, dtype="float8_e5m2", data=vals)
    
    # 用 Add 加 0.5: 1.5 + 0.5 = 2.0
    t_add = Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32))
    add_op = ADD(inputs=[], outputs=[], dtype="float32")
    res = add_op.forward(t_in, t_add)["tensor"].data
    
    if res[0] == 2.0:
        print("✅ SUCCESS: E5M2 计算正确")
    else:
        print(f"❌ FAILED: 预期 2.0, 实际 {res[0]}")

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
    
    print("\n=========================================================")
    print(" 验证块四: 边缘计算混合精度 (A8W4 / A16W8)")
    print("=========================================================")
    
    run_mixed_precision_check()
    run_float8_verification()