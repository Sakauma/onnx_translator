import numpy as np
import subprocess
import os
import nn
from nn import Tensor
import matplotlib.pyplot as plt
from nn.Operators import RELU, COS, ABS, ADD, SUB, MUL, DIV, QuantizeLinear, DequantizeLinear, Conv, MaxPool, Gemm, Softmax

# =============================================================================
# 1. 辅助工具
# =============================================================================

def get_dtype_limits(dtype):
    """
    获取不同数据类型的数值范围限制
    返回: (min_val, max_val, is_saturating)
    is_saturating=True 表示溢出时应该卡在最大值 (如 Int8, E4M3)
    is_saturating=False 表示溢出时应该变 Inf (如 FP16, FP32)
    """
    if dtype == "float16":
        return -65504.0, 65504.0, False
    if dtype == "bfloat16":
        return -3.38e38, 3.38e38, False
    if dtype == "float8_e4m3":
        return -448.0, 448.0, True 
    if dtype == "float8_e5m2":
        return -57344.0, 57344.0, False
    if dtype == "int8":
        return -128, 127, True
    if dtype == "int4":
        return -8, 7, True
    if dtype == "int32":
        return -2147483648, 2147483647, False 
    
    # Float32 视为无限
    return -float('inf'), float('inf'), False

def float32_to_bfloat16_bits(arr_f32):
    """
    将 float32 数组转换为 bfloat16 的位存储 (uint16)
    """
    u32 = arr_f32.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    guard = (u32 >> 15) & 1
    sticky = (u32 & 0x7FFF) != 0
    round_up = guard & (sticky | lsb)
    u32_rounded = u32 + (round_up.astype(np.uint32) << 16)
    is_nan = np.isnan(arr_f32)
    final_u32 = np.where(is_nan, u32, u32_rounded)

    return (final_u32 >> 16).astype(np.uint16)

def bfloat16_bits_to_float32(arr_u16):
    """
    将 bfloat16 位存储 (uint16) 还原为 float32
    左移 16 位
    """
    arr_u32 = arr_u16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)

def decode_float8_e4m3(val_uint8):
    val_uint8 = int(val_uint8)
    s = (val_uint8 & 0x80) >> 7
    e = (val_uint8 & 0x78) >> 3
    m = (val_uint8 & 0x07)
    sign = -1.0 if s else 1.0
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6) if m != 0 else 0.0
    elif e == 0xF and m == 0x7:
        return np.nan
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))

def decode_float8_e5m2(val_uint8):
    val_uint8 = int(val_uint8)
    s = (val_uint8 & 0x80) >> 7
    e = (val_uint8 & 0x7C) >> 2
    m = (val_uint8 & 0x03)
    sign = -1.0 if s else 1.0
    if e == 0:
        return sign * (m / 4.0) * (2.0 ** -14) if m != 0 else 0.0
    elif e == 0x1F:
        return (sign * np.inf) if m == 0 else np.nan
    return sign * (1.0 + m / 4.0) * (2.0 ** (e - 15))

vec_decode_e4m3 = np.vectorize(decode_float8_e4m3)
vec_decode_e5m2 = np.vectorize(decode_float8_e5m2)

def to_float32(data, dtype):
    """
    关键修复：将存储在 int 容器中的位模式正确解码为 float32 数值
    """
    # 1. BFloat16: data 是 uint16 位模式 -> 需要位解码
    if dtype == "bfloat16":
        return bfloat16_bits_to_float32(data)
    
    # 2. Float8: data 是 uint8 位模式 -> 需要查表解码
    if "float8_e4m3" in dtype: return vec_decode_e4m3(data).astype(np.float32)
    if "float8_e5m2" in dtype: return vec_decode_e5m2(data).astype(np.float32)
    
    # 3. Float16: numpy 原生支持
    if dtype == "float16": return data.astype(np.float32)
    
    # 4. 整数类型: 直接转换数值
    return data.astype(np.float32)

# =============================================================================
# 2. 数据生成 (修复 BFloat16 生成逻辑)
# =============================================================================

def generate_random_data(shape, dtype):
    size = np.prod(shape)
    
    # --- 整数生成 ---
    if "int" in dtype and "float" not in dtype:
        if dtype == "int4": return np.random.randint(-7, 7, shape).astype(np.int8)
        if dtype == "int8": return np.random.randint(-120, 120, shape).astype(np.int8)
        limit = 1000
        return np.random.randint(-limit, limit, shape).astype(nn.DTYPE_TO_NUMPY.get(dtype, np.int32))

    # --- 浮点位模式生成 (Float8) ---
    if "float8" in dtype:
        return np.random.randint(0, 256, size=shape).astype(np.uint8)

    # --- 浮点数值生成 (Float16/32/BF16) ---
    # 策略: 50% 常规, 25% 大数(溢出测试), 25% 小数(精度测试)
    part_normal = np.random.uniform(-10, 10, size=size)
    part_large = np.random.uniform(-1000, 1000, size=size)
    part_tiny = np.random.uniform(-0.01, 0.01, size=size)
    
    choices = np.random.choice([0, 1, 2], size=size, p=[0.5, 0.25, 0.25])
    raw_f32 = np.select([choices==0, choices==1, choices==2], 
                         [part_normal, part_large, part_tiny]).reshape(shape)
    if dtype == "bfloat16":
        return float32_to_bfloat16_bits(raw_f32) 
    if dtype == "float16":
        return raw_f32.astype(np.float16)
        
    return raw_f32.astype(np.float32)

# =============================================================================
# 3. 验证与执行逻辑
# =============================================================================

def run_cuda_ground_truth(op_name, inputs_f32, params_binary=None, output_dtype=np.float32, target_shape=None):
    exe = f"./cache/verify_{op_name}"
    if not os.path.exists(exe):
        print(f"⚠️  Missing CUDA executable: {exe}")
        return None
        
    cuda_inputs = list(inputs_f32) # Copy list

    files = []
    for i, arr in enumerate(cuda_inputs):
        if arr is None:
            files.append("null")
            continue
        fname = f"tmp_in_{i}.bin"
        #arr.tofile(fname)
        np.ascontiguousarray(arr).tofile(fname)
        files.append(fname)
    
    if params_binary is not None:
        p_fname = "tmp_params.bin"
        with open(p_fname, "wb") as f:
            f.write(params_binary)
        files.append(p_fname)

    out_fname = "tmp_out.bin"
    
    try:
        args = [exe, str(cuda_inputs[0].size)] + files + [out_fname]
        if target_shape is not None:
             out_elem_count = np.prod(target_shape)
             args[1] = str(out_elem_count)

        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        final_shape = target_shape if target_shape is not None else cuda_inputs[0].shape
        result = np.fromfile(out_fname, dtype=output_dtype).reshape(final_shape)
        
    except Exception as e:
        print(f"CUDA Fail [{op_name}]: {e}") 
        result = None
    finally:
        for f in files:
            if f != "null" and os.path.exists(f): os.remove(f)
        if os.path.exists(out_fname): os.remove(out_fname)
            
    return result

def check_accuracy(nps_val, cuda_val, atol, rtol, dtype):
    """
    严谨的验证逻辑：支持数值对比、溢出判定和 NaN 匹配
    """
    min_limit, max_limit, is_saturating = get_dtype_limits(dtype)
    nan_match = np.isnan(nps_val) & np.isnan(cuda_val)
    inf_match = np.isinf(nps_val) & np.isinf(cuda_val) & (np.sign(nps_val) == np.sign(cuda_val))
    
    cuda_finite = np.isfinite(cuda_val)
    gt_overflow_pos = cuda_finite & (cuda_val > max_limit)
    gt_overflow_neg = cuda_finite & (cuda_val < min_limit)
    
    if is_saturating:
        overflow_pos_match = gt_overflow_pos & (nps_val == max_limit)
        overflow_neg_match = gt_overflow_neg & (nps_val == min_limit)
    else:
        overflow_pos_match = gt_overflow_pos & (nps_val == np.inf)
        overflow_neg_match = gt_overflow_neg & (nps_val == -np.inf)
        
    logic_pass = nan_match | inf_match | overflow_pos_match | overflow_neg_match
    valid_numeric_mask = np.isfinite(nps_val) & np.isfinite(cuda_val)
    current_max_abs = 0.0
    current_max_rel = 0.0
    numeric_pass_mask = np.zeros_like(nps_val, dtype=bool)
    
    if np.any(valid_numeric_mask):
        # 提取数值
        v_nps = nps_val[valid_numeric_mask]
        v_cuda = cuda_val[valid_numeric_mask]
        
        # 计算误差
        diff = np.abs(v_nps - v_cuda)
        ref = np.abs(v_cuda) + 1e-12 # 防止除零
        rel = diff / ref
        
        current_max_abs = np.max(diff)
        current_max_rel = np.max(rel)
        
        tolerance = atol + rtol * np.abs(v_cuda)
        is_close = diff <= tolerance
        
        numeric_pass_mask[valid_numeric_mask] = is_close

    final_pass = logic_pass | numeric_pass_mask
    fail_mask = ~final_pass
    
    if np.all(final_pass):
        return True, current_max_abs, current_max_rel, None
    else:
        numeric_fail = fail_mask & valid_numeric_mask
        
        fail_abs = -1.0
        fail_rel = -1.0
        
        if np.any(numeric_fail):
            diff = np.abs(nps_val[numeric_fail] - cuda_val[numeric_fail])
            fail_abs = np.max(diff)
            ref = np.abs(cuda_val[numeric_fail]) + 1e-12
            fail_rel = np.max(diff / ref)
        elif np.any(fail_mask):
            fail_abs = -999.0
            fail_rel = -999.0
            
        return False, fail_abs, fail_rel, fail_mask

def verify_op(op_cls, op_name, shapes, dtypes, out_dtype, init_args={}, iterations=5):
    print(f"🧪 Testing {op_name.upper()}: {dtypes} -> {out_dtype}")
    
    atol, rtol = 1e-4, 1e-4
    if "float16" in out_dtype: atol, rtol = 0.01, 0.01 
    if "bfloat16" in out_dtype: atol, rtol = 0.1, 0.02
    if "float8" in out_dtype: atol, rtol = 0.1, 0.1    
    if "int" in out_dtype: atol, rtol = 0, 0
    if op_name == "cos": atol = max(atol, 0.02)

    pass_cnt = 0
    stats_abs = []
    stats_rel = []
    
    for i in range(iterations):
        # 1. 生成数据
        inputs_np = []
        for s, d in zip(shapes, dtypes):
            if s is None: inputs_np.append(None)
            else: inputs_np.append(generate_random_data(s, d))
            
        if op_name in ["quantize_linear", "dequantize_linear"]:
            if inputs_np[1] is not None: inputs_np[1] = np.abs(inputs_np[1]) + 1e-4
            if inputs_np[2] is not None:
                inputs_np[2] = np.round(inputs_np[2])
                if op_name == "quantize_linear": inputs_np[2] = np.clip(inputs_np[2], -128, 127)

        inputs_tensor = []
        for data, d in zip(inputs_np, dtypes):
            if data is not None: inputs_tensor.append(Tensor(*data.shape, dtype=d, data=data))
            else: inputs_tensor.append(None)
            
        # 2. NPS 运行
        try:
            op = op_cls(inputs=[], outputs=[], dtype=out_dtype, **init_args)
            if op_name == "conv2d" or op_name == "gemm":
                nps_out = op.forward(inputs_tensor[0], inputs_tensor[1], inputs_tensor[2])["tensor"].data
            else:
                valid_tensors = [t for t in inputs_tensor if t is not None]
                nps_out = op.forward(*valid_tensors)["tensor"].data
        except Exception as e:
            print(f"  ❌ Iter {i} Crash: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # 3. CUDA 参数打包
        params_bin = None
        if op_name == "conv2d":
            x, w = inputs_np[0], inputs_np[1]
            pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
            oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                      oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "max_pool":
            x = inputs_np[0]
            k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
            oh = (x.shape[2] + pads[0] + pads[2] - k[0])//s[0] + 1
            ow = (x.shape[3] + pads[1] + pads[3] - k[1])//s[1] + 1
            p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow, k[0], k[1], pads[0], pads[1], s[0], s[1]]
            params_bin = np.array(p_list, dtype=np.int32).tobytes()
        elif op_name == "gemm":
            a, b, c = inputs_np[0], inputs_np[1], inputs_np[2]
            tA, tB = init_args['transA'], init_args['transB']
            M = a.shape[0] if tA==0 else a.shape[1]
            K = a.shape[1] if tA==0 else a.shape[0]
            N = b.shape[1] if tB==0 else b.shape[0]
            has_c = 1 if c is not None else 0
            c_type = 0
            if has_c:
                if c.size == 1: c_type=1
                elif c.ndim==1 or (c.ndim==2 and c.shape[0]==1): c_type=2
                elif c.ndim==2 and c.shape[1]==1: c_type=3
                else: c_type=4
            ints = np.array([M, N, K, tA, tB, c_type, has_c], dtype=np.int32).tobytes()
            floats = np.array([init_args['alpha'], init_args['beta']], dtype=np.float32).tobytes()
            params_bin = ints + floats
        elif op_name == "softmax":
            x, axis = inputs_np[0], init_args['axis']
            if axis < 0: axis += x.ndim
            inner, outer = x.shape[axis], int(np.prod(x.shape[:axis]))
            rem = int(np.prod(x.shape[axis+1:]))
            params_bin = np.array([outer, inner, rem], dtype=np.int32).tobytes()

        # 4. 数据转换与 广播处理
        expected_shape = nps_out.shape
        is_complex_kernel = op_name in ["conv2d", "max_pool", "gemm", "softmax"] # 这些算子自己处理形状
        is_double_kernel = is_complex_kernel or op_name in ["quantize_linear", "dequantize_linear"]
        
        cuda_inputs = []
        for i, (inp, d) in enumerate(zip(inputs_np, dtypes)):
            if inp is None:
                cuda_inputs.append(None)
            else:
                target_dtype = np.float64 if is_double_kernel else np.float32
                val_f32 = to_float32(inp, d)
                
                # 广播逻辑
                if not is_complex_kernel:
                    try:
                        if val_f32.shape != expected_shape:
                            val_f32 = np.broadcast_to(val_f32, expected_shape)
                    except Exception as e:
                        print(f"Warning: Broadcast failed for input {i} in {op_name}: {e}")
                
                cuda_inputs.append(val_f32.astype(target_dtype))
        
        # 5. 执行 CUDA
        cuda_out = run_cuda_ground_truth(
            op_name, 
            cuda_inputs, 
            params_binary=params_bin, 
            output_dtype=np.float64 if is_double_kernel else np.float32,
            target_shape=expected_shape
        ) 
            
        if cuda_out is None: continue
        
        # 6. 对比
        nps_f32 = to_float32(nps_out, out_dtype)
        is_ok, max_abs, max_rel, fail_mask = check_accuracy(nps_f32, cuda_out, atol, rtol, out_dtype)
        
        if max_abs >= 0:
            stats_abs.append(max_abs)
            stats_rel.append(max_rel)
        if is_ok:
            pass_cnt += 1
        else:
            print(f"  ❌ Iter {i} FAILED")
            if max_abs == -999.0: print(f"     Failed due to Overflow/Inf Logic Mismatch")
            elif max_abs == -1.0: print(f"     Failed due to NaN/Inf Mismatch")
            else:
                print(f"     Max Abs Diff: {max_abs:.6f} (Limit: {atol})")
                print(f"     Max Rel Diff: {max_rel:.6f} (Limit: {rtol})")
            
            if fail_mask is not None and np.any(fail_mask):
                idx_flat = np.argmax(fail_mask)
                idx = np.unravel_index(idx_flat, fail_mask.shape)
                print(f"     🔍 Debug Sample at {idx}:")
                print(f"        GT (CUDA) = {cuda_out[idx]}")
                print(f"        NPS (C)   = {nps_f32[idx]}")
                # 显示原始输入值
                for k, inp_arr in enumerate(inputs_np):
                    val_disp = ""
                    if inp_arr is None: val_disp = "None"
                    else:
                        try:
                            if not is_complex_kernel:
                                val_disp = np.broadcast_to(inp_arr, expected_shape)[idx]
                            else:
                                if inp_arr.shape == expected_shape:
                                    val_disp = inp_arr[idx]
                                else:
                                    val_disp = f"Shape{inp_arr.shape} (No direct mapping)"
                        except Exception as e:
                            val_disp = f"Error: {e}"
                            
                    print(f"        Input {k}   = {val_disp}")
            break

    if pass_cnt == iterations:
        print(f"  ✅ Pass ({pass_cnt}/{iterations})\n")
    else:
        print(f"  ⚠️  Fail\n")
    return stats_abs, stats_rel

# =============================================================================
# 3. 测试计划
# =============================================================================
if __name__ == "__main__":
    plans = [
        (ADD, "add", [(64,64), (64,64)], ["float32", "float32"], "float32"),
        (SUB, "sub", [(64,64), (64,64)], ["float16", "float16"], "float16"),
        (MUL, "mul", [(64,64), (64,64)], ["bfloat16", "bfloat16"], "bfloat16"),
        (DIV, "div", [(64,64), (64,64)], ["float32", "float32"], "float32"),
        (DIV, "div", [(64,64), (64,64)], ["float16", "float32"], "float16"),
        
        # Int8 GEMM 模拟: Int8 * Int8 -> Int32 (防止溢出)
        (MUL, "mul", [(64,64), (64,64)], ["int8", "int8"], "int32"),
        # Int8 累加: Int8 + Int32 -> Int32
        (ADD, "add", [(64,64), (64,64)], ["int8", "int32"], "int32"),
        # 极限 Int4: Int4 * Int4 -> Int16
        (MUL, "mul", [(64,64), (64,64)], ["int4", "int4"], "int16"),
        # A32W4 场景: FP32 + Int4 -> FP32
        (MUL, "mul", [(64,64), (64,64)], ["float32", "int4"], "float32"),
        (ADD, "add", [(64,64), (64,64)], ["float32", "int4"], "float32"),
        # FP16 + INT8 -> FP16
        (MUL, "mul", [(64,64), (64,64)], ["float16", "int8"], "float16"),
        (ADD, "add", [(64,64), (64,64)], ["float16", "int8"], "float16"),
        # FP32 + INT8 -> FP32
        (MUL, "mul", [(64,64), (64,64)], ["float32", "int8"], "float32"),
        (ADD, "add", [(64,64), (64,64)], ["float32", "int8"], "float32"),
        # 混合精度累加: FP16 + FP32 -> FP32 (ResNet/Transformer 常见)
        (ADD, "add", [(64,64), (64,64)], ["float16", "float32"], "float32"),
        # BF16 混合: BF16 * FP32 -> FP32
        (MUL, "mul", [(64,64), (64,64)], ["bfloat16", "float32"], "float32"),
        # 降级转换测试: FP32 / FP16 -> FP16
        (DIV, "div", [(64,64), (64,64)], ["float32", "float16"], "float16"),
        # E4M3 (权重) * E4M3 (激活) -> FP16
        (MUL, "mul", [(64,64), (64,64)], ["float8_e4m3", "float8_e4m3"], "float16"),
        # E5M2 (梯度) + FP16 -> FP16
        (ADD, "add", [(64,64), (64,64)], ["float8_e5m2", "float16"], "float16"),
        # 混合 FP8: E4M3 * E5M2 -> FP32
        (MUL, "mul", [(64,64), (64,64)], ["float8_e4m3", "float8_e5m2"], "float32"),
        
        (MUL, "mul", [(64,64), (64,64)], ["bfloat16", "bfloat16"], "bfloat16"),
        (ADD, "add", [(64,64), (64,64)], ["float8_e4m3", "float16"], "float16"),
        (DIV, "div", [(10, 10, 10), (10, 1)], ["float32", "float32"], "float32"),
        (SUB, "sub", [(4, 1, 16), (16,)], ["float32", "float32"], "float32"),
        
        (ABS, "abs", [(100,)], ["float8_e4m3"], "float8_e4m3"),
        (COS, "cos", [(100,)], ["float32"], "float32"),
        (COS, "cos", [(100,)], ["float16"], "float16"),
        (RELU, "relu", [(100,100)], ["float32"], "float32"),
        (RELU, "relu", [(100,100)], ["float16"], "float16"),
        (RELU, "relu", [(100,100)], ["int8"], "int8"),
        
        # --- QDQ 测试 ---
        # QuantizeLinear: FP32(Data) + FP32(Scale) + FP32(ZP) -> INT8
        # 测试 1: 标量 Scale/ZP 广播到张量
        (QuantizeLinear, "quantize_linear", 
         [(64, 64), (1,), (1,)], 
         ["float32", "float32", "float32"], "int8"),
         
        # 测试 2: Per-Channel 量化 (Scale/ZP 是向量)
        (QuantizeLinear, "quantize_linear", 
         [(2, 16, 4, 4), (1, 16, 1, 1), (1, 16, 1, 1)], 
         ["float32", "float32", "float32"], "int8"),

        # DequantizeLinear: INT8(Data) + FP32(Scale) + FP32(ZP) -> FP32
        (DequantizeLinear, "dequantize_linear", 
         [(64, 64), (1,), (1,)], 
         ["int8", "float32", "float32"], "float32"),
        
        (Conv, "conv2d", 
         [(1,1,5,5), (1,1,3,3), (1,)], 
         ["float32", "float32", "float32"], "float32",
         {"pads":[0,0,0,0], "strides":[1,1], "dilations":[1,1], "group":1}),
         
        # MaxPool: X(1,1,4,4) k=2 s=2
        (MaxPool, "max_pool",
         [(1,1,4,4)], ["float32"], "float32",
         {"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),
         
        # Gemm: A(2,3) B(3,4) C(4)
        (Gemm, "gemm",
         [(2,3), (3,4), (4,)], ["float32", "float32", "float32"], "float32",
         {"alpha":1.0, "beta":1.0, "transA":0, "transB":0}),
         
        # Softmax: X(2,5) axis=1
        (Softmax, "softmax",
         [(2,5)], ["float32"], "float32", {"axis":1}),
        
        # 1. [FP16 推理]: Half 输入/权重 + Float 偏置 -> Float 输出
        # 这是 TensorRT/ONNXRuntime 中最常见的混合精度模式
        (Conv, "conv2d",
         [(1, 2, 7, 7), (4, 2, 3, 3), (4,)], 
         ["float16", "float16", "float32"], "float32",
         {"pads":[1,1,1,1], "strides":[1,1], "dilations":[1,1], "group":1}),
         
        # 2. [FP8 极限推理]: E4M3 输入/权重 + FP16 偏置 -> FP16 输出
        # 模拟 NVIDIA H100 上的 FP8 卷积
        (Conv, "conv2d",
         [(1, 4, 8, 8), (8, 4, 3, 3), (8,)], 
         ["float8_e4m3", "float8_e4m3", "float16"], "float16",
         {"pads":[0,0,0,0], "strides":[2,2], "dilations":[1,1], "group":1}),

        # 3. [BF16 Depthwise]: BFloat16 深度卷积 (Group = InChannel)
        # 验证 Group 卷积逻辑与 BF16 的结合
        (Conv, "conv2d",
         [(1, 8, 6, 6), (8, 1, 3, 3), None], # 无 Bias
         ["bfloat16", "bfloat16", "float32"], "bfloat16",
         {"pads":[1,1,1,1], "strides":[1,1], "dilations":[1,1], "group":8}),

        # --- Gemm 混合精度场景 ---

        # 4. [Tensor Core 模式]: FP16 * FP16 + FP32 -> FP32
        # 典型的混合精度累加测试
        (Gemm, "gemm",
         [(16, 32), (32, 8), (8,)],
         ["float16", "float16", "float32"], "float32",
         {"alpha":1.0, "beta":1.0, "transA":0, "transB":0}),

        # 5. [FP8 混合]: E5M2 (高动态范围) * E4M3 (高精度) -> BF16
        # 测试 TransB=1 (矩阵转置) 和非对称 FP8 类型
        (Gemm, "gemm",
         [(8, 16), (8, 16), None],
         ["float8_e5m2", "float8_e4m3", "float32"], "bfloat16",
         {"alpha":0.5, "beta":0.0, "transA":0, "transB":1}),

        # --- 其他算子低精度测试 ---

        # 6. [BF16 MaxPool]
        (MaxPool, "max_pool",
         [(1, 2, 16, 16)], ["bfloat16"], "bfloat16",
         {"kernel_shape":[2,2], "pads":[0,0,0,0], "strides":[2,2]}),

        # 7. [FP16 Softmax]
        # 验证在低精度下 exp/sum 的数值稳定性
        (Softmax, "softmax",
         [(4, 64)], ["float16"], "float16", {"axis":-1}),
         
        # 8. [FP8 Softmax] (E4M3)
        # 极低精度下的 Softmax，用于验证饱和截断逻辑
        (Softmax, "softmax",
         [(2, 10)], ["float8_e4m3"], "float8_e4m3", {"axis":1}),
    ]

    print("🚀 开始数值验证 ...")
    ops_stats = {}
    for plan in plans:
        if len(plan) == 5:
            op_cls, op_name, shapes, dtypes, out_dtype = plan
            init_args = {}
        elif len(plan) == 6:
            op_cls, op_name, shapes, dtypes, out_dtype, init_args = plan
        else:
            print(f"⚠️ 跳过格式错误的测试计划: {plan}")
            continue
        abs_errs, rel_errs = verify_op(op_cls, op_name, shapes, dtypes, out_dtype, init_args=init_args, iterations=200)
        # 按算子名称聚合数据
        if op_name not in ops_stats:
            ops_stats[op_name] = {'abs': [], 'rel': []}
        ops_stats[op_name]['abs'].extend(abs_errs)
        ops_stats[op_name]['rel'].extend(rel_errs)
    print("\n📊 正在按算子绘制误差分布直方图...")
    for op_name, stats in ops_stats.items():
        valid_abs = [x for x in stats['abs'] if x >= 0]
        valid_rel = [x for x in stats['rel'] if x >= 0]
        if len(stats['abs']) == 0:
            print(f"⚠️ [{op_name.upper()}] 没有收集到有效误差数据 (可能全为逻辑匹配)")
            continue     
        plt.figure(figsize=(14, 6)) 
        # --- 子图 1: 绝对误差分布 ---
        plt.subplot(1, 2, 1)
        plt.hist(stats['abs'], bins=50, color='skyblue', edgecolor='black', log=True)
        plt.title(f'Operator [{op_name.upper()}] - Absolute Error Dist')
        plt.xlabel('Max Absolute Error')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2) 
        # 标注 99% 分位数 (P99)
        if len(stats['abs']) > 0:
            p99_abs = np.percentile(stats['abs'], 99)
            plt.axvline(p99_abs, color='red', linestyle='dashed', linewidth=1)
            plt.text(p99_abs, plt.ylim()[1]*0.9, f' P99: {p99_abs:.2e}', color='red')
        # --- 子图 2: 相对误差分布 ---
        plt.subplot(1, 2, 2)
        plt.hist(stats['rel'], bins=50, color='salmon', edgecolor='black', log=True)
        plt.title(f'Operator [{op_name.upper()}] - Relative Error Dist')
        plt.xlabel('Max Relative Error')
        plt.ylabel('Count (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        # 标注 99% 分位数 (P99)
        if len(stats['rel']) > 0:
            p99_rel = np.percentile(stats['rel'], 99)
            plt.axvline(p99_rel, color='red', linestyle='dashed', linewidth=1)
            plt.text(p99_rel, plt.ylim()[1]*0.9, f' P99: {p99_rel:.2e}', color='red')
        
        # # 保存图片
        # filename = f'error_dist_{op_name}.png'
        # plt.tight_layout()
        # plt.savefig(filename)
        # plt.close() # 关闭画布释放内存
        # plt.show()
        # print(f"✅ [{op_name.upper()}] 图表已保存至: {filename}")
    print("\n📈 详细统计报告 (99th Percentile Summary):")
    print(f"{'Operator':<10} | {'Abs (99%)':<12} | {'Rel (99%)':<12} | {'Samples':<8}")
    print("-" * 50)
    for op_name, stats in ops_stats.items():
        if len(stats['abs']) > 0:
            p99_abs = np.percentile(stats['abs'], 99)
            p99_rel = np.percentile(stats['rel'], 99)
            count = len(stats['abs'])
            print(f"{op_name.upper():<10} | {p99_abs:.2e}     | {p99_rel:.2e}     | {count:<8}")
    print("-" * 50)