"""文件功能：封装数值验证中 dtype 范围、低精度位模式和 float32 解码逻辑。
作者：Egor Izmaylov
时间：2026-06-02
"""

import numpy as np


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
