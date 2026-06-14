# /**
#   ******************************************************************************
#   * @file        data.py
#   * @author      Egor Izmaylov
#   * @brief       生成数值验证输入数据，并提供随机算子的确定性参考实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

import nn

from .dtype import float32_to_bfloat16_bits, from_float32


def generate_random_data(shape, dtype):
    size = np.prod(shape)
    
    # --- 整数生成 ---
    if "int" in dtype and "float" not in dtype:
        if dtype == "int4": return np.random.randint(-8, 8, shape).astype(np.int8)
        if dtype == "uint4": return np.random.randint(0, 16, shape).astype(np.uint8)
        if dtype == "int2": return np.random.randint(-2, 2, shape).astype(np.int8)
        if dtype == "uint2": return np.random.randint(0, 4, shape).astype(np.uint8)
        if dtype == "int8": return np.random.randint(-120, 120, shape).astype(np.int8)
        limit = 1000
        return np.random.randint(-limit, limit, shape).astype(nn.DTYPE_TO_NUMPY.get(dtype, np.int32))

    # --- 浮点位模式生成 (Float8) ---
    if dtype == "float4_e2m1":
        return from_float32(np.random.uniform(-6.0, 6.0, size=shape).astype(np.float32), dtype)
    if dtype == "float8_e8m0":
        exponents = np.random.randint(-8, 9, size=shape)
        return from_float32(np.ldexp(np.ones(shape, dtype=np.float32), exponents), dtype)
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
    if dtype == "bool":
        return (np.random.randint(0, 2, size=shape).astype(np.uint8)).astype(np.bool_)
        
    return raw_f32.astype(np.float32)

def random_uniform_like_reference(shape, low, high, seed):
    numel = int(np.prod(shape))
    out = np.empty(numel, dtype=np.float32)

    for i in range(numel):
        s = np.uint32(seed) ^ np.uint32(i)
        s = np.uint32((np.uint64(s) * 1664525 + 1013904223) & 0xFFFFFFFF)
        u = np.float32(int(s & np.uint32(0x00FFFFFF)) / 16777216.0)
        out[i] = np.float32(low + (high - low) * u)

    return out.reshape(shape).astype(np.float32)
