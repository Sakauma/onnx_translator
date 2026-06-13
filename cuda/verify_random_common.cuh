/**
  ******************************************************************************
  * @file        verify_random_common.cuh
  * @author      Egor Izmaylov
  * @brief       提供随机类 CUDA verifier 共用的确定性伪随机辅助函数。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef VERIFY_RANDOM_COMMON_CUH
#define VERIFY_RANDOM_COMMON_CUH

#include <math.h>
#include <stdint.h>

#define VERIFY_RANDOM_TWO_PI 6.283185307179586476925286766559

// 生成与 C 后端 simple_lcg 一致的 31 bit 伪随机整数。
__device__ __forceinline__ uint32_t verify_random_lcg_next(uint32_t state) {
    return (state * 1103515245u + 12345u) & 0x7fffffffu;
}

// 根据元素下标派生随机状态，保证 CUDA reference 与 C 后端逐元素可复现。
__device__ __forceinline__ uint32_t verify_random_state_for_index(uint32_t seed, int index) {
    return seed ^ (uint32_t)index;
}

// 生成 [0, 1) 均匀随机数。
__device__ __forceinline__ double verify_random_uniform01(uint32_t seed, int index) {
    uint32_t state = verify_random_state_for_index(seed, index);
    uint32_t r = verify_random_lcg_next(state);
    return (double)r / 2147483648.0;
}

// 使用 Box-Muller 变换生成标准正态随机数。
__device__ __forceinline__ double verify_random_normal01(uint32_t seed, int index) {
    uint32_t state = verify_random_state_for_index(seed, index);
    uint32_t r1 = verify_random_lcg_next(state);
    uint32_t r2 = verify_random_lcg_next(r1);
    double u1 = ((double)r1 + 1.0) / 2147483649.0;
    double u2 = ((double)r2 + 1.0) / 2147483649.0;
    return sqrt(-2.0 * log(u1)) * cos(VERIFY_RANDOM_TWO_PI * u2);
}

#endif
