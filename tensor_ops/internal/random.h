/**
  ******************************************************************************
  * @file        random.h
  * @author      Egor Izmaylov
  * @brief       提供随机算子共用的确定性线性同余发生器。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_random.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_RANDOM_H
#define TENSOR_OPS_INTERNAL_RANDOM_H

#include "../tensor_ops_internal.h"

/* simple_lcg 用于数值验证可复现性，不提供密码学随机性，也不维护全局状态。 */

static uint32_t simple_lcg(uint32_t* state);

// 实现 `simple_lcg` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static uint32_t simple_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

#endif /* TENSOR_OPS_INTERNAL_RANDOM_H */
