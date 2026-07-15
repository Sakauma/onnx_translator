/**
  ******************************************************************************
  * @file        random.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_RANDOM_H
#define TENSOR_OPS_INTERNAL_RANDOM_H

#include "../tensor_ops_internal.h"

static uint32_t simple_lcg(uint32_t* state);

// 实现 `simple_lcg` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static uint32_t simple_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

#endif /* TENSOR_OPS_INTERNAL_RANDOM_H */
