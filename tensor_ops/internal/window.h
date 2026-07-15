/**
  ******************************************************************************
  * @file        window.h
  * @author      Egor Izmaylov
  * @brief       声明对应 C 算子分片独占的内部辅助逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_WINDOW_H
#define TENSOR_OPS_INTERNAL_WINDOW_H

#include "../tensor_ops_internal.h"

// 获取窗函数大小
// 封装 `get_window_size` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static int64_t get_window_size(const Tensor* size_tensor) {
    if (!size_tensor) return 0;
    return get_value_as_int64(size_tensor, 0);
}

#endif /* TENSOR_OPS_INTERNAL_WINDOW_H */
