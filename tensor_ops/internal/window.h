/**
  ******************************************************************************
  * @file        window.h
  * @author      Egor Izmaylov
  * @brief       提供 Hann、Hamming 和 Blackman 窗共享的标量长度读取逻辑。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_window.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_WINDOW_H
#define TENSOR_OPS_INTERNAL_WINDOW_H

#include "../tensor_ops_internal.h"

/* Window 算子的 size 输入是单元素整数 Tensor；空输入按零长度处理。 */

// 获取窗函数大小
// 封装 `get_window_size` 的 Tensor ABI 读写或复制逻辑，统一 Python ctypes 与 C 后端的数据解释方式。
static int64_t get_window_size(const Tensor* size_tensor) {
    if (!size_tensor) return 0;
    return get_value_as_int64(size_tensor, 0);
}

#endif /* TENSOR_OPS_INTERNAL_WINDOW_H */
