/**
  ******************************************************************************
  * @file        tensor_ops_dynamic_quant.c
  * @author      Egor Izmaylov
  * @brief       实现动态量化类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从 matrix/quant shard 拆分 DynamicQuantizeLinear。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// DynamicQuantizeLinear
// 仅支持映射到 uint8 ([0, 255])
// 实现 `dynamic quantize linear` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dynamic_quantize_linear_forward(const Tensor* x, Tensor* y, Tensor* y_scale, Tensor* y_zp) {
    if (!x || !y || !y_scale || !y_zp) return;
    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;

    for (size_t i = 0; i < x->size; i++) {
        double val = get_value_as_double(x, i);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    min_val = fmin(min_val, 0.0);
    max_val = fmax(max_val, 0.0);

    // 计算 Scale 和 ZeroPoint
    // Q_max = 255, Q_min = 0
    double scale = (max_val - min_val) / 255.0;
    if (scale == 0.0) scale = 1.0; // 避免除以 0

    double zp_double = 0.0 - min_val / scale;
    // Saturate ZP to [0, 255]
    zp_double = round(zp_double);
    if (zp_double < 0.0) zp_double = 0.0;
    if (zp_double > 255.0) zp_double = 255.0;
    uint8_t zp = (uint8_t)zp_double;

    // 写入参数输出
    set_tensor_value_from_float(y_scale, 0, scale);
    // 直接写入 uint8 原始数据到 scalar tensor
    // 假设 y_zp 是 uint8 类型
    if (y_zp->dtype == DTYPE_UINT8) {
        ((uint8_t*)y_zp->data)[0] = zp;
    } else {
        set_tensor_value_from_float(y_zp, 0, (double)zp);
    }

    // 执行量化
    // y = saturate(round(x / scale) + zp)
    _Pragma("omp parallel for")
    for (size_t i = 0; i < x->size; i++) {
        double val = get_value_as_double(x, i);
        double q_val = rint(val / scale) + (double)zp;

        // Saturate to uint8
        if (q_val < 0.0) q_val = 0.0;
        if (q_val > 255.0) q_val = 255.0;

        // 写入
        // set_tensor_value 会根据 y 的类型 (uint8) 自动转换
        set_tensor_value_from_float(y, i, q_val);
    }
}
