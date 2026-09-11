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
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (size_t i = 0; i < x->size; i++) {
        float val = get_value_as_float(x, i);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    min_val = fminf(min_val, 0.0f);
    max_val = fmaxf(max_val, 0.0f);

    // 计算 Scale 和 ZeroPoint
    // Q_max = 255, Q_min = 0
    // ONNX 将 scale 作为 float32 输出；zero point 和量化数据必须使用同一个
    // 已物化的 float32 值，避免不可见的 double 精度改变舍入结果。
    float range_f32 = max_val - min_val;
    float scale_f32 = range_f32 / 255.0f;
    if (scale_f32 == 0.0f) scale_f32 = 1.0f; // 避免除以 0

    float zp_float = 0.0f - min_val / scale_f32;
    // Saturate ZP to [0, 255]
    // ONNX uses round-to-nearest, ties-to-even for the zero point.
    zp_float = nearbyintf(zp_float);
    if (zp_float < 0.0f) zp_float = 0.0f;
    if (zp_float > 255.0f) zp_float = 255.0f;
    uint8_t zp = (uint8_t)zp_float;

    // 写入参数输出
    set_tensor_value_from_float(y_scale, 0, (double)scale_f32);
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
        float val = get_value_as_float(x, i);
        float quotient = val / scale_f32;
        float q_val = rintf(quotient) + (float)zp;

        // Saturate to uint8
        if (q_val < 0.0f) q_val = 0.0f;
        if (q_val > 255.0f) q_val = 255.0f;

        // 写入
        // set_tensor_value 会根据 y 的类型 (uint8) 自动转换
        set_tensor_value_from_float(y, i, q_val);
    }
}
