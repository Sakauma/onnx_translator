/**
  ******************************************************************************
  * @file        trig.h
  * @author      Egor Izmaylov
  * @brief       提供 Cos 算子的线程安全余弦查找表及线性插值。
  * @details     2026.07.15  V1.0.0  从 tensor_ops_internal.h 拆分
  ******************************************************************************
  * @attention   仅供 tensor_ops_trig.c 使用，不属于公共 ABI。
  ******************************************************************************
*/

#ifndef TENSOR_OPS_INTERNAL_TRIG_H
#define TENSOR_OPS_INTERNAL_TRIG_H

#include "../tensor_ops_internal.h"
#include <pthread.h>

/* LUT 状态保持在 trig translation unit 内；首次初始化由互斥锁保护。 */

// 余弦查找表大小
#define COS_LUT_SIZE 4096
// 余弦查找表位数
#define COS_LUT_BITS 12
// 余弦查找表
static double cos_lut[COS_LUT_SIZE + 1];
// 余弦查找表初始化标志
static int cos_lut_initialized = 0;
// 余弦查找表初始化互斥锁
static pthread_mutex_t cos_lut_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * 初始化余弦查找表
 * 使用泰勒级数展开计算余弦值并存储在查找表中
 */
// 实现 `init_cos_lut` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。

/**
 * 使用查找表计算余弦值
 *
 * @param x 输入角度（弧度）
 * @return 余弦值
 */
// 实现 `cos_lut_lookup` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
static double cos_lut_lookup(double x) {
    // 如果查找表未初始化，则先初始化
    if (!cos_lut_initialized) {
        init_cos_lut();
    }
    // 处理负角度并归一化到[0, 2π]区间
    double reduced = fmod(fabs(x), TWO_PI);
    // 计算查找表索引和插值因子
    double idx_f = reduced * COS_LUT_SIZE / TWO_PI;
    int idx = (int)idx_f;
    double frac = idx_f - idx;
    // 边界处理
    if (idx >= COS_LUT_SIZE) {
        idx = COS_LUT_SIZE - 1;
        frac = 0.0;
    }
    // 线性插值计算余弦值
    return cos_lut[idx] * (1.0 - frac) + cos_lut[idx + 1] * frac;
}

#endif /* TENSOR_OPS_INTERNAL_TRIG_H */
