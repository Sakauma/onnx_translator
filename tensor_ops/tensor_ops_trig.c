/**
  ******************************************************************************
  * @file        tensor_ops_trig.c
  * @author      Egor Izmaylov
  * @brief       实现三角函数类 C 后端算子。
  * @details     2026.06.28  V1.0.0  从基础 elementwise shard 拆分 Cos LUT 和 Cos。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


/**
 * 初始化余弦查找表
 * 使用泰勒级数展开计算余弦值并存储在查找表中
 */
// 实现 `init_cos_lut` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
void init_cos_lut(void) {
    pthread_mutex_lock(&cos_lut_mutex);
    if (!cos_lut_initialized) {
        // 遍历查找表的每个位置
        for (int i = 0; i <= COS_LUT_SIZE; i++) {
            // 计算对应的角度值
            double x = (double)i * TWO_PI / COS_LUT_SIZE;
            double sign = 1.0;

            // 将角度映射到[0, π]区间
            if (x > PI) {
                x = TWO_PI - x;
            }
            // 将角度映射到[0, π/2]区间
            if (x > HALF_PI) {
                x = PI - x;
                sign = -1.0;
            }
            // 计算x的平方
            double x2 = x * x;
            double result;

            // 根据角度大小选择不同的计算方法
            if (x < 0.785398163397448) {
                // 使用余弦泰勒级数展开
                result = 1.0 + x2 * (-0.5 + x2 * (0.04166666666666666 +
                         x2 * (-0.001388888888888889 + x2 * 0.000024801587301587302)));
            } else {
                // 使用正弦泰勒级数展开，因为cos(x) = sin(π/2 - x)
                double t = HALF_PI - x;
                double t2 = t * t;
                result = t * (1.0 + t2 * (-0.16666666666666666 +
                         t2 * (0.008333333333333333 + t2 * (-0.0001984126984126984 +
                         t2 * 0.0000027557319223985893))));
            }
            // 存储带符号的计算结果
            cos_lut[i] = sign * result;
        }
        __sync_synchronize();
        // 标记查找表已初始化
        cos_lut_initialized = 1;
    }
    // 解锁
    pthread_mutex_unlock(&cos_lut_mutex);
}


/**
 * 余弦函数前向传播
 * * @param input 输入张量
 * @param output 输出张量
 */
// 实现 `cos` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void cos_forward(const Tensor* input, Tensor* output) {
    // 检查输入参数是否有效
    if (!input || !output || !input->data || !output->data || input->size != output->size) {
        return;
    }

    if (!cos_lut_initialized) init_cos_lut();

    #pragma omp parallel for
    for (size_t i = 0; i < input->size; i++) {
        double val = get_value_as_double(input, i); // 输入转 double
        double res = cos_lut_lookup(val);           // 查表
        set_tensor_value_from_float(output, i, res); // 安全写入输出
    }
}
