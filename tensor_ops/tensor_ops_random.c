/**
  ******************************************************************************
  * @file        tensor_ops_random.c
  * @author      Egor Izmaylov
  * @brief       实现随机和概率采样类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"

// 根据元素下标派生随机状态，保证随机算子的输出不受 OpenMP 线程数和调度顺序影响。
static uint32_t random_state_for_index(uint32_t base_seed, size_t index) {
    return base_seed ^ (uint32_t)index;
}


// 生成与 CUDA verifier 一致的 [0, 1) 均匀随机数。
static double random_uniform01_for_index(uint32_t base_seed, size_t index) {
    uint32_t state = random_state_for_index(base_seed, index);
    uint32_t r = simple_lcg(&state);
    return (double)r / 2147483648.0;
}


// 使用 Box-Muller 变换生成与 CUDA verifier 一致的标准正态随机数。
static double random_normal01_for_index(uint32_t base_seed, size_t index) {
    uint32_t state = random_state_for_index(base_seed, index);
    uint32_t r1 = simple_lcg(&state);
    uint32_t r2 = simple_lcg(&state);
    double u1 = ((double)r1 + 1.0) / 2147483649.0;
    double u2 = ((double)r2 + 1.0) / 2147483649.0;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

// 实现 `multinomial` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void multinomial_forward(const Tensor* input, Tensor* output, int sample_size, uint32_t seed) {
    if (!input || !output || input->ndim != 2 || output->ndim != 2 || sample_size < 0) return;
    int batch = input->shape[0];
    int classes = input->shape[1];
    if (output->shape[0] != batch || output->shape[1] != sample_size) return;

    for (int row = 0; row < batch; row++) {
        double total = 0.0;
        for (int c = 0; c < classes; c++) {
            double p = get_value_as_double(input, (size_t)row * classes + c);
            if (p > 0.0) total += p;
        }
        if (total <= 0.0) continue;

        uint32_t state = seed ? (seed + (uint32_t)row * 747796405u) : (uint32_t)time(NULL) + (uint32_t)row;
        for (int sample = 0; sample < sample_size; sample++) {
            uint32_t r = simple_lcg(&state);
            double threshold = ((double)r / 2147483648.0) * total;
            double cumulative = 0.0;
            int selected = classes - 1;
            for (int c = 0; c < classes; c++) {
                double p = get_value_as_double(input, (size_t)row * classes + c);
                if (p <= 0.0) continue;
                cumulative += p;
                if (threshold < cumulative) {
                    selected = c;
                    break;
                }
            }
            set_tensor_value_from_int(output, (size_t)row * sample_size + sample, selected);
        }
    }
}

// 实现 `random uniform like` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void random_uniform_like_forward(Tensor* output, float low, float high, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    double range = high - low;

    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double r_norm = random_uniform01_for_index(base_seed, i);
        double val = low + r_norm * range;
        set_tensor_value_from_float(output, i, val);
    }
}

// RandomNormal: Box-Muller 变换
// 实现 `random normal` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void random_normal_forward(Tensor* output, float mean, float scale, float seed) {
    if (!output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double z0 = random_normal01_for_index(base_seed, i);
        double val = (double)mean + z0 * (double)scale;
        set_tensor_value_from_float(output, i, val);
    }
}


// Bernoulli: 生成 0 或 1
// 实现 `bernoulli` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void bernoulli_forward(const Tensor* input, Tensor* output, float seed) {
    if (!input || !output) return;
    
    uint32_t base_seed = (uint32_t)seed;
    if (seed == 0.0f) base_seed = (uint32_t)time(NULL);
    
    #pragma omp parallel for
    for (size_t i = 0; i < output->size; i++) {
        double prob = get_value_as_double(input, i);
        double r_norm = random_uniform01_for_index(base_seed, i);
        double res = (r_norm < prob) ? 1.0 : 0.0;
        set_tensor_value_from_float(output, i, res);
    }
}


// Dropout (Inference Mode)
// 实现 `dropout` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void dropout_forward(const Tensor* input, Tensor* output, float ratio, int training_mode) {
    if (!input || !output) return;
    
    // 如果是推理模式(training_mode=0)，或者是比例为0，直接复制
    if (training_mode == 0 || ratio == 0.0f) {
        size_t elem_size = get_dtype_size(input->dtype);
        // 如果输入输出类型一致且大小一致
        if (input->dtype == output->dtype && input->size == output->size) {
            memcpy(output->data, input->data, input->size * elem_size);
        } else {
            // 类型转换复制
            cast_forward(input, output);
        }
        return;
    }
    
    // 训练模式下的 Dropout (简单的随机置0)
    // 标准 Dropout 还需要 scale (val / (1-ratio)) 以保持期望值
    double scale_factor = 1.0 / (1.0 - (double)ratio);
    uint32_t base_seed = (uint32_t)time(NULL);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t local_state = base_seed + tid;
        
        #pragma omp for
        for (size_t i = 0; i < input->size; i++) {
            uint32_t r = simple_lcg(&local_state);
            double r_norm = (double)r / 2147483648.0;
            
            double val = get_value_as_double(input, i);
            if (r_norm < ratio) {
                set_tensor_value_from_float(output, i, 0.0);
            } else {
                set_tensor_value_from_float(output, i, val * scale_factor);
            }
        }
    }
}
