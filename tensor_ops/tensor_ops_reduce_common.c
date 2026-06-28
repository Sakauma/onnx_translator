/**
  ******************************************************************************
  * @file        tensor_ops_reduce_common.c
  * @author      Egor Izmaylov
  * @brief       实现归约类 C 后端共享坐标工具。
  * @details     2026.06.28  V1.0.0  从 reduce/arg shard 拆分公共归约坐标逻辑。
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 初始化归约坐标，将输出坐标映射回输入坐标，归约轴先置零。
void prepare_reduce_coords(
    size_t out_index,
    const Tensor* input,
    const Tensor* output,
    const ReduceParams* params,
    int* coords
) {
    int out_coords[MAX_NDIM];
    get_coords_from_index(out_index, out_coords, output->shape, output->ndim);

    if (params->keepdims) {
        for (int d = 0; d < input->ndim; d++) {
            coords[d] = is_axis_reduced(d, params->axes, params->num_axes) ? 0 : out_coords[d];
        }
        return;
    }

    int out_dim_idx = 0;
    for (int d = 0; d < input->ndim; d++) {
        if (is_axis_reduced(d, params->axes, params->num_axes)) {
            coords[d] = 0;
        } else {
            coords[d] = out_coords[out_dim_idx++];
        }
    }
}


// 计算归约轴组合数量，作为内层循环的展开空间。
size_t reduce_total_steps_for(const Tensor* input, const ReduceParams* params) {
    size_t reduce_total_steps = 1;
    for (int i = 0; i < params->num_axes; i++) {
        reduce_total_steps *= input->shape[params->axes[i]];
    }
    return reduce_total_steps;
}


// 根据归约空间线性索引更新当前输入坐标。
void update_reduce_coords(const Tensor* input, const ReduceParams* params, int* coords, size_t reduce_index) {
    size_t temp_r = reduce_index;
    for (int k = params->num_axes - 1; k >= 0; k--) {
        int axis_idx = params->axes[k];
        int dim_size = input->shape[axis_idx];
        coords[axis_idx] = temp_r % dim_size;
        temp_r /= dim_size;
    }
}
