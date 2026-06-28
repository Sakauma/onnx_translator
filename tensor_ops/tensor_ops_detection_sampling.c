/**
  ******************************************************************************
  * @file        tensor_ops_detection_sampling.c
  * @author      Egor Izmaylov
  * @brief       实现检测后处理和采样类 C 后端算子。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


// 实现 `non max suppression` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
int non_max_suppression_forward(const Tensor* boxes, const Tensor* scores, Tensor* output,
                                int max_output_boxes_per_class, float iou_threshold,
                                float score_threshold, int center_point_box) {
    if (!boxes || !scores || !output) return 0;
    if (boxes->ndim != 3 || scores->ndim != 3 || boxes->shape[2] != 4) return 0;
    int batch_count = boxes->shape[0];
    int num_boxes = boxes->shape[1];
    int class_count = scores->shape[1];
    if (scores->shape[0] != batch_count || scores->shape[2] != num_boxes || max_output_boxes_per_class <= 0) return 0;

    int* candidates = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    int* kept = (int*)malloc((num_boxes == 0 ? 1 : num_boxes) * sizeof(int));
    if (!candidates || !kept) {
        free(candidates);
        free(kept);
        return 0;
    }

    int out_rows = 0;
    for (int b = 0; b < batch_count; b++) {
        for (int cls = 0; cls < class_count; cls++) {
            int candidate_count = 0;
            for (int box = 0; box < num_boxes; box++) {
                size_t score_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)box;
                double score = get_value_as_double(scores, score_idx);
                if (score >= (double)score_threshold) {
                    candidates[candidate_count++] = box;
                }
            }

            for (int i = 1; i < candidate_count; i++) {
                int current = candidates[i];
                size_t current_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)current;
                double current_score = get_value_as_double(scores, current_idx);
                int j = i - 1;
                while (j >= 0) {
                    int prev = candidates[j];
                    size_t prev_idx = ((size_t)b * class_count + (size_t)cls) * num_boxes + (size_t)prev;
                    double prev_score = get_value_as_double(scores, prev_idx);
                    if (prev_score >= current_score) break;
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = current;
            }

            int kept_count = 0;
            for (int i = 0; i < candidate_count && kept_count < max_output_boxes_per_class; i++) {
                int candidate = candidates[i];
                int suppress = 0;
                for (int k = 0; k < kept_count; k++) {
                    if (nms_iou(boxes, b, candidate, kept[k], center_point_box) > (double)iou_threshold) {
                        suppress = 1;
                        break;
                    }
                }
                if (!suppress) {
                    kept[kept_count++] = candidate;
                    if ((size_t)(out_rows + 1) * 3 <= output->size) {
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 0, b);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 1, cls);
                        set_tensor_value_from_int(output, (size_t)out_rows * 3 + 2, candidate);
                    }
                    out_rows++;
                }
            }
        }
    }

    free(candidates);
    free(kept);
    return out_rows;
}


// 实现 `grid sample` 算子的 C 后端入口，校验张量缓冲区并按目标 dtype 写入计算结果。
void grid_sample_forward(const Tensor* input, const Tensor* grid, Tensor* output,
                         int mode, int padding_mode, int align_corners) {
    if (!input || !grid || !output) return;
    if (input->ndim != 4 || grid->ndim != 4 || output->ndim != 4 || grid->shape[3] != 2) return;
    int n_batches = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];
    int out_h = grid->shape[1];
    int out_w = grid->shape[2];
    if (grid->shape[0] != n_batches || output->shape[0] != n_batches || output->shape[1] != channels ||
        output->shape[2] != out_h || output->shape[3] != out_w) return;

    _Pragma("omp parallel for collapse(4)")
    for (int n = 0; n < n_batches; n++) {
        for (int c = 0; c < channels; c++) {
            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    size_t grid_idx = ((size_t)n * out_h * out_w * 2) + ((size_t)oy * out_w * 2) + ((size_t)ox * 2);
                    double x_norm = get_value_as_double(grid, grid_idx);
                    double y_norm = get_value_as_double(grid, grid_idx + 1);
                    double in_x = grid_denormalize(x_norm, width, align_corners);
                    double in_y = grid_denormalize(y_norm, height, align_corners);
                    double value;
                    if (mode == 1) {
                        double sy = nearbyint(grid_sample_coordinate(in_y, height, padding_mode, align_corners));
                        double sx = nearbyint(grid_sample_coordinate(in_x, width, padding_mode, align_corners));
                        value = grid_get_pixel_2d(input, n, c, sy, sx, padding_mode, align_corners);
                    } else if (mode == 2) {
                        value = grid_bicubic_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    } else {
                        value = grid_bilinear_sample_2d(input, n, c, in_y, in_x, padding_mode, align_corners);
                    }
                    size_t out_idx = ((size_t)n * channels * out_h * out_w)
                                   + ((size_t)c * out_h * out_w)
                                   + ((size_t)oy * out_w)
                                   + (size_t)ox;
                    set_tensor_value_from_float(output, out_idx, value);
                }
            }
        }
    }
}
