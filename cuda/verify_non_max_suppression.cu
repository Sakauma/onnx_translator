/**
  ******************************************************************************
  * @file        verify_non_max_suppression.cu
  * @author      Egor Izmaylov
  * @brief       提供 NonMaxSuppression 算子的 CUDA 参考验证程序。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define VERIFY_NMS_MAX_BOXES 256

struct NmsParams {
    int32_t batch_count;
    int32_t num_boxes;
    int32_t class_count;
    int32_t max_output_boxes_per_class;
    int32_t center_point_box;
    float iou_threshold;
    float score_threshold;
};

// 将 box 转成角点格式，复刻 C 后端对 corner/center 两种输入格式的解释。
__device__ void nms_box_corners(
    const double* boxes,
    int batch,
    int box_idx,
    NmsParams params,
    double* y1,
    double* x1,
    double* y2,
    double* x2
) {
    size_t base = ((size_t)batch * (size_t)params.num_boxes + (size_t)box_idx) * 4u;
    double a = boxes[base + 0];
    double b = boxes[base + 1];
    double c = boxes[base + 2];
    double d = boxes[base + 3];

    if (params.center_point_box) {
        double x_center = a;
        double y_center = b;
        double width = c;
        double height = d;
        *y1 = y_center - height / 2.0;
        *x1 = x_center - width / 2.0;
        *y2 = y_center + height / 2.0;
        *x2 = x_center + width / 2.0;
    } else {
        *y1 = a;
        *x1 = b;
        *y2 = c;
        *x2 = d;
    }

    if (*y1 > *y2) {
        double tmp = *y1;
        *y1 = *y2;
        *y2 = tmp;
    }
    if (*x1 > *x2) {
        double tmp = *x1;
        *x1 = *x2;
        *x2 = tmp;
    }
}

// 计算两个候选框的 IoU。
__device__ double nms_iou(const double* boxes, int batch, int lhs, int rhs, NmsParams params) {
    double ay1, ax1, ay2, ax2;
    double by1, bx1, by2, bx2;
    nms_box_corners(boxes, batch, lhs, params, &ay1, &ax1, &ay2, &ax2);
    nms_box_corners(boxes, batch, rhs, params, &by1, &bx1, &by2, &bx2);

    double inter_h = fmax(0.0, fmin(ay2, by2) - fmax(ay1, by1));
    double inter_w = fmax(0.0, fmin(ax2, bx2) - fmax(ax1, bx1));
    double inter = inter_h * inter_w;
    double area_a = fmax(0.0, ay2 - ay1) * fmax(0.0, ax2 - ax1);
    double area_b = fmax(0.0, by2 - by1) * fmax(0.0, bx2 - bx1);
    double union_area = area_a + area_b - inter;
    return union_area <= 0.0 ? 0.0 : inter / union_area;
}

// 在单个 CUDA 线程中执行 NMS reference，保持排序稳定性和输出顺序与 C 后端一致。
__global__ void nms_kernel(const double* boxes, const double* scores, int64_t* output, int32_t* out_rows, NmsParams params) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (params.num_boxes > VERIFY_NMS_MAX_BOXES) {
        *out_rows = -1;
        return;
    }

    int candidates[VERIFY_NMS_MAX_BOXES];
    int kept[VERIFY_NMS_MAX_BOXES];
    int rows = 0;

    for (int b = 0; b < params.batch_count; ++b) {
        for (int cls = 0; cls < params.class_count; ++cls) {
            int candidate_count = 0;
            for (int box = 0; box < params.num_boxes; ++box) {
                size_t score_idx = ((size_t)b * (size_t)params.class_count + (size_t)cls) * (size_t)params.num_boxes + (size_t)box;
                double score = scores[score_idx];
                if (score >= (double)params.score_threshold) {
                    candidates[candidate_count++] = box;
                }
            }

            for (int i = 1; i < candidate_count; ++i) {
                int current = candidates[i];
                size_t current_idx = ((size_t)b * (size_t)params.class_count + (size_t)cls) * (size_t)params.num_boxes + (size_t)current;
                double current_score = scores[current_idx];
                int j = i - 1;
                while (j >= 0) {
                    int prev = candidates[j];
                    size_t prev_idx = ((size_t)b * (size_t)params.class_count + (size_t)cls) * (size_t)params.num_boxes + (size_t)prev;
                    double prev_score = scores[prev_idx];
                    if (prev_score >= current_score) break;
                    candidates[j + 1] = candidates[j];
                    --j;
                }
                candidates[j + 1] = current;
            }

            int kept_count = 0;
            for (int i = 0; i < candidate_count && kept_count < params.max_output_boxes_per_class; ++i) {
                int candidate = candidates[i];
                int suppress = 0;
                for (int k = 0; k < kept_count; ++k) {
                    if (nms_iou(boxes, b, candidate, kept[k], params) > (double)params.iou_threshold) {
                        suppress = 1;
                        break;
                    }
                }
                if (!suppress) {
                    kept[kept_count++] = candidate;
                    output[(size_t)rows * 3u + 0u] = (int64_t)b;
                    output[(size_t)rows * 3u + 1u] = (int64_t)cls;
                    output[(size_t)rows * 3u + 2u] = (int64_t)candidate;
                    ++rows;
                }
            }
        }
    }

    *out_rows = rows;
}

static int read_params(const char* path, NmsParams* params) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    int32_t ints[5];
    float floats[2];
    if (fread(ints, sizeof(int32_t), 5, fp) != 5) {
        fclose(fp);
        return 0;
    }
    if (fread(floats, sizeof(float), 2, fp) != 2) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    params->batch_count = ints[0];
    params->num_boxes = ints[1];
    params->class_count = ints[2];
    params->max_output_boxes_per_class = ints[3];
    params->center_point_box = ints[4];
    params->iou_threshold = floats[0];
    params->score_threshold = floats[1];
    return params->batch_count > 0 && params->num_boxes >= 0 && params->class_count > 0;
}

static int read_double_file(const char* path, double* data, size_t n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t got = fread(data, sizeof(double), n, fp);
    fclose(fp);
    return got == n;
}

static int write_i64_file(const char* path, const int64_t* data, size_t n) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    size_t wrote = fwrite(data, sizeof(int64_t), n, fp);
    fclose(fp);
    return wrote == n;
}

// 作为 CUDA 验证程序入口，执行 NMS reference 并写回 int64 selected_indices。
int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr, "Usage: %s <out_len> <boxes.bin> <scores.bin> <max.bin> <iou.bin> <score.bin> <params.bin> <out.bin>\n", argv[0]);
        return 1;
    }

    size_t out_len = (size_t)atoll(argv[1]);
    NmsParams params;
    if (!read_params(argv[7], &params)) return 1;
    size_t boxes_len = (size_t)params.batch_count * (size_t)params.num_boxes * 4u;
    size_t scores_len = (size_t)params.batch_count * (size_t)params.class_count * (size_t)params.num_boxes;
    int max_rows = params.batch_count * params.class_count * params.max_output_boxes_per_class;
    size_t max_out_len = (size_t)(max_rows > 0 ? max_rows : 1) * 3u;
    if (out_len % 3u != 0u || (int)(out_len / 3u) > max_rows) return 1;

    double* h_boxes = (double*)malloc(boxes_len * sizeof(double));
    double* h_scores = (double*)malloc(scores_len * sizeof(double));
    int64_t* h_output = (int64_t*)calloc(max_out_len, sizeof(int64_t));
    int32_t h_rows = 0;
    if (!h_boxes || !h_scores || !h_output) return 1;
    if (!read_double_file(argv[2], h_boxes, boxes_len)) return 1;
    if (!read_double_file(argv[3], h_scores, scores_len)) return 1;

    double* d_boxes = NULL;
    double* d_scores = NULL;
    int64_t* d_output = NULL;
    int32_t* d_rows = NULL;
    cudaMalloc(&d_boxes, boxes_len * sizeof(double));
    cudaMalloc(&d_scores, scores_len * sizeof(double));
    cudaMalloc(&d_output, max_out_len * sizeof(int64_t));
    cudaMalloc(&d_rows, sizeof(int32_t));
    cudaMemcpy(d_boxes, h_boxes, boxes_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, h_scores, scores_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, max_out_len * sizeof(int64_t));
    cudaMemset(d_rows, 0, sizeof(int32_t));

    nms_kernel<<<1, 1>>>(d_boxes, d_scores, d_output, d_rows, params);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_rows, d_rows, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (h_rows < 0 || (size_t)h_rows * 3u != out_len) {
        cudaFree(d_boxes);
        cudaFree(d_scores);
        cudaFree(d_output);
        cudaFree(d_rows);
        free(h_boxes);
        free(h_scores);
        free(h_output);
        return 1;
    }
    cudaMemcpy(h_output, d_output, (out_len == 0 ? 1u : out_len) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    int ok = write_i64_file(argv[8], h_output, out_len);

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_output);
    cudaFree(d_rows);
    free(h_boxes);
    free(h_scores);
    free(h_output);
    return ok ? 0 : 1;
}
