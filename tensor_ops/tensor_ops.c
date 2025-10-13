// tensor_ops/tensor_ops.c
#include "tensor_ops.h"
#include <stdlib.h>

#define COS_LUT_SIZE 4096
#define COS_LUT_BITS 12
static double cos_lut[COS_LUT_SIZE + 1];
static int cos_lut_initialized = 0;
#define PI 3.141592653589793238462643383279502884197
#define TWO_PI (2.0 * PI)
#define HALF_PI (PI / 2.0)


static inline uint16_t float_to_float16(float value) {
    uint32_t bits = *(uint32_t*)&value;
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (bits >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | frac;
}

static inline float float16_to_float(uint16_t value) {
    uint32_t sign = (value & 0x8000) << 16;
    uint32_t exp = (value >> 10) & 0x1F;
    uint32_t frac = value & 0x3FF;
    if (exp == 0) return *(float*)&sign;
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (frac << 13);
        return *(float*)&bits;
    }
    exp = exp - 15 + 127;
    uint32_t bits = sign | (exp << 23) | (frac << 13);
    return *(float*)&bits;
}

static inline uint16_t float_to_bfloat16(float value) {
    uint32_t bits = *(uint32_t*)&value;
    return (uint16_t)(bits >> 16);
}

static inline float bfloat16_to_float(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    return *(float*)&bits;
}

// Initialize cosine lookup table
void init_cos_lut(void) {
    if (cos_lut_initialized) return;
    for (int i = 0; i <= COS_LUT_SIZE; i++) {
        double x = (double)i * TWO_PI / COS_LUT_SIZE;
        double sign = 1.0;
        if (x > PI) {
            x = TWO_PI - x;
        }
        if (x > HALF_PI) {
            x = PI - x;
            sign = -1.0;
        }
        double x2 = x * x;
        double result;

        if (x < 0.785398163397448) {
            result = 1.0 + x2 * (-0.5 + x2 * (0.04166666666666666 +
            x2 * (-0.001388888888888889 + x2 * 0.000024801587301587302)));
        } else {
            // Use sin for [π/4, π/2] since cos(x) = sin(π/2 - x)
            double t = HALF_PI - x;
            double t2 = t * t;
            result = t * (1.0 + t2 * (-0.16666666666666666 +
            t2 * (0.008333333333333333 + t2 * (-0.0001984126984126984 +
            t2 * 0.0000027557319223985893))));
            }
        cos_lut[i] = sign * result;
        }
    cos_lut_initialized = 1;
    }

static double cos_lut_lookup(double x) {
    if (!cos_lut_initialized) {
        init_cos_lut();
        }

    double reduced = x;
    if (reduced < 0) {
        reduced = -reduced;
        }

    if (reduced > TWO_PI) {
        int n = (int)(reduced / TWO_PI);
        reduced -= n * TWO_PI;
        }

    double idx_f = reduced * COS_LUT_SIZE / TWO_PI;
    int idx = (int)idx_f;
    double frac = idx_f - idx;
    if (idx >= COS_LUT_SIZE) {
        idx = COS_LUT_SIZE - 1;
        frac = 0.0;
        }

    return cos_lut[idx] * (1.0 - frac) + cos_lut[idx + 1] * frac;
    }

// Create tensor
Tensor* create_tensor(int* shape, int ndim, DataType dtype) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    tensor->dtype = dtype;
    // Calculate total size
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
// Allocate data based on dtype
    size_t elem_size = 0;
    switch (dtype) {
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            elem_size = 2;
            break;
        case DTYPE_FLOAT32:
            elem_size = 4;
            break;
        case DTYPE_FLOAT64:
            elem_size = 8;
            break;
        case DTYPE_INT8:
            elem_size = 1;
            break;
        case DTYPE_INT16:
            elem_size = 2;
            break;
        case DTYPE_INT32:
            elem_size = 4;
            break;
        case DTYPE_INT64:
            elem_size = 8;
            break;
    }
    tensor->data = malloc(tensor->size * elem_size);
    return tensor;
    }


// Free tensor
void free_tensor(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
        }
    }


// RELU implementation
void relu_forward(const Tensor* input, Tensor* output) {
    for (size_t i = 0; i < input->size; i++) {
        switch (input->dtype) {
            case DTYPE_FLOAT32: {
                float val = ((float*)input->data)[i];
                ((float*)output->data)[i] = val > 0.0f ? val : 0.0f;
                break;
                }
            case DTYPE_FLOAT64: {
                double val = ((double*)input->data)[i];
                ((double*)output->data)[i] = val > 0.0 ? val : 0.0;
                break;
                }
            case DTYPE_FLOAT16: {
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = float16_to_float(val);
                ((uint16_t*)output->data)[i] = fval > 0.0f ? val : 0;
                break;
                }
            case DTYPE_BFLOAT16: {
                uint16_t val = ((uint16_t*)input->data)[i];
                float fval = bfloat16_to_float(val);
                ((uint16_t*)output->data)[i] = fval > 0.0f ? val : 0;
                break;
                }
            case DTYPE_INT8: {
                int8_t val = ((int8_t*)input->data)[i];
                ((int8_t*)output->data)[i] = val > 0 ? val : 0;
                break;
                }
            case DTYPE_INT16: {
                int16_t val = ((int16_t*)input->data)[i];
                ((int16_t*)output->data)[i] = val > 0 ? val : 0;
                break;
                }
            case DTYPE_INT32: {
                int32_t val = ((int32_t*)input->data)[i];
                ((int32_t*)output->data)[i] = val > 0 ? val : 0;
                break;
                }
            case DTYPE_INT64: {
                int64_t val = ((int64_t*)input->data)[i];
                ((int64_t*)output->data)[i] = val > 0 ? val : 0;
                break;
                }
            }
        }
    }



