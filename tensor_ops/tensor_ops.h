#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdint.h>
#include <string.h>

typedef enum {
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64
    } DataType;

typedef struct {
    void* data;
    int* shape;
    int ndim;
    size_t size;
    DataType dtype;
    } Tensor;

Tensor* create_tensor(int* shape, int ndim, DataType dtype);
void free_tensor(Tensor* tensor);
void relu_forward(const Tensor* input, Tensor* output);
void cos_forward(const Tensor* input, Tensor* output);

#endif
