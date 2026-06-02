/**
  ******************************************************************************
  * @file        tensor_ops_core.c
  * @author      Egor Izmaylov
  * @brief       实现 Tensor 创建、释放等 C 后端核心 ABI 入口。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#include "tensor_ops_internal.h"


/**
 * 创建张量
 * 
 * @param shape 张量形状数组
 * @param ndim 张量维度数
 * @param dtype 数据类型
 * @return 创建的张量指针
 */
// 实现 `create_tensor` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
Tensor* create_tensor(int* shape, int ndim, DataType dtype) {
    if (ndim < 0) {
        return NULL;
    }

    // 分配张量结构体内存
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        return NULL;
    }
    
    // 设置维度数
    tensor->ndim = ndim;
    
    // 分配并复制形状数组
    tensor->shape = NULL;
    if (ndim > 0) {
        if (!shape) {
            free(tensor);
            return NULL;
        }
        tensor->shape = (int*)malloc(ndim * sizeof(int));
        if (!tensor->shape) {
            free(tensor);
            return NULL;
        }
        memcpy(tensor->shape, shape, ndim * sizeof(int));
    }
    
    // 设置数据类型
    tensor->dtype = dtype;
    
    // 计算总元素数
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
        tensor->size *= shape[i];
    }
    
    // 根据数据类型分配数据内存
    size_t elem_size = 0;
    switch (dtype) {
        case DTYPE_FLOAT8_E4M3:
        case DTYPE_FLOAT8_E5M2:
            elem_size = 1;  // 8位浮点数
            break;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            elem_size = 2;  // 16位浮点数
            break;
        case DTYPE_FLOAT32:
            elem_size = 4;  // 32位浮点数
            break;
        case DTYPE_FLOAT64:
            elem_size = 8;  // 64位浮点数
            break;
        case DTYPE_INT4:
            elem_size = 1;  // 4位整数
            break;
        case DTYPE_INT8:
            elem_size = 1;  // 8位整数
            break;
        case DTYPE_UINT8:
            elem_size = 1;  // 8位无符号整数
            break;
        case DTYPE_INT16:
            elem_size = 2;  // 16位整数
            break;
        case DTYPE_INT32:
            elem_size = 4;  // 32位整数
            break;
        case DTYPE_INT64:
            elem_size = 8;  // 64位整数
            break;
        default:
            elem_size = 4;  // 默认32位
            break;
    }
    
    // 分配数据内存
    //tensor->data = malloc(tensor->size * elem_size);
    size_t alloc_count = tensor->size == 0 ? 1 : tensor->size;
    tensor->data = calloc(alloc_count, elem_size);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    return tensor;
}


/**
 * 释放张量内存
 * 
 * @param tensor 要释放的张量指针
 */
// 实现 `free_tensor` 共享辅助逻辑，集中处理索引、形状、随机数、归约或数学细节。
void free_tensor(Tensor* tensor) {
    // 检查张量指针是否有效
    if (tensor) {
        // 释放数据内存
        free(tensor->data);
        // 释放形状数组内存
        free(tensor->shape);
        // 释放张量结构体内存
        free(tensor);
    }
}

