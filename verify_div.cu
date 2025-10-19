#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>  // 用于fabs判断除数绝对值

// CUDA错误检查宏，增强调试能力
#define CHECK_CUDA_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

/**
 * CUDA核函数：计算float32类型数据的逐元素除法
 *
 * @param input1 被除数数据指针（分子）
 * @param input2 除数数据指针（分母）
 * @param output 输出结果指针（商）
 * @param size 数据元素个数（两个输入的元素个数需相同）
 */
__global__ void div_kernel_float32(const float* input1, const float* input2, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 处理除数接近0的情况（避免NaN/无穷大，float精度阈值设为1e-6）
        if (fabs(input2[idx]) < 1e-6f) {
            printf("Error: Divide by zero at index %zu (input2[idx] = %f)\n", idx, input2[idx]);
            output[idx] = NAN;  // 赋值为NaN，标记错误结果
            return;
        }
        // DIV的核心逻辑: output = input1（分子） / input2（分母）
        output[idx] = input1[idx] / input2[idx];
    }
}

/**
 * 主函数：演示CUDA DIV计算功能
 * 从两个输入文件读取数据，计算逐元素商后写入输出文件
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <num_elements> <input_file1> <input_file2> <output_file>\n", argv[0]);
        fprintf(stderr, "Note: input_file1 = numerator (被除数), input_file2 = denominator (除数)\n");
        return 1;
    }

    size_t num_elements = atol(argv[1]);
    const char* input_filename1 = argv[2];  // 输入文件1：被除数（分子）
    const char* input_filename2 = argv[3];  // 输入文件2：除数（分母）
    const char* output_filename = argv[4];  // 输出文件：商
    size_t data_size = num_elements * sizeof(float);  // 单个输入/输出的数据大小

    // 1. 主机端（CPU）分配内存并读取输入文件
    float* h_input1 = (float*)malloc(data_size);
    float* h_input2 = (float*)malloc(data_size);
    if (!h_input1 || !h_input2) {
        fprintf(stderr, "Error: Allocate host memory failed.\n");
        return 1;
    }

    // 读取被除数文件（input1）
    FILE* fp_in1 = fopen(input_filename1, "rb");
    if (!fp_in1 || fread(h_input1, 1, data_size, fp_in1) != data_size) {
        fprintf(stderr, "Error: Read input file 1 (numerator) failed.\n");
        return 1;
    }
    fclose(fp_in1);

    // 读取除数文件（input2）
    FILE* fp_in2 = fopen(input_filename2, "rb");
    if (!fp_in2 || fread(h_input2, 1, data_size, fp_in2) != data_size) {
        fprintf(stderr, "Error: Read input file 2 (denominator) failed.\n");
        return 1;
    }
    fclose(fp_in2);

    // 2. 设备端（GPU）分配内存
    float *d_input1, *d_input2, *d_output;
    cudaError_t err;
    err = cudaMalloc((void**)&d_input1, data_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_input2, data_size);
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_output, data_size);
    CHECK_CUDA_ERROR(err);

    // 3. 主机数据复制到设备（Host -> Device）
    err = cudaMemcpy(d_input1, h_input1, data_size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_input2, h_input2, data_size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // 4. 配置并启动CUDA核函数
    int threads_per_block = 256;  // 每个线程块的线程数（CUDA推荐值）
    int block_num = (num_elements + threads_per_block - 1) / threads_per_block;  // 向上取整计算线程块数
    div_kernel_float32<<<block_num, threads_per_block>>>(d_input1, d_input2, d_output, num_elements);

    // 检查核函数启动错误
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err);
    // 等待核函数执行完成（同步设备与主机）
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // 5. 设备结果复制回主机（Device -> Host）
    float* h_result = (float*)malloc(data_size);
    if (!h_result) {
        fprintf(stderr, "Error: Allocate host result memory failed.\n");
        return 1;
    }
    err = cudaMemcpy(h_result, d_output, data_size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    // 6. 将结果写入输出文件
    FILE* fp_out = fopen(output_filename, "wb");
    if (!fp_out || fwrite(h_result, 1, data_size, fp_out) != data_size) {
        fprintf(stderr, "Error: Write output file failed.\n");
        return 1;
    }
    fclose(fp_out);
    printf("Success: DIV calculation completed, result saved to %s\n", output_filename);

    // 7. 释放所有内存（避免内存泄漏）
    free(h_input1);
    free(h_input2);
    free(h_result);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    return 0;
}