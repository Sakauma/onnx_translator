// verify_mul.cu - CUDA验证程序，用于测试MUL操作的正确性
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * CUDA核函数：计算float32类型数据的逐元素乘法
 *
 * @param input1 第一个输入数据指针
 * @param input2 第二个输入数据指针
 * @param output 输出数据指针
 * @param size 数据元素个数（两个输入的元素个数需相同）
 */
__global__ void mul_kernel_float32(const float* input1, const float* input2, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // MUL的核心逻辑: output = input1 * input2
        output[idx] = input1[idx] * input2[idx];
    }
}

/**
 * 主函数：演示CUDA MUL计算功能
 * 从两个输入文件读取数据，计算逐元素积后写入输出文件
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <num_elements> <input_file1> <input_file2> <output_file>\n", argv[0]);
        return 1;
    }

    size_t num_elements = atol(argv[1]);
    const char* input_filename1 = argv[2];  // 第一个输入文件
    const char* input_filename2 = argv[3];  // 第二个输入文件
    const char* output_filename = argv[4];  // 输出文件
    size_t data_size = num_elements * sizeof(float);  // 单个输入/输出的数据大小

    // 1. 从文件读取主机端两个输入数据
    float* h_input1 = (float*)malloc(data_size);
    float* h_input2 = (float*)malloc(data_size);
    if (!h_input1 || !h_input2) {
        fprintf(stderr, "Error allocating host memory.\n");
        return 1;
    }

    // 读取第一个输入文件
    FILE* fp_in1 = fopen(input_filename1, "rb");
    if (!fp_in1 || fread(h_input1, 1, data_size, fp_in1) != data_size) {
        fprintf(stderr, "Error reading input file 1.\n");
        return 1;
    }
    fclose(fp_in1);

    // 读取第二个输入文件
    FILE* fp_in2 = fopen(input_filename2, "rb");
    if (!fp_in2 || fread(h_input2, 1, data_size, fp_in2) != data_size) {
        fprintf(stderr, "Error reading input file 2.\n");
        return 1;
    }
    fclose(fp_in2);

    // 2. 在GPU上分配内存（两个输入+一个输出）
    float *d_input1, *d_input2, *d_output;
    cudaMalloc((void**)&d_input1, data_size);
    cudaMalloc((void**)&d_input2, data_size);
    cudaMalloc((void**)&d_output, data_size);

    // 3. 将主机数据复制到设备
    cudaMemcpy(d_input1, h_input1, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, data_size, cudaMemcpyHostToDevice);

    // 4. 配置并启动CUDA核函数
    int threads = 256;  // 每个块的线程数
    int blocks = (num_elements + threads - 1) / threads;  // 计算所需块数
    mul_kernel_float32<<<blocks, threads>>>(d_input1, d_input2, d_output, num_elements);
    cudaDeviceSynchronize();  // 等待核函数执行完成

    // 5. 将结果复制回主机
    float* h_result = (float*)malloc(data_size);
    cudaMemcpy(h_result, d_output, data_size, cudaMemcpyDeviceToHost);

    // 6. 将主机端结果写入文件
    FILE* fp_out = fopen(output_filename, "wb");
    if (!fp_out || fwrite(h_result, 1, data_size, fp_out) != data_size) {
        fprintf(stderr, "Error writing output file.\n");
        return 1;
    }
    fclose(fp_out);

    // 7. 释放资源
    free(h_input1);
    free(h_input2);
    free(h_result);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    return 0;
}