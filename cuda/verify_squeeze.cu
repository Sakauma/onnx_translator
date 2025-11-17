#include <cstdio>
#include <cstdlib>

// Squeeze 只是数据拷贝（形状变换在Python端处理）
__global__ void squeeze_kernel(float* output, const float* input, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx];  // 直接拷贝，因为squeeze不改变数据
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <num_elements> <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int num_elements = atoi(argv[1]);
    const char* input_file = argv[2];
    const char* output_file = argv[3];

    // 分配主机内存
    float* h_input = new float[num_elements];
    float* h_output = new float[num_elements];

    // 读取输入数据
    FILE* f = fopen(input_file, "rb");
    fread(h_input, sizeof(float), num_elements, f);
    fclose(f);

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, num_elements * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数（简单拷贝）
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    squeeze_kernel<<<grid_size, block_size>>>(d_output, d_input, num_elements);

    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 写入输出文件
    f = fopen(output_file, "wb");
    fwrite(h_output, sizeof(float), num_elements, f);
    fclose(f);

    // 清理资源
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}