#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// 简单的CUDA核函数，只是拷贝数据（因为reshape不改变数据，只改变形状）
__global__ void reshape_kernel(float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx];  // 只是简单拷贝
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <input_file> <output_file> <new_shape>" << std::endl;
        std::cerr << "Note: Reshape CUDA verification only checks data preservation, not shape changes" << std::endl;
        return 1;
    }
    
    int num_elements = std::atoi(argv[1]);
    std::string new_shape_str = argv[4];  // 新形状参数（虽然CUDA kernel不直接使用）
    
    float* h_input = new float[num_elements];
    float* h_output = new float[num_elements];
    
    // 读取输入数据
    std::ifstream input_file(argv[2], std::ios::binary);
    input_file.read(reinterpret_cast<char*>(h_input), num_elements * sizeof(float));
    input_file.close();
    
    // 分配GPU内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_output, num_elements * sizeof(float));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_input, h_input, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动kernel - 只是简单拷贝，因为reshape不改变数据内容
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    reshape_kernel<<<grid_size, block_size>>>(d_input, d_output, num_elements);
    
    // 拷贝结果回CPU
    cudaMemcpy(h_output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 写入输出文件
    std::ofstream output_file(argv[3], std::ios::binary);
    output_file.write(reinterpret_cast<char*>(h_output), num_elements * sizeof(float));
    output_file.close();
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
