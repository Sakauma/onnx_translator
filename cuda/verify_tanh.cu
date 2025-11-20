#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void tanh_kernel(float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = tanhf(input[idx]);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <input_file> <output_file>" << std::endl;
        return 1;
    }
    
    int num_elements = std::atoi(argv[1]);
    std::string input_filename = argv[2];
    std::string output_filename = argv[3];
    
    std::cout << "CUDA Tanh Verification: " << num_elements << " elements" << std::endl;
    std::cout << "Input: " << input_filename << ", Output: " << output_filename << std::endl;
    
    // 分配主机内存
    float* h_input = new float[num_elements];
    float* h_output = new float[num_elements];
    
    // 读取输入数据
    std::ifstream input_file(input_filename, std::ios::binary);
    if (!input_file) {
        std::cerr << "Error: Cannot open input file " << input_filename << std::endl;
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    input_file.read(reinterpret_cast<char*>(h_input), num_elements * sizeof(float));
    input_file.close();
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaError_t err;
    
    err = cudaMalloc(&d_input, num_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    err = cudaMalloc(&d_output, num_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    // 拷贝数据到GPU
    err = cudaMemcpy(d_input, h_input, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    // 启动kernel
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    std::cout << "Launching kernel: grid=" << grid_size << ", block=" << block_size << std::endl;
    
    tanh_kernel<<<grid_size, block_size>>>(d_input, d_output, num_elements);
    
    // 检查kernel执行是否成功
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    // 等待kernel完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    // 拷贝结果回CPU
    err = cudaMemcpy(h_output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    
    // 写入输出文件
    std::ofstream output_file(output_filename, std::ios::binary);
    if (!output_file) {
        std::cerr << "Error: Cannot open output file " << output_filename << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
        return 1;
    }
    output_file.write(reinterpret_cast<char*>(h_output), num_elements * sizeof(float));
    output_file.close();
    
    std::cout << "Successfully wrote output to " << output_filename << std::endl;
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
