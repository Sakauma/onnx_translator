#ifndef ONNX_TRANSLATOR_VERIFY_COMMON_CUH
#define ONNX_TRANSLATOR_VERIFY_COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t verify_error = (call); \
    if (verify_error != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(verify_error)); \
        exit(2); \
    } \
} while (0)

#define CUDA_CHECK_LAUNCH() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while (0)

static inline void* verify_malloc(size_t bytes) {
    void* data = malloc(bytes);
    if (!data && bytes != 0) {
        fprintf(stderr, "host allocation failed for %zu bytes\n", bytes);
        exit(3);
    }
    return data;
}

static inline void verify_read_file(const char* path, void* data, size_t bytes) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed: %s\n", path);
        exit(4);
    }
    size_t got = fread(data, 1, bytes, fp);
    int close_result = fclose(fp);
    if (got != bytes || close_result != 0) {
        fprintf(stderr, "read input failed: %s (expected %zu bytes, got %zu)\n", path, bytes, got);
        exit(4);
    }
}

static inline void verify_write_file(const char* path, const void* data, size_t bytes) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed: %s\n", path);
        exit(5);
    }
    size_t wrote = fwrite(data, 1, bytes, fp);
    int close_result = fclose(fp);
    if (wrote != bytes || close_result != 0) {
        fprintf(stderr, "write output failed: %s (expected %zu bytes, wrote %zu)\n", path, bytes, wrote);
        exit(5);
    }
}

static inline void verify_fread_exact(void* data, size_t size, size_t count, FILE* fp) {
    if (!fp) {
        fprintf(stderr, "read failed: invalid file handle\n");
        exit(4);
    }
    size_t got = fread(data, size, count, fp);
    if (got != count) {
        fprintf(stderr, "read failed: expected %zu elements, got %zu\n", count, got);
        exit(4);
    }
}

static inline void verify_fwrite_exact(const void* data, size_t size, size_t count, FILE* fp) {
    if (!fp) {
        fprintf(stderr, "write failed: invalid file handle\n");
        exit(5);
    }
    size_t wrote = fwrite(data, size, count, fp);
    if (wrote != count) {
        fprintf(stderr, "write failed: expected %zu elements, wrote %zu\n", count, wrote);
        exit(5);
    }
}

static inline void verify_close_file(FILE* fp) {
    if (!fp || fclose(fp) != 0) {
        fprintf(stderr, "close file failed\n");
        exit(5);
    }
}

#endif
