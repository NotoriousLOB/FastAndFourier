/**
 * @file test_cuda.cpp
 * @brief CUDA-specific tests
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"

#ifdef FAF_HAVE_CUDA

#include <cuda_runtime.h>

/* Test CUDA availability */
TEST(CUDATest, DeviceAvailable) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_GT(device_count, 0);
}

/* Test CUDA memory operations */
TEST(CUDATest, MemoryOperations) {
    const size_t n = 1024;
    
    float *d_data = nullptr;
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_NE(d_data, nullptr);
    
    if (d_data) {
        /* Initialize on device */
        err = cudaMemset(d_data, 0, n * sizeof(float));
        EXPECT_EQ(err, cudaSuccess);
        
        /* Copy to host and verify */
        float *h_data = new float[n];
        err = cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess);
        
        for (size_t i = 0; i < n; i++) {
            EXPECT_FLOAT_EQ(h_data[i], 0.0f);
        }
        
        delete[] h_data;
        cudaFree(d_data);
    }
}

/* Test CUDA FFT execution */
TEST(CUDATest, FFTExecute) {
    const size_t n = 64;
    
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    float *in = (float*)aligned_alloc(64, n * sizeof(float));
    float *out = (float*)aligned_alloc(64, n * sizeof(float));
    
    /* Impulse input */
    for (size_t i = 0; i < n; i++) {
        in[i] = (i == 0) ? 1.0f : 0.0f;
    }
    
    /* Execute using CUDA */
    faf_cuda_execute_f32(t, out, in);
    
    /* FFT of impulse should be all ones */
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], 1.0f, 1e-3f) << "Mismatch at index " << i;
    }
    
    free(in); free(out);
    faf_destroy_transform(t);
}

/* Test CUDA with various sizes */
TEST(CUDATest, VariousSizes) {
    size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    
    for (size_t n : sizes) {
        faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
        ASSERT_NE(t, nullptr) << "Failed to create FFT of size " << n;
        
        float *in = (float*)aligned_alloc(64, n * sizeof(float));
        float *out = (float*)aligned_alloc(64, n * sizeof(float));
        
        for (size_t i = 0; i < n; i++) {
            in[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        }
        
        faf_cuda_execute_f32(t, out, in);
        
        /* Check for NaN/Inf */
        for (size_t i = 0; i < n; i++) {
            EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i << " for size " << n;
            EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i << " for size " << n;
        }
        
        free(in); free(out);
        faf_destroy_transform(t);
    }
}

#endif /* FAF_HAVE_CUDA */
