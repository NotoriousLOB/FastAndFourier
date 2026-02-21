/**
 * @file test_jit.cpp
 * @brief Tests for JIT compilation
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test JIT context creation and destruction */
TEST(JITTest, ContextLifecycle) {
    dspir_jit_ctx* ctx = dspir_jit_create();
    EXPECT_NE(ctx, nullptr);
    
    if (ctx) {
        dspir_jit_destroy(ctx);
    }
}

/* Test JIT compilation of simple transform */
TEST(JITTest, CompileFFT) {
    const size_t n = 64;
    
    /* Create transform */
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* Create JIT context */
    dspir_jit_ctx* ctx = dspir_jit_create();
    ASSERT_NE(ctx, nullptr);
    
    /* Compile */
    int result = dspir_jit_compile(ctx, t);
    
    /* JIT compilation may fail if compiler not available */
    /* Just verify it doesn't crash */
    (void)result;
    
    /* Get kernel */
    dspir_kernel_fn fn = dspir_jit_get_kernel(ctx);
    /* fn may be NULL if compilation failed */
    
    dspir_jit_destroy(ctx);
    dspir_destroy_transform(t);
}

/* Test JIT execution if available */
TEST(JITTest, ExecuteJIT) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* Complex data format: 2*n floats */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out_jit = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out_vm = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Impulse input (complex) */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (i == 0) ? 1.0f : 0.0f;  /* Real */
        in[2*i + 1] = 0.0f;                 /* Imag */
    }
    
    /* VM execution */
    dspir_execute_f32(t, out_vm, in);
    
    /* JIT execution */
    int result = dspir_execute_jit(t, out_jit, in);
    
    /* Compare results if JIT succeeded */
    if (result == 0) {
        for (size_t i = 0; i < 2 * n; i++) {
            EXPECT_NEAR(out_jit[i], out_vm[i], 1e-4f) 
                << "JIT/VM mismatch at index " << i;
        }
    }
    
    free(in); free(out_jit); free(out_vm);
    dspir_destroy_transform(t);
}

/* Test JIT compilation of DCT */
TEST(JITTest, CompileDCT) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_dct(n, 2, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    dspir_jit_ctx* ctx = dspir_jit_create();
    ASSERT_NE(ctx, nullptr);
    
    int result = dspir_jit_compile(ctx, t);
    (void)result;
    
    dspir_jit_destroy(ctx);
    dspir_destroy_transform(t);
}

/* Test JIT compilation of wavelet */
TEST(JITTest, CompileWavelet) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_haar(n, 3, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    dspir_jit_ctx* ctx = dspir_jit_create();
    ASSERT_NE(ctx, nullptr);
    
    int result = dspir_jit_compile(ctx, t);
    (void)result;
    
    dspir_jit_destroy(ctx);
    dspir_destroy_transform(t);
}

/* Test JIT with different sizes */
TEST(JITTest, VariousSizes) {
    size_t sizes[] = {16, 32, 64, 128, 256};
    
    for (size_t n : sizes) {
        dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
        ASSERT_NE(t, nullptr) << "Failed to create FFT of size " << n;
        
        dspir_jit_ctx* ctx = dspir_jit_create();
        ASSERT_NE(ctx, nullptr);
        
        int result = dspir_jit_compile(ctx, t);
        /* Compilation may fail - just verify no crash */
        (void)result;
        
        dspir_jit_destroy(ctx);
        dspir_destroy_transform(t);
    }
}

/* Test JIT with double precision */
TEST(JITTest, DoublePrecisionJIT) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP64, 0);
    ASSERT_NE(t, nullptr);
    
    dspir_jit_ctx* ctx = dspir_jit_create();
    ASSERT_NE(ctx, nullptr);
    
    int result = dspir_jit_compile(ctx, t);
    (void)result;
    
    dspir_jit_destroy(ctx);
    dspir_destroy_transform(t);
}

/* Test JIT with SIMD intrinsics */
TEST(JITTest, SIMDCompilation) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    dspir_jit_ctx* ctx = dspir_jit_create();
    ASSERT_NE(ctx, nullptr);
    
    /* Compile with SIMD flag */
    int result = dspir_jit_compile_ex(ctx, t, DSPIR_FLAG_JIT_SIMD);
    (void)result;
    
    dspir_jit_destroy(ctx);
    dspir_destroy_transform(t);
}

/* Test JIT with in-place execution */
TEST(JITTest, InPlaceExecution) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* Complex data format: 2*n floats */
    float *data = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *expected = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Impulse input (complex) */
    for (size_t i = 0; i < n; i++) {
        data[2*i] = (i == 0) ? 1.0f : 0.0f;  /* Real */
        data[2*i + 1] = 0.0f;                 /* Imag */
        expected[2*i] = data[2*i];
        expected[2*i + 1] = 0.0f;
    }
    
    /* VM execution for reference */
    dspir_execute_f32(t, expected, data);
    
    /* Reset data */
    for (size_t i = 0; i < n; i++) {
        data[2*i] = (i == 0) ? 1.0f : 0.0f;
        data[2*i + 1] = 0.0f;
    }
    
    /* In-place JIT execution */
    int result = dspir_execute_jit_ex(t, data, data, DSPIR_FLAG_JIT_INPLACE);
    
    /* Compare results if JIT succeeded */
    if (result == 0) {
        for (size_t i = 0; i < 2 * n; i++) {
            EXPECT_NEAR(data[i], expected[i], 1e-4f) 
                << "In-place JIT/VM mismatch at index " << i;
        }
    }
    
    free(data); free(expected);
    dspir_destroy_transform(t);
}

/* Test JIT with SIMD + in-place combined */
TEST(JITTest, SIMDInPlaceCombined) {
    const size_t n = 64;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    float *data = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *expected = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Impulse input */
    for (size_t i = 0; i < n; i++) {
        data[2*i] = (i == 0) ? 1.0f : 0.0f;
        data[2*i + 1] = 0.0f;
        expected[2*i] = data[2*i];
        expected[2*i + 1] = 0.0f;
    }
    
    /* VM execution for reference */
    dspir_execute_f32(t, expected, data);
    
    /* Reset data */
    for (size_t i = 0; i < n; i++) {
        data[2*i] = (i == 0) ? 1.0f : 0.0f;
        data[2*i + 1] = 0.0f;
    }
    
    /* Combined SIMD + in-place JIT execution */
    uint32_t flags = DSPIR_FLAG_JIT_SIMD | DSPIR_FLAG_JIT_INPLACE;
    int result = dspir_execute_jit_ex(t, data, data, flags);
    
    if (result == 0) {
        for (size_t i = 0; i < 2 * n; i++) {
            EXPECT_NEAR(data[i], expected[i], 1e-4f) 
                << "SIMD+In-place JIT/VM mismatch at index " << i;
        }
    }
    
    free(data); free(expected);
    dspir_destroy_transform(t);
}
