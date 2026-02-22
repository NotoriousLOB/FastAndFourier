/**
 * @file test_vm.cpp
 * @brief Tests for the virtual machine execution
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* aligned_alloc requires size to be a multiple of alignment */
#define ALIGNED_ALLOC64(n) aligned_alloc(64, (((size_t)(n)) + 63u) & ~(size_t)63u)

/* Helper: Compare floats with tolerance */
static bool near_equal(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

/* Test simple FFT execution */
TEST(VMTest, FFTExecute) {
    const size_t n = 64;
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* Create complex input: impulse at position 0 (real=1, imag=0) */
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (i == 0) ? 1.0f : 0.0f;  /* Real part */
        in[2*i + 1] = 0.0f;                 /* Imag part */
    }
    
    /* Execute */
    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);
    
    /* FFT of impulse should be all ones (real=1, imag=0) */
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(out[2*i], 1.0f, 1e-3f) << "Real mismatch at index " << i;
        EXPECT_NEAR(out[2*i + 1], 0.0f, 1e-3f) << "Imag mismatch at index " << i;
    }
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test FFT of constant signal */
TEST(VMTest, FFTConstant) {
    const size_t n = 64;
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    /* Constant complex signal (1 + 0i) */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = 1.0f;      /* Real part */
        in[2*i + 1] = 0.0f;  /* Imag part */
    }
    
    faf_execute_f32(t, out, in);
    
    /* FFT of constant: impulse at DC (position 0) with value n */
    EXPECT_NEAR(out[0], (float)n, 1e-2f);  /* DC real = n */
    EXPECT_NEAR(out[1], 0.0f, 1e-3f);      /* DC imag = 0 */
    for (size_t i = 1; i < n; i++) {
        EXPECT_NEAR(out[2*i], 0.0f, 1e-3f) << "Non-zero real at index " << i;
        EXPECT_NEAR(out[2*i + 1], 0.0f, 1e-3f) << "Non-zero imag at index " << i;
    }
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test FFT of sine wave */
TEST(VMTest, FFTSineWave) {
    const size_t n = 64;
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    /* Single frequency sine wave: sin(2*pi*k*i/n) = imag part of exp(j*2*pi*k*i/n) */
    /* FFT of sine has peaks at k and n-k with opposite signs in imaginary parts */
    const int k = 4;  /* 4 cycles */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * k * (float)i / (float)n);   /* Real */
        in[2*i + 1] = 0.0f;                                              /* Imag */
    }
    
    faf_execute_f32(t, out, in);
    
    /* Should have peaks at bins k and n-k */
    for (size_t i = 0; i < n; i++) {
        float real_val = out[2*i];
        float imag_val = out[2*i + 1];
        float magnitude = std::sqrt(real_val * real_val + imag_val * imag_val);
        
        if (i == (size_t)k || i == n - k) {
            EXPECT_GT(magnitude, 10.0f) << "Expected peak at index " << i;
        } else {
            EXPECT_NEAR(magnitude, 0.0f, 5.0f) << "Unexpected value at index " << i;
        }
    }
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test DCT-II execution - DCT uses real input */
TEST(VMTest, DCTExecute) {
    const size_t n = 64;
    faf_transform* t = faf_create_dct(n, 2, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* DCT operates on real data, but our VM expects complex */
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    /* Impulse input */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (i == 0) ? 1.0f : 0.0f;  /* Real */
        in[2*i + 1] = 0.0f;                 /* Imag */
    }
    
    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);
    
    /* For now just check it runs - DCT implementation is simplified */
    (void)out;
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test Haar wavelet */
TEST(VMTest, HaarWavelet) {
    const size_t n = 64;
    faf_transform* t = faf_create_haar(n, 1, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* Haar operates on real data */
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    /* Constant input of 1.0 - Haar average should be 1.0 */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = 1.0f;
        in[2*i + 1] = 0.0f;  /* Imag = 0 */
    }
    
    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);
    
    /* After Haar transform on constant input:
     * BFLY2 does (a+b, a-b), so for input (1,1): sum=2, diff=0
     * - All sums should be 2.0 (since input is constant 1.0)
     * - All differences should be 0 (since no variation)
     */
    for (size_t i = 0; i < n / 2; i += 2) {
        /* Even indices: sums (a+b) */
        EXPECT_NEAR(out[2*i], 2.0f, 0.1f) << "Sum at i=" << i;
        /* Odd indices: differences (a-b) */
        EXPECT_NEAR(out[2*(i+1)], 0.0f, 0.1f) << "Diff at i=" << i+1;
    }
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test double precision execution */
TEST(VMTest, DoublePrecision) {
    const size_t n = 64;
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP64, 0);
    ASSERT_NE(t, nullptr);
    
    double *in = (double*)ALIGNED_ALLOC64(2 * n * sizeof(double));
    double *out = (double*)ALIGNED_ALLOC64(2 * n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (i == 0) ? 1.0 : 0.0;  /* Real */
        in[2*i + 1] = 0.0;                /* Imag */
    }
    
    int result = faf_execute_f64(t, out, in);
    EXPECT_EQ(result, 0);
    
    /* FFT of impulse should be all ones */
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(out[2*i], 1.0, 1e-10) << "Real mismatch at index " << i;
        EXPECT_NEAR(out[2*i + 1], 0.0, 1e-10) << "Imag mismatch at index " << i;
    }
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* Test various FFT sizes */
TEST(VMTest, VariousSizes) {
    size_t sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    for (size_t n : sizes) {
        faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
        ASSERT_NE(t, nullptr) << "Failed to create FFT of size " << n;
        
        /* aligned_alloc requires size to be a multiple of alignment */
        size_t bufsize = ((2 * n * sizeof(float)) + 63u) & ~(size_t)63u;
        float *in = (float*)aligned_alloc(64, bufsize);
        float *out = (float*)aligned_alloc(64, bufsize);
        
        /* Ramp function for complex input */
        for (size_t i = 0; i < n; i++) {
            in[2*i] = (float)i;     /* Real */
            in[2*i + 1] = 0.0f;     /* Imag */
        }
        
        int result = faf_execute_f32(t, out, in);
        EXPECT_EQ(result, 0) << "Execution failed for size " << n;
        
        free(in);
        free(out);
        faf_destroy_transform(t);
    }
}

/* Test MDCT execution */
TEST(VMTest, MDCTExecute) {
    const size_t n = 64;
    faf_transform* t = faf_create_mdct(n, FAF_PREC_FP32, 0);
    ASSERT_NE(t, nullptr);
    
    /* MDCT input is real, but our VM expects complex format */
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i + 1] = 0.0f;
    }
    
    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
