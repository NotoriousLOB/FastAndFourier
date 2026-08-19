/**
 * @file test_precision.cpp
 * @brief Tests for different precision levels
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "faf_test_util.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test FP32 precision */
TEST(PrecisionTest, FP32Accuracy) {
    const size_t n = 256;
    faf_transform* t = test_fft_n(n);
    ASSERT_NE(t, nullptr);
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Single frequency */
    const int k = 8;
    for (size_t i = 0; i < n; i++) {
        in[2*i] = cosf(2.0f * (float)M_PI * k * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    faf_execute_f32(t, out, in);
    
    /* Check that energy is concentrated at the expected frequency */
    float peak_energy = 0.0f;
    size_t peak_idx = 0;
    for (size_t i = 0; i < n; i++) {
        float energy = out[2*i] * out[2*i] + out[2*i+1] * out[2*i+1];
        if (energy > peak_energy) {
            peak_energy = energy;
            peak_idx = i;
        }
    }
    
    EXPECT_EQ(peak_idx, k);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

/* Test FP64 precision (higher accuracy) */
TEST(PrecisionTest, FP64Accuracy) {
    const size_t n = 256;
    faf_transform* t = test_fft(n, false, FAF_PREC_FP64);
    ASSERT_NE(t, nullptr);
    
    double *in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double *out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    
    /* Single frequency */
    const int k = 8;
    for (size_t i = 0; i < n; i++) {
        in[2*i] = cos(2.0 * M_PI * k * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    faf_execute_f64(t, out, in);
    
    /* Check that energy is concentrated at the expected frequency */
    double peak_energy = 0.0;
    size_t peak_idx = 0;
    for (size_t i = 0; i < n; i++) {
        double energy = out[2*i] * out[2*i] + out[2*i+1] * out[2*i+1];
        if (energy > peak_energy) {
            peak_energy = energy;
            peak_idx = i;
        }
    }
    
    EXPECT_EQ(peak_idx, k);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

/* Compare FP32 vs FP64 accuracy */
TEST(PrecisionTest, FP32vsFP64) {
    const size_t n = 256;
    
    faf_transform* t32 = test_fft_n(n);
    faf_transform* t64 = test_fft(n, false, FAF_PREC_FP64);
    ASSERT_NE(t32, nullptr);
    ASSERT_NE(t64, nullptr);
    
    float *in32 = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out32 = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    double *in64 = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double *out64 = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    
    /* Challenging signal: sum of many frequencies */
    for (size_t i = 0; i < n; i++) {
        double val = 0.0;
        for (int k = 1; k <= 16; k++) {
            val += sin(2.0 * M_PI * k * (double)i / (double)n) / k;
        }
        in32[2*i] = (float)val;
        in32[2*i+1] = 0.0f;
        in64[2*i] = val;
        in64[2*i+1] = 0.0;
    }
    
    faf_execute_f32(t32, out32, in32);
    faf_execute_f64(t64, out64, in64);
    
    /* Compare results - they should be close but not identical */
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)out32[2*i] - out64[2*i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    /* FP32 should be within ~1e-4 of FP64 for this test */
    EXPECT_LT(max_diff, 1e-3);
    
    free(in32); free(out32);
    free(in64); free(out64);
    faf_destroy_transform(t32);
    faf_destroy_transform(t64);
}

/* Test numerical stability with large transforms */
TEST(PrecisionTest, LargeTransformStability) {
    const size_t n = 4096;
    faf_transform* t = test_fft_n(n);
    ASSERT_NE(t, nullptr);
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Random-like signal */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 100.0f * (float)i / (float)n) +
                0.5f * sinf(2.0f * (float)M_PI * 200.0f * (float)i / (float)n) +
                0.25f * sinf(2.0f * (float)M_PI * 400.0f * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    faf_execute_f32(t, out, in);
    
    /* Check that output doesn't contain NaN or Inf */
    for (size_t i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(out[2*i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[2*i])) << "Inf at index " << i;
        EXPECT_FALSE(std::isnan(out[2*i+1])) << "NaN at imag index " << i;
        EXPECT_FALSE(std::isinf(out[2*i+1])) << "Inf at imag index " << i;
    }
    
    free(in); free(out);
    faf_destroy_transform(t);
}

/* Test DCT with different precisions */
TEST(PrecisionTest, DCTPrecision) {
    const size_t n = 128;
    
    /* FP32 DCT */
    faf_transform* t32 = test_dct(n, 2);
    ASSERT_NE(t32, nullptr);
    
    float *in32 = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out32 = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in32[2*i] = cosf((float)M_PI * (float)i / (float)n);
        in32[2*i+1] = 0.0f;
    }
    
    faf_execute_f32(t32, out32, in32);
    
    /* Check for NaN/Inf */
    for (size_t i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(out32[2*i]));
        EXPECT_FALSE(std::isinf(out32[2*i]));
        EXPECT_FALSE(std::isnan(out32[2*i+1]));
        EXPECT_FALSE(std::isinf(out32[2*i+1]));
    }
    
    free(in32); free(out32);
    faf_destroy_transform(t32);
}
