/**
 * @file test_twiddles.cpp
 * @brief Tests for twiddle factor generation
 */

#include <gtest/gtest.h>
#include "faf.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test twiddle factor correctness */
TEST(TwiddlesTest, FFTTwiddles) {
    const size_t n = 64;
    float *tw = (float*)aligned_alloc(64, n * 2 * sizeof(float));
    
    faf_gen_twiddles_f32(tw, n, false);
    
    /* Check first twiddle (k=0): W_N^0 = 1 + 0i */
    EXPECT_NEAR(tw[0], 1.0f, 1e-6f);
    EXPECT_NEAR(tw[1], 0.0f, 1e-6f);
    
    /* Check twiddle properties */
    for (size_t k = 0; k < n / 2; k++) {
        float re = tw[2*k];
        float im = tw[2*k + 1];
        
        /* Magnitude should be 1 */
        float mag = sqrtf(re * re + im * im);
        EXPECT_NEAR(mag, 1.0f, 1e-5f) << "Magnitude error at k=" << k;
        
        /* Expected values */
        float angle = -2.0f * (float)M_PI * (float)k / (float)n;
        float expected_re = cosf(angle);
        float expected_im = sinf(angle);
        
        EXPECT_NEAR(re, expected_re, 1e-5f) << "Real part error at k=" << k;
        EXPECT_NEAR(im, expected_im, 1e-5f) << "Imag part error at k=" << k;
    }
    
    /* Symmetry: W_N^(k+N/4) = W_N^k * (-i)
     * If W_N^k = a + bi, then W_N^(k+N/4) = b - ai
     * So: re2 = im1, im2 = -re1
     */
    for (size_t k = 0; k < n / 4; k++) {
        float re1 = tw[2*k];
        float im1 = tw[2*k + 1];
        float re2 = tw[2*(k + n/4)];
        float im2 = tw[2*(k + n/4) + 1];
        
        EXPECT_NEAR(re2, im1, 1e-5f) << "Symmetry error (real) at k=" << k;
        EXPECT_NEAR(im2, -re1, 1e-5f) << "Symmetry error (imag) at k=" << k;
    }
    
    free(tw);
}

/* Test inverse FFT twiddles */
TEST(TwiddlesTest, InverseFFTTwiddles) {
    const size_t n = 64;
    float *tw_forward = (float*)aligned_alloc(64, n * 2 * sizeof(float));
    float *tw_inverse = (float*)aligned_alloc(64, n * 2 * sizeof(float));
    
    faf_gen_twiddles_f32(tw_forward, n, false);
    faf_gen_twiddles_f32(tw_inverse, n, true);
    
    /* Inverse twiddles should be conjugates of forward twiddles */
    for (size_t k = 0; k < n / 2; k++) {
        EXPECT_NEAR(tw_forward[2*k], tw_inverse[2*k], 1e-6f);
        EXPECT_NEAR(tw_forward[2*k + 1], -tw_inverse[2*k + 1], 1e-6f);
    }
    
    free(tw_forward);
    free(tw_inverse);
}

/* Test double precision twiddles */
TEST(TwiddlesTest, DoublePrecisionTwiddles) {
    const size_t n = 64;
    double *tw = (double*)aligned_alloc(64, n * 2 * sizeof(double));
    
    faf_gen_twiddles_f64(tw, n, false);
    
    /* Check with higher precision */
    for (size_t k = 0; k < n / 2; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)n;
        double expected_re = cos(angle);
        double expected_im = sin(angle);
        
        EXPECT_NEAR(tw[2*k], expected_re, 1e-12);
        EXPECT_NEAR(tw[2*k + 1], expected_im, 1e-12);
    }
    
    free(tw);
}

/* Test DCT twiddles */
TEST(TwiddlesTest, DCTTwiddles) {
    const size_t n = 64;
    float *tw = (float*)aligned_alloc(64, n * 2 * sizeof(float));
    
    faf_gen_dct_twiddles_f32(tw, n, 2);
    
    /* DCT-II pre-twiddles: exp(-i*pi*k/(2*n)) */
    for (size_t k = 0; k < n; k++) {
        float angle = -(float)M_PI * (float)k / (2.0f * (float)n);
        float expected_re = cosf(angle);
        float expected_im = sinf(angle);
        
        EXPECT_NEAR(tw[2*k], expected_re, 1e-5f) << "DCT twiddle error at k=" << k;
        EXPECT_NEAR(tw[2*k + 1], expected_im, 1e-5f) << "DCT twiddle error at k=" << k;
    }
    
    free(tw);
}

/* Test DST twiddles */
TEST(TwiddlesTest, DSTTwiddles) {
    const size_t n = 64;
    float *tw = (float*)aligned_alloc(64, n * 2 * sizeof(float));
    
    faf_gen_dst_twiddles_f32(tw, n, 2);
    
    /* Verify twiddles have unit magnitude */
    for (size_t k = 0; k < n; k++) {
        float re = tw[2*k];
        float im = tw[2*k + 1];
        float mag = sqrtf(re * re + im * im);
        EXPECT_NEAR(mag, 1.0f, 1e-5f);
    }
    
    free(tw);
}

/* Test window functions */
TEST(TwiddlesTest, WindowFunctions) {
    const size_t n = 64;
    float *win = (float*)aligned_alloc(64, n * sizeof(float));
    
    /* Hann window */
    faf_gen_hann_window_f32(win, n);
    
    /* Check endpoints */
    EXPECT_NEAR(win[0], 0.0f, 1e-6f);
    EXPECT_NEAR(win[n-1], 0.0f, 1e-6f);
    
    /* Check symmetry */
    for (size_t i = 0; i < n / 2; i++) {
        EXPECT_NEAR(win[i], win[n - 1 - i], 1e-6f);
    }
    
    /* Check maximum at center */
    float max_val = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (win[i] > max_val) {
            max_val = win[i];
            max_idx = i;
        }
    }
    EXPECT_NEAR(max_idx, n / 2, 1);
    EXPECT_NEAR(max_val, 1.0f, 0.01f);
    
    /* Hamming window */
    faf_gen_hamming_window_f32(win, n);
    
    /* Hamming doesn't go to zero at endpoints */
    EXPECT_GT(win[0], 0.05f);
    EXPECT_GT(win[n-1], 0.05f);
    
    /* Check symmetry */
    for (size_t i = 0; i < n / 2; i++) {
        EXPECT_NEAR(win[i], win[n - 1 - i], 1e-6f);
    }
    
    /* Blackman window */
    faf_gen_blackman_window_f32(win, n);
    
    /* Check endpoints */
    EXPECT_NEAR(win[0], 0.0f, 1e-5f);
    EXPECT_NEAR(win[n-1], 0.0f, 1e-5f);
    
    free(win);
}

/* Test wavelet coefficients */
TEST(TwiddlesTest, WaveletCoefficients) {
    float lo[4], hi[4];
    
    /* Haar coefficients */
    faf_gen_haar_coeffs_f32(lo, hi);
    
    /* Check orthogonality: lo[0]*lo[1] + hi[0]*hi[1] = 0 */
    float ortho = lo[0] * lo[1] + hi[0] * hi[1];
    EXPECT_NEAR(ortho, 0.0f, 1e-6f);
    
    /* Check reconstruction: lo[i]^2 + hi[i]^2 = 1 for each i */
    for (int i = 0; i < 2; i++) {
        float norm = lo[i] * lo[i] + hi[i] * hi[i];
        EXPECT_NEAR(norm, 1.0f, 1e-6f);
    }
    
    /* Check values */
    float expected = 0.7071067811865476f;
    EXPECT_NEAR(lo[0], expected, 1e-6f);
    EXPECT_NEAR(lo[1], expected, 1e-6f);
    EXPECT_NEAR(hi[0], expected, 1e-6f);
    EXPECT_NEAR(hi[1], -expected, 1e-6f);
    
    /* Daubechies-4 coefficients */
    float lo_d4[4], hi_d4[4];
    faf_gen_daubechies4_coeffs_f32(lo_d4, hi_d4);
    
    /* Check sum of low-pass coefficients */
    float sum_lo = lo_d4[0] + lo_d4[1] + lo_d4[2] + lo_d4[3];
    EXPECT_NEAR(sum_lo, 0.7071067811865476f * 2, 0.1f);
    
    /* Check sum of high-pass coefficients */
    float sum_hi = hi_d4[0] + hi_d4[1] + hi_d4[2] + hi_d4[3];
    EXPECT_NEAR(sum_hi, 0.0f, 1e-6f);
}
