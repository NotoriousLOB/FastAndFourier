/**
 * @file test_core.cpp
 * @brief Tests for core library functionality
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test version information */
TEST(CoreTest, Version) {
    const char* version = dspir_version();
    EXPECT_NE(version, nullptr);
    EXPECT_STREQ(version, "1.0.0");
}

/* Test architecture detection */
TEST(CoreTest, ArchName) {
    const char* arch = dspir_arch_name();
    EXPECT_NE(arch, nullptr);
    EXPECT_STRNE(arch, "unknown");
}

/* Test alignment */
TEST(CoreTest, Alignment) {
    size_t align = dspir_get_alignment();
    EXPECT_GE(align, 16);
    EXPECT_LE(align, 256);
    /* Alignment should be power of 2 */
    EXPECT_EQ(align & (align - 1), 0);
}

/* Test precision sizes */
TEST(CoreTest, PrecisionSizes) {
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_FP8), 1);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_FP16), 2);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_BF16), 2);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_FP32), 4);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_FP64), 8);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_INT8), 1);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_INT16), 2);
    EXPECT_EQ(dspir_precision_size(DSPIR_PREC_INT32), 4);
}

/* Test precision names */
TEST(CoreTest, PrecisionNames) {
    EXPECT_STREQ(dspir_precision_name(DSPIR_PREC_FP8), "fp8");
    EXPECT_STREQ(dspir_precision_name(DSPIR_PREC_FP16), "fp16");
    EXPECT_STREQ(dspir_precision_name(DSPIR_PREC_BF16), "bf16");
    EXPECT_STREQ(dspir_precision_name(DSPIR_PREC_FP32), "fp32");
    EXPECT_STREQ(dspir_precision_name(DSPIR_PREC_FP64), "fp64");
}

/* Test transform names */
TEST(CoreTest, TransformNames) {
    EXPECT_STREQ(dspir_transform_name(DSPIR_TRANSFORM_FFT), "fft");
    EXPECT_STREQ(dspir_transform_name(DSPIR_TRANSFORM_IFFT), "ifft");
    EXPECT_STREQ(dspir_transform_name(DSPIR_TRANSFORM_DCT_II), "dct_ii");
    EXPECT_STREQ(dspir_transform_name(DSPIR_TRANSFORM_MDCT), "mdct");
    EXPECT_STREQ(dspir_transform_name(DSPIR_TRANSFORM_HAAR), "haar");
}

/* Test size support checking */
TEST(CoreTest, SizeSupport) {
    /* FFT requires power of 2 */
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 1));
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 2));
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 4));
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 1024));
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 3));
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 5));
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 100));
    
    /* MDCT requires even size */
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_MDCT, 2));
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_MDCT, 4));
    EXPECT_TRUE(dspir_is_size_supported(DSPIR_TRANSFORM_MDCT, 1024));
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_MDCT, 3));
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_MDCT, 5));
    
    /* Zero size is never supported */
    EXPECT_FALSE(dspir_is_size_supported(DSPIR_TRANSFORM_FFT, 0));
}

/* Test recommended size */
TEST(CoreTest, RecommendedSize) {
    /* Should return power of 2 for FFT */
    EXPECT_EQ(dspir_get_recommended_size(DSPIR_TRANSFORM_FFT, 1), 1);
    EXPECT_EQ(dspir_get_recommended_size(DSPIR_TRANSFORM_FFT, 2), 2);
    EXPECT_EQ(dspir_get_recommended_size(DSPIR_TRANSFORM_FFT, 3), 4);
    EXPECT_EQ(dspir_get_recommended_size(DSPIR_TRANSFORM_FFT, 100), 128);
    EXPECT_EQ(dspir_get_recommended_size(DSPIR_TRANSFORM_FFT, 1000), 1024);
}

/* Test error handling */
TEST(CoreTest, ErrorHandling) {
    dspir_clear_error();
    EXPECT_STREQ(dspir_get_error(), "");
    
    /* Create an invalid transform to trigger error */
    dspir_transform* t = dspir_create_fft(3, false, DSPIR_PREC_FP32, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(dspir_get_error(), "");
    
    dspir_clear_error();
    EXPECT_STREQ(dspir_get_error(), "");
}

/* Test transform creation and destruction */
TEST(CoreTest, TransformLifecycle) {
    /* Valid FFT creation */
    dspir_transform* t = dspir_create_fft(64, false, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid inverse FFT */
    t = dspir_create_fft(64, true, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid DCT */
    t = dspir_create_dct(64, 2, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid DST */
    t = dspir_create_dst(64, 2, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid MDCT */
    t = dspir_create_mdct(64, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid Haar wavelet */
    t = dspir_create_haar(64, 0, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Valid Daubechies-4 */
    t = dspir_create_daubechies4(64, 0, DSPIR_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        dspir_destroy_transform(t);
    }
    
    /* Null destruction should not crash */
    dspir_destroy_transform(nullptr);
}

/* Test double precision transform */
TEST(CoreTest, DoublePrecisionTransform) {
    dspir_transform* t = dspir_create_fft(64, false, DSPIR_PREC_FP64, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        EXPECT_EQ(t->precision, DSPIR_PREC_FP64);
        dspir_destroy_transform(t);
    }
}
