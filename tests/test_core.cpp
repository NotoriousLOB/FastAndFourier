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
    const char* version = faf_version();
    EXPECT_NE(version, nullptr);
    EXPECT_STREQ(version, "1.0.0");
}

/* Test architecture detection */
TEST(CoreTest, ArchName) {
    const char* arch = faf_arch_name();
    EXPECT_NE(arch, nullptr);
    EXPECT_STRNE(arch, "unknown");
}

/* Test alignment */
TEST(CoreTest, Alignment) {
    size_t align = faf_get_alignment();
    EXPECT_GE(align, 16);
    EXPECT_LE(align, 256);
    /* Alignment should be power of 2 */
    EXPECT_EQ(align & (align - 1), 0);
}

/* Test precision sizes */
TEST(CoreTest, PrecisionSizes) {
    EXPECT_EQ(faf_precision_size(FAF_PREC_FP8), 1);
    EXPECT_EQ(faf_precision_size(FAF_PREC_FP16), 2);
    EXPECT_EQ(faf_precision_size(FAF_PREC_BF16), 2);
    EXPECT_EQ(faf_precision_size(FAF_PREC_FP32), 4);
    EXPECT_EQ(faf_precision_size(FAF_PREC_FP64), 8);
    EXPECT_EQ(faf_precision_size(FAF_PREC_INT8), 1);
    EXPECT_EQ(faf_precision_size(FAF_PREC_INT16), 2);
    EXPECT_EQ(faf_precision_size(FAF_PREC_INT32), 4);
}

/* Test precision names */
TEST(CoreTest, PrecisionNames) {
    EXPECT_STREQ(faf_precision_name(FAF_PREC_FP8), "fp8");
    EXPECT_STREQ(faf_precision_name(FAF_PREC_FP16), "fp16");
    EXPECT_STREQ(faf_precision_name(FAF_PREC_BF16), "bf16");
    EXPECT_STREQ(faf_precision_name(FAF_PREC_FP32), "fp32");
    EXPECT_STREQ(faf_precision_name(FAF_PREC_FP64), "fp64");
}

/* Test transform names */
TEST(CoreTest, TransformNames) {
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_FFT), "fft");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_IFFT), "ifft");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_DCT_II), "dct_ii");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_MDCT), "mdct");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_HAAR), "haar");
}

/* Test size support checking */
TEST(CoreTest, SizeSupport) {
    /* FFT requires power of 2 */
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 1));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 2));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 4));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 1024));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 3));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 5));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 100));
    
    /* MDCT requires even size */
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_MDCT, 2));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_MDCT, 4));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_MDCT, 1024));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_MDCT, 3));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_MDCT, 5));
    
    /* Zero size is never supported */
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 0));
}

/* Test recommended size */
TEST(CoreTest, RecommendedSize) {
    /* Should return power of 2 for FFT */
    EXPECT_EQ(faf_get_recommended_size(FAF_TRANSFORM_FFT, 1), 1);
    EXPECT_EQ(faf_get_recommended_size(FAF_TRANSFORM_FFT, 2), 2);
    EXPECT_EQ(faf_get_recommended_size(FAF_TRANSFORM_FFT, 3), 4);
    EXPECT_EQ(faf_get_recommended_size(FAF_TRANSFORM_FFT, 100), 128);
    EXPECT_EQ(faf_get_recommended_size(FAF_TRANSFORM_FFT, 1000), 1024);
}

/* Test error handling */
TEST(CoreTest, ErrorHandling) {
    faf_clear_error();
    EXPECT_STREQ(faf_get_error(), "");
    
    /* Create an invalid transform to trigger error */
    faf_transform* t = faf_create_fft(3, false, FAF_PREC_FP32, 0);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(faf_get_error(), "");
    
    faf_clear_error();
    EXPECT_STREQ(faf_get_error(), "");
}

/* Test transform creation and destruction */
TEST(CoreTest, TransformLifecycle) {
    /* Valid FFT creation */
    faf_transform* t = faf_create_fft(64, false, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid inverse FFT */
    t = faf_create_fft(64, true, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid DCT */
    t = faf_create_dct(64, 2, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid DST */
    t = faf_create_dst(64, 2, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid MDCT */
    t = faf_create_mdct(64, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid Haar wavelet */
    t = faf_create_haar(64, 0, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid Daubechies-4 */
    t = faf_create_daubechies4(64, 0, FAF_PREC_FP32, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Null destruction should not crash */
    faf_destroy_transform(nullptr);
}

/* Test double precision transform */
TEST(CoreTest, DoublePrecisionTransform) {
    faf_transform* t = faf_create_fft(64, false, FAF_PREC_FP64, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        EXPECT_EQ(t->precision, FAF_PREC_FP64);
        faf_destroy_transform(t);
    }
}
