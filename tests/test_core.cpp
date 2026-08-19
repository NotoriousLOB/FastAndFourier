/**
 * @file test_core.cpp
 * @brief Tests for core library functionality
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "faf_test_util.h"
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
    faf_transform* t = test_fft_n(3);
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(faf_get_error(), "");
    
    faf_clear_error();
    EXPECT_STREQ(faf_get_error(), "");
}

/* Test transform creation and destruction */
TEST(CoreTest, TransformLifecycle) {
    /* Valid FFT creation */
    faf_transform* t = test_fft_n(64);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid inverse FFT */
    t = test_fft(64, true, FAF_PREC_FP32);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid DCT */
    t = test_dct(64, 2);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid DST */
    t = test_dst(64, 2);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid MDCT */
    t = test_mdct(64);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid Haar wavelet */
    t = test_haar(64, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Valid Daubechies-4 */
    t = test_d4(64, 0);
    EXPECT_NE(t, nullptr);
    if (t) {
        faf_destroy_transform(t);
    }
    
    /* Null destruction should not crash */
    faf_destroy_transform(nullptr);
}

/* Test double precision transform */
TEST(CoreTest, DoublePrecisionTransform) {
    faf_transform* t = test_fft(64, false, FAF_PREC_FP64);
    EXPECT_NE(t, nullptr);
    if (t) {
        EXPECT_EQ(t->precision, FAF_PREC_FP64);
        faf_destroy_transform(t);
    }
}

TEST(CoreTest, ConfigDefaultsAndSnapshot) {
    faf_config c = faf_config_init(128);
    EXPECT_EQ(c.n, 128u);
    EXPECT_EQ(c.precision, FAF_PREC_FP32);
    EXPECT_EQ(c.layout, FAF_LAYOUT_DEFAULT);
    EXPECT_EQ(c.dir, FAF_DIR_FORWARD);

    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_SPLIT);
    EXPECT_EQ(t->cfg.norm, FAF_NORM_NONE);
    EXPECT_EQ(t->n, 128u);

    faf_config snap = faf_config_from(t);
    EXPECT_EQ(snap.layout, FAF_LAYOUT_SPLIT);
    EXPECT_EQ(snap.n, 128u);

    faf_transform *inv = faf_create_inverse(t);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->cfg.dir, FAF_DIR_INVERSE);
    EXPECT_EQ(inv->cfg.n, t->cfg.n);
    EXPECT_EQ(inv->cfg.layout, t->cfg.layout);
    EXPECT_EQ(inv->cfg.norm, t->cfg.norm);
    EXPECT_EQ(inv->cfg.precision, t->cfg.precision);
    EXPECT_EQ(inv->type, FAF_TRANSFORM_IFFT);

    faf_destroy_transform(inv);
    faf_destroy_transform(t);
}

TEST(CoreTest, ExecuteRejectsLayoutMismatch) {
    faf_config c = faf_config_init(32);
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_SPLIT);

    float re[32] = {0}, im[32] = {0};
    faf_buffer in = faf_buffer_interleaved(re, 32);
    faf_buffer out = faf_buffer_interleaved(im, 32);
    EXPECT_NE(faf_execute(t, &out, &in), 0);

    in = faf_buffer_split(re, im, 32);
    out = faf_buffer_split(re, im, 32);
    EXPECT_EQ(faf_execute(t, &out, &in), 0);

    faf_destroy_transform(t);
}

TEST(CoreTest, InverseRoundtripNoneNorm) {
    const size_t n = 64;
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    c.norm = FAF_NORM_NONE;
    faf_transform *fwd = faf_create_fft(&c);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);

    float in_re[64], in_im[64], spec_re[64], spec_im[64], out_re[64], out_im[64];
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
        in_im[i] = 0.0f;
    }
    faf_buffer in = faf_buffer_split(in_re, in_im, n);
    faf_buffer spec = faf_buffer_split(spec_re, spec_im, n);
    faf_buffer out = faf_buffer_split(out_re, out_im, n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(out_re[i], in_re[i], 1e-4f) << "i=" << i;
        EXPECT_NEAR(out_im[i], 0.0f, 1e-4f) << "i=" << i;
    }
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}
