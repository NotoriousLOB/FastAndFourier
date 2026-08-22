/**
 * @file test_layout.cpp
 * @brief Explicit interleave/deinterleave and Chirp :layout
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "chirp.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(LayoutTest, InterleaveRoundtripF32) {
    const size_t n = 8;
    float re[8], im[8], z[16], re2[8], im2[8];
    for (size_t i = 0; i < n; i++) {
        re[i] = (float)i;
        im[i] = -(float)i;
    }
    ASSERT_EQ(faf_interleave_f32(z, re, im, n), 0);
    EXPECT_EQ(z[0], 0.0f);
    EXPECT_EQ(z[1], 0.0f);
    EXPECT_EQ(z[2], 1.0f);
    EXPECT_EQ(z[3], -1.0f);
    ASSERT_EQ(faf_deinterleave_f32(re2, im2, z, n), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(re2[i], re[i]);
        EXPECT_EQ(im2[i], im[i]);
    }
}

TEST(LayoutTest, InterleaveRoundtripF64) {
    const size_t n = 4;
    double re[4] = {1, 2, 3, 4}, im[4] = {5, 6, 7, 8};
    double z[8], re2[4], im2[4];
    ASSERT_EQ(faf_interleave_f64(z, re, im, n), 0);
    ASSERT_EQ(faf_deinterleave_f64(re2, im2, z, n), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(re2[i], re[i]);
        EXPECT_EQ(im2[i], im[i]);
    }
}

TEST(LayoutTest, ExecuteDoesNotConvert) {
    faf_config c = faf_config_init(32);
    c.layout = FAF_LAYOUT_SPLIT;
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);

    float z[64] = {0};
    faf_buffer in = faf_buffer_interleaved(z, 32);
    faf_buffer out = faf_buffer_interleaved(z, 32);
    EXPECT_NE(faf_execute(t, &out, &in), 0);

    float re[32] = {0}, im[32] = {0};
    in = faf_buffer_split(re, im, 32);
    out = faf_buffer_split(re, im, 32);
    EXPECT_EQ(faf_execute(t, &out, &in), 0);
    faf_destroy_transform(t);
}

TEST(LayoutTest, ConvertAtEdgeThenExecute) {
    const size_t n = 32;
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);

    float interleaved[64];
    for (size_t i = 0; i < n; i++) {
        interleaved[2 * i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
        interleaved[2 * i + 1] = 0.0f;
    }
    float re[32], im[32], ore[32], oim[32], back[64];
    ASSERT_EQ(faf_deinterleave_f32(re, im, interleaved, n), 0);
    faf_buffer in = faf_buffer_split(re, im, n);
    faf_buffer out = faf_buffer_split(ore, oim, n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    ASSERT_EQ(faf_interleave_f32(back, ore, oim, n), 0);
    EXPECT_GT(hypotf(ore[3], oim[3]), 8.0f);
    faf_destroy_transform(t);
}

TEST(LayoutTest, ChirpFftSplitLayout) {
    faf_transform *t = chirp_compile("(fft :size 32 :layout split)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_SPLIT);

    float re_in[32] = {0}, im_in[32] = {0};
    float re_out[32], im_out[32];
    re_in[0] = 1.0f;
    faf_buffer in = faf_buffer_split(re_in, im_in, 32);
    faf_buffer out = faf_buffer_split(re_out, im_out, 32);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t k = 0; k < 32; k++) {
        EXPECT_NEAR(re_out[k], 1.0f, 1e-5f) << k;
        EXPECT_NEAR(im_out[k], 0.0f, 1e-5f) << k;
    }
    faf_destroy_transform(t);
}

TEST(LayoutTest, ChirpFftDefaultStaysInterleaved) {
    faf_transform *t = chirp_compile("(fft :size 16)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_INTERLEAVED);
    faf_destroy_transform(t);
}

TEST(LayoutTest, ChirpRfftLayoutHermitian) {
    faf_transform *t = chirp_compile("(rfft :size 32 :layout hermitian)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_HERMITIAN);
    faf_destroy_transform(t);
}

TEST(LayoutTest, NullConverterRejected) {
    float re[4] = {0};
    EXPECT_NE(faf_interleave_f32(nullptr, re, re, 4), 0);
    EXPECT_NE(faf_deinterleave_f32(re, re, nullptr, 4), 0);
}
