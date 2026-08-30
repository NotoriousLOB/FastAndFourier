/**
 * @file test_inverse.cpp
 * @brief Inverse inheritance and Chirp (inverse) tests
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"
}

#include "faf_test_util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float max_abs_err_f32(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ---- Inverse inheritance: config fields are copied ---- */

TEST(InverseInherit, FFT_InheritsAllFields) {
    faf_config c = faf_config_init(256);
    c.layout = FAF_LAYOUT_SPLIT;
    c.norm = FAF_NORM_ORTHO;
    c.precision = FAF_PREC_FP64;
    faf_transform *fwd = faf_create_fft(&c);
    ASSERT_NE(fwd, nullptr);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->cfg.dir, FAF_DIR_INVERSE);
    EXPECT_EQ(inv->cfg.n, fwd->cfg.n);
    EXPECT_EQ(inv->cfg.layout, fwd->cfg.layout);
    EXPECT_EQ(inv->cfg.norm, fwd->cfg.norm);
    EXPECT_EQ(inv->cfg.precision, fwd->cfg.precision);
    faf_destroy_transform(inv);
    faf_destroy_transform(fwd);
}

TEST(InverseInherit, RFFT_InheritsAllFields) {
    faf_config c = faf_config_init(512);
    c.norm = FAF_NORM_ORTHO;
    c.layout = FAF_LAYOUT_HERMITIAN;
    faf_transform *fwd = faf_create_rfft(&c);
    ASSERT_NE(fwd, nullptr);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->cfg.dir, FAF_DIR_INVERSE);
    EXPECT_EQ(inv->cfg.n, fwd->cfg.n);
    EXPECT_EQ(inv->cfg.layout, fwd->cfg.layout);
    EXPECT_EQ(inv->cfg.norm, fwd->cfg.norm);
    EXPECT_EQ(inv->cfg.precision, fwd->cfg.precision);
    faf_destroy_transform(inv);
    faf_destroy_transform(fwd);
}

static const faf_wavelet_family kDwtFamilies[] = {
    FAF_WAVELET_HAAR, FAF_WAVELET_D4, FAF_WAVELET_CDF53,
    FAF_WAVELET_CDF97, FAF_WAVELET_SYM4
};

TEST(InverseInherit, DWT_InheritsAllFields) {
    for (auto fam : kDwtFamilies) {
        faf_config c = faf_config_init(256);
        c.family = fam;
        c.levels = 3;
        faf_transform *fwd = faf_create_dwt(&c);
        ASSERT_NE(fwd, nullptr) << "family=" << (int)fam;
        faf_transform *inv = faf_create_inverse(fwd);
        ASSERT_NE(inv, nullptr) << "family=" << (int)fam;
        EXPECT_EQ(inv->cfg.dir, FAF_DIR_INVERSE);
        EXPECT_EQ(inv->cfg.n, fwd->cfg.n);
        EXPECT_EQ(inv->cfg.family, fwd->cfg.family);
        EXPECT_EQ(inv->cfg.levels, fwd->cfg.levels);
        EXPECT_EQ(inv->cfg.precision, fwd->cfg.precision);
        EXPECT_EQ(inv->cfg.conv, fwd->cfg.conv);
        EXPECT_EQ(inv->cfg.dwt_backend, fwd->cfg.dwt_backend);
        faf_destroy_transform(inv);
        faf_destroy_transform(fwd);
    }
}

/* ---- Chirp (inverse) ---- */

TEST(ChirpInverse, BareInverseWithoutPipelineFails) {
    faf_transform *t = chirp_compile("(inverse)");
    EXPECT_EQ(t, nullptr);
}

TEST(ChirpInverse, InverseWithRedundantSizeFails) {
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (dwt :family cdf97 :size 1024 :levels 5)"
        "  (inverse :size 1024))");
    EXPECT_EQ(t, nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "inherited"), nullptr)
        << "Error was: " << err;
}

TEST(ChirpInverse, DwtPipelineRoundtrip) {
    chirp_register_standard_builtins();
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (dwt :family cdf97 :size 1024 :levels 5)"
        "  (threshold :mode soft :lambda 0.08)"
        "  (inverse))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");

    float in[2048] = {0}, out[2048] = {0};
    for (int i = 0; i < 1024; i++) {
        in[2*i]   = sinf(2.0f * (float)M_PI * 3.0f * (float)i / 1024.0f);
        in[2*i+1] = 0.0f;
    }
    int rc = faf_execute_f32(t, out, in);
    EXPECT_EQ(rc, 0);
    faf_destroy_transform(t);
}

TEST(ChirpInverse, RfftPipelineRoundtrip) {
    chirp_register_standard_builtins();
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 256) (spectral notch) (inverse))");
    if (!t) {
        const char *err = faf_get_error();
        GTEST_SKIP() << "Spectral 'notch' not registered: "
                     << (err ? err : "unknown");
    }
    float rbuf_in[256] = {0}, rbuf_out[256] = {0};
    faf_buffer in  = faf_buffer_real(rbuf_in, 256);
    faf_buffer out = faf_buffer_real(rbuf_out, 256);
    EXPECT_EQ(faf_execute(t, &out, &in), 0);
    faf_destroy_transform(t);
}

TEST(ChirpInverse, LetNamedDwtRoundtrip) {
    faf_transform *t = chirp_compile(
        "(let W (dwt :family haar :size 64 :levels 3)\n"
        "  (pipeline W (inverse W)))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");
    float in[128] = {0}, out[128] = {0};
    for (int i = 0; i < 64; i++) in[2 * i] = (float)(i + 1);
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);
    for (int i = 0; i < 64; i++)
        EXPECT_NEAR(out[2 * i], (float)(i + 1), 1e-4f) << i;
    faf_destroy_transform(t);
}

TEST(ChirpInverse, LetNamedRfftRoundtrip) {
    faf_transform *t = chirp_compile(
        "(let F (rfft :size 64 :norm none :layout hermitian)\n"
        "  (pipeline F (inverse F)))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(t->type, FAF_TRANSFORM_PIPELINE);
    const size_t n = 64;
    float x[64], y[64];
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 2e-4f) << i;
    faf_destroy_transform(t);
}

TEST(ChirpInverse, TopLevelLetThenPipeline) {
    faf_transform *t = chirp_compile(
        "(let F (rfft :size 32 :layout hermitian))\n"
        "(pipeline F (inverse F))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");
    float x[32] = {0}, y[32] = {0};
    x[0] = 1.0f;
    faf_buffer in = faf_buffer_real(x, 32);
    faf_buffer out = faf_buffer_real(y, 32);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    EXPECT_NEAR(y[0], 1.0f, 1e-4f);
    faf_destroy_transform(t);
}

TEST(ChirpInverse, InverseUnknownNameFails) {
    faf_transform *t = chirp_compile(
        "(pipeline (dwt :family haar :size 32) (inverse nope))");
    EXPECT_EQ(t, nullptr);
}

TEST(ChirpInverse, InverseNamedWithSizeFails) {
    faf_transform *t = chirp_compile(
        "(let W (dwt :family haar :size 32)\n"
        "  (pipeline W (inverse W :size 32)))");
    EXPECT_EQ(t, nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "inherited"), nullptr);
}

TEST(ChirpInverse, HaarSimpleRoundtrip) {
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (dwt :family haar :size 64 :levels 3)"
        "  (inverse))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");

    float in[128] = {0}, out[128] = {0};
    for (int i = 0; i < 64; i++) {
        in[2*i] = (float)(i + 1);
        in[2*i+1] = 0.0f;
    }
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);

    float ref_re[64];
    for (int i = 0; i < 64; i++) ref_re[i] = (float)(i + 1);
    float out_re[64];
    for (int i = 0; i < 64; i++) out_re[i] = out[2*i];
    EXPECT_LT(max_abs_err_f32(out_re, ref_re, 64), 1e-4f);
    faf_destroy_transform(t);
}
