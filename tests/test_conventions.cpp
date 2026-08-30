/**
 * @file test_conventions.cpp
 * @brief Tests for wavelet convention dispatch, validation, and Haar variants
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <numeric>

extern "C" {
#include "fastandfourier.h"
#include "faf.h"
#include "chirp.h"
}

#include "faf_test_util.h"

static double norm2(const double *x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

static float max_abs_err_f32(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double max_abs_err_f64(const double *a, const double *b, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ---- Convention name round-trip ---- */

TEST(ConventionHelpers, NameRoundtrip) {
    faf_wavelet_convention conv;
    EXPECT_STREQ(faf_convention_name(FAF_CONV_HAAR_ORTHO), "haar-ortho");
    EXPECT_STREQ(faf_convention_name(FAF_CONV_CDF97_JPEG), "cdf97-jpeg");
    EXPECT_STREQ(faf_convention_name(FAF_CONV_UNSPEC), "unspec");
    EXPECT_EQ(faf_convention_from_name("haar-ortho", &conv), 0);
    EXPECT_EQ(conv, FAF_CONV_HAAR_ORTHO);
    EXPECT_EQ(faf_convention_from_name("cdf97_jpeg", &conv), 0);
    EXPECT_EQ(conv, FAF_CONV_CDF97_JPEG);
    EXPECT_EQ(faf_convention_from_name("bogus", &conv), -1);
}

/* ---- Haar ortho: energy preservation ---- */

TEST(HaarConventions, OrthoEnergyPreservation) {
    const size_t N = 256;
    double x[N], y[N];
    for (size_t i = 0; i < N; i++) x[i] = (double)(i + 1);
    double x_norm = norm2(x, N);

    memcpy(y, x, sizeof(y));
    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_ORTHO, 0);
    double y_norm = norm2(y, N);
    EXPECT_NEAR(y_norm, x_norm, x_norm * 4.5e-16)
        << "Haar ortho should preserve L2 norm";
}

/* ---- Haar lazy: roundtrip ---- */

TEST(HaarConventions, LazyRoundtrip) {
    const size_t N = 128;
    double x[N], y[N];
    for (size_t i = 0; i < N; i++) x[i] = sin(2.0 * M_PI * 3.0 * (double)i / (double)N);
    memcpy(y, x, sizeof(y));

    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_LAZY, 0);
    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_LAZY, 1);
    EXPECT_LT(max_abs_err_f64(y, x, N), 1e-12)
        << "Haar lazy should round-trip";
}

/* ---- Haar mean: DC in approximation band ---- */

TEST(HaarConventions, MeanDcInApprox) {
    const size_t N = 64;
    double x[N], y[N];
    for (size_t i = 0; i < N; i++) x[i] = 7.0;
    memcpy(y, x, sizeof(y));

    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN, 0);
    size_t half = N / 2;
    for (size_t i = 0; i < half; i++) {
        EXPECT_NEAR(y[i], 7.0, 1e-15)
            << "Mean convention: approx band should equal DC for constant signal";
    }
    double hp_dot_dc = 0.0;
    for (size_t i = 0; i < half; i++) hp_dot_dc += y[half + i];
    EXPECT_NEAR(hp_dot_dc, 0.0, 1e-13)
        << "HP inner product with DC signal should be zero";
}

/* ---- Haar mean: roundtrip ---- */

TEST(HaarConventions, MeanRoundtrip) {
    const size_t N = 128;
    double x[N], y[N];
    for (size_t i = 0; i < N; i++) x[i] = sin(2.0 * M_PI * 5.0 * (double)i / (double)N);
    memcpy(y, x, sizeof(y));

    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN, 0);
    faf_dwt_level_conv_f64(y, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN, 1);
    EXPECT_LT(max_abs_err_f64(y, x, N), 1e-12)
        << "Haar mean should round-trip";
}

/* ---- Cross-convention: ortho != lazy ---- */

TEST(HaarConventions, CrossConventionDifference) {
    const size_t N = 64;
    double a[N], b[N];
    for (size_t i = 0; i < N; i++) a[i] = b[i] = (double)(i + 1);

    faf_dwt_level_conv_f64(a, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_ORTHO, 0);
    faf_dwt_level_conv_f64(b, N, FAF_WAVELET_HAAR, FAF_CONV_HAAR_LAZY, 0);

    bool all_same = true;
    for (size_t i = 0; i < N; i++) {
        if (fabs(a[i] - b[i]) > 1e-15) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same) << "Ortho and lazy must produce different coefficients";
}

/* ---- D4/Sym4 vanishing moments: linear ramp ---- */

TEST(ConventionKernels, D4VanishingMoments) {
    const size_t N = 256;
    double x[N];
    for (size_t i = 0; i < N; i++) x[i] = 42.0;

    faf_dwt_level_conv_f64(x, N, FAF_WAVELET_D4, FAF_CONV_D4_ORTHO, 0);
    size_t half = N / 2;
    double max_detail = 0.0;
    for (size_t i = half; i < N; i++) {
        double d = fabs(x[i]);
        if (d > max_detail) max_detail = d;
    }
    EXPECT_LT(max_detail, 1e-10)
        << "D4 has 2 vanishing moments; constant detail should be ~0";
}

TEST(ConventionKernels, Sym4VanishingMoments) {
    const size_t N = 256;
    double x[N];
    for (size_t i = 0; i < N; i++) x[i] = 42.0;

    faf_dwt_level_conv_f64(x, N, FAF_WAVELET_SYM4, FAF_CONV_SYM4_ORTHO, 0);
    size_t half = N / 2;
    double max_detail = 0.0;
    for (size_t i = half; i < N; i++) {
        double d = fabs(x[i]);
        if (d > max_detail) max_detail = d;
    }
    EXPECT_LT(max_detail, 1e-10)
        << "Sym4 has 4 vanishing moments; constant detail should be ~0";
}

/* ---- CDF 5/3 integer lifting roundtrip ---- */

TEST(ConventionKernels, Cdf53IntRoundtrip) {
    const size_t N = 64;
    float x[N], y[N];
    for (size_t i = 0; i < N; i++) x[i] = (float)(i * 3 + 1);
    memcpy(y, x, sizeof(y));

    faf_dwt_level_conv_f32(y, N, FAF_WAVELET_CDF53, FAF_CONV_CDF53_INT, 0);
    faf_dwt_level_conv_f32(y, N, FAF_WAVELET_CDF53, FAF_CONV_CDF53_INT, 1);
    EXPECT_LT(max_abs_err_f32(y, x, N), 1e-4f)
        << "CDF 5/3 integer lifting should round-trip";
}

/* ---- Convention/family conflict rejected ---- */

TEST(ConventionValidation, FamilyConflictRejected) {
    faf_config c = faf_config_init(256);
    c.family = FAF_WAVELET_HAAR;
    c.conv = FAF_CONV_CDF53_INT;
    faf_transform *t = faf_create_dwt(&c);
    EXPECT_EQ(t, nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "incompatible"), nullptr) << "Error: " << err;
}

/* ---- Norm/convention conflict rejected ---- */

TEST(ConventionValidation, NormConflictRejected) {
    faf_config c = faf_config_init(256);
    c.family = FAF_WAVELET_CDF53;
    c.conv = FAF_CONV_CDF53_INT;
    c.norm = FAF_NORM_ORTHO;
    faf_transform *t = faf_create_dwt(&c);
    EXPECT_EQ(t, nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "ORTHO"), nullptr) << "Error: " << err;
}

/* ---- Convention set via C API: full DWT roundtrip ---- */

TEST(ConventionApi, HaarLazyFullDwtRoundtrip) {
    const size_t N = 256;
    faf_config c = faf_config_init(N);
    c.family = FAF_WAVELET_HAAR;
    c.conv = FAF_CONV_HAAR_LAZY;
    c.levels = 3;
    faf_transform *fwd = faf_create_dwt(&c);
    ASSERT_NE(fwd, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(fwd->cfg.conv, FAF_CONV_HAAR_LAZY);

    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->cfg.conv, FAF_CONV_HAAR_LAZY);

    float in[512] = {0}, fwd_out[512] = {0}, inv_out[512] = {0};
    for (size_t i = 0; i < N; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 7.0f * (float)i / (float)N);
    }
    ASSERT_EQ(faf_execute_f32(fwd, fwd_out, in), 0);
    ASSERT_EQ(faf_execute_f32(inv, inv_out, fwd_out), 0);

    float in_re[N], out_re[N];
    for (size_t i = 0; i < N; i++) { in_re[i] = in[2*i]; out_re[i] = inv_out[2*i]; }
    EXPECT_LT(max_abs_err_f32(out_re, in_re, N), 1e-4f);

    faf_destroy_transform(inv);
    faf_destroy_transform(fwd);
}

TEST(ConventionApi, HaarMeanFullDwtRoundtrip) {
    const size_t N = 128;
    faf_config c = faf_config_init(N);
    c.family = FAF_WAVELET_HAAR;
    c.conv = FAF_CONV_HAAR_MEAN;
    c.levels = 2;
    faf_transform *fwd = faf_create_dwt(&c);
    ASSERT_NE(fwd, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");

    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(inv, nullptr);

    float in[256] = {0}, fwd_out[256] = {0}, inv_out[256] = {0};
    for (size_t i = 0; i < N; i++) {
        in[2*i] = (float)(i + 1);
    }
    ASSERT_EQ(faf_execute_f32(fwd, fwd_out, in), 0);
    ASSERT_EQ(faf_execute_f32(inv, inv_out, fwd_out), 0);

    float in_re[N], out_re[N];
    for (size_t i = 0; i < N; i++) { in_re[i] = in[2*i]; out_re[i] = inv_out[2*i]; }
    EXPECT_LT(max_abs_err_f32(out_re, in_re, N), 1e-4f);

    faf_destroy_transform(inv);
    faf_destroy_transform(fwd);
}

/* ---- Convention set via Chirp :conv keyword ---- */

TEST(ConventionChirp, HaarLazyViaChirp) {
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (dwt :family haar :conv haar-lazy :size 128 :levels 3)"
        "  (inverse))");
    ASSERT_NE(t, nullptr) << "Error: " << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(t->cfg.conv, FAF_CONV_HAAR_LAZY);

    float in[256] = {0}, out[256] = {0};
    for (int i = 0; i < 128; i++) {
        in[2*i] = (float)(i + 1);
    }
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);

    float in_re[128], out_re[128];
    for (int i = 0; i < 128; i++) { in_re[i] = in[2*i]; out_re[i] = out[2*i]; }
    EXPECT_LT(max_abs_err_f32(out_re, in_re, 128), 1e-4f);
    faf_destroy_transform(t);
}

TEST(ConventionChirp, BadConventionFails) {
    faf_transform *t = chirp_compile(
        "(dwt :family haar :conv cdf53-int :size 64)");
    EXPECT_EQ(t, nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "incompatible"), nullptr) << "Error: " << err;
}

TEST(ConventionChirp, UnknownConventionFails) {
    faf_transform *t = chirp_compile(
        "(dwt :family haar :conv bogus-conv :size 64)");
    EXPECT_EQ(t, nullptr);
}

/* ---- Default convention resolved correctly ---- */

TEST(ConventionValidation, DefaultResolved) {
    faf_config c = faf_config_init(256);
    c.family = FAF_WAVELET_CDF97;
    faf_transform *t = faf_create_dwt(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.conv, FAF_CONV_CDF97_JPEG);
    faf_destroy_transform(t);
}
