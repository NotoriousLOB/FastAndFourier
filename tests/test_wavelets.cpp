/**
 * @file test_wavelets.cpp
 * @brief Perfect-reconstruction and vanishing-moment tests for DWT families
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "faf_test_util.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ALIGNED_ALLOC64(n) aligned_alloc(64, (((size_t)(n)) + 63u) & ~(size_t)63u)

static faf_wavelet_family kFamilies[] = {
    FAF_WAVELET_HAAR,
    FAF_WAVELET_D4,
    FAF_WAVELET_CDF53,
    FAF_WAVELET_CDF97,
    FAF_WAVELET_SYM4,
};

static void fill_tone_f32(float *buf, size_t n, float freq) {
    for (size_t i = 0; i < n; i++) {
        buf[2 * i] = sinf(2.0f * (float)M_PI * freq * (float)i / (float)n);
        buf[2 * i + 1] = 0.0f;
    }
}

static void fill_tone_f64(double *buf, size_t n, double freq) {
    for (size_t i = 0; i < n; i++) {
        buf[2 * i] = sin(2.0 * M_PI * freq * (double)i / (double)n);
        buf[2 * i + 1] = 0.0;
    }
}

static float max_abs_re_f32(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(a[2 * i] - b[2 * i]);
        if (e > m) m = e;
    }
    return m;
}

static double max_abs_re_f64(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[2 * i] - b[2 * i]);
        if (e > m) m = e;
    }
    return m;
}

static float detail_energy_f32(const float *coef, size_t n, size_t levels) {
    size_t approx = n >> levels;
    if (approx == 0) approx = 1;
    float e = 0.0f;
    for (size_t i = approx; i < n; i++) {
        float v = coef[2 * i];
        e += v * v;
    }
    return e;
}

class WaveletTest : public ::testing::Test {
protected:
    void SetUp() override { faf_init(); }
};

TEST_F(WaveletTest, FamilyNamesAndTaps) {
    EXPECT_STREQ(faf_wavelet_name(FAF_WAVELET_HAAR), "haar");
    EXPECT_STREQ(faf_wavelet_name(FAF_WAVELET_D4), "d4");
    EXPECT_STREQ(faf_wavelet_name(FAF_WAVELET_CDF53), "cdf53");
    EXPECT_STREQ(faf_wavelet_name(FAF_WAVELET_CDF97), "cdf97");
    EXPECT_STREQ(faf_wavelet_name(FAF_WAVELET_SYM4), "sym4");
    EXPECT_EQ(faf_wavelet_taps(FAF_WAVELET_HAAR), 2);
    EXPECT_EQ(faf_wavelet_taps(FAF_WAVELET_D4), 4);
    EXPECT_EQ(faf_wavelet_taps(FAF_WAVELET_SYM4), 8);

    faf_wavelet_family f;
    EXPECT_EQ(faf_wavelet_from_name("haar", &f), 0);
    EXPECT_EQ(f, FAF_WAVELET_HAAR);
    EXPECT_EQ(faf_wavelet_from_name("daubechies4", &f), 0);
    EXPECT_EQ(f, FAF_WAVELET_D4);
    EXPECT_EQ(faf_wavelet_from_name("legall", &f), 0);
    EXPECT_EQ(f, FAF_WAVELET_CDF53);
    EXPECT_EQ(faf_wavelet_from_name("db2", &f), 0);
    EXPECT_EQ(f, FAF_WAVELET_D4);
    EXPECT_NE(faf_wavelet_from_name("db4", &f), 0);
    EXPECT_NE(faf_wavelet_from_name("nope", &f), 0);
}

TEST_F(WaveletTest, PerfectReconstructionF32) {
    const size_t sizes[] = {32, 64, 256, 1024};
    const size_t level_choices[] = {1, 3, 0};

    for (faf_wavelet_family fam : kFamilies) {
        for (size_t n : sizes) {
            for (size_t lv : level_choices) {
                faf_transform *fwd = test_dwt(fam, n, lv, false, FAF_PREC_FP32);
                ASSERT_NE(fwd, nullptr) << faf_wavelet_name(fam) << " n=" << n;
                faf_transform *inv = test_dwt(fam, n, fwd->levels, true, FAF_PREC_FP32);
                ASSERT_NE(inv, nullptr);

                float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
                float *mid = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
                float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
                fill_tone_f32(in, n, 3.0f);
                for (size_t i = 0; i < n; i++)
                    in[2 * i] += 0.15f * ((i < n / 2) ? 1.0f : -0.4f);

                ASSERT_EQ(faf_execute_vm(fwd, mid, in), 0);
                ASSERT_EQ(faf_execute_vm(inv, out, mid), 0);

                float tol = (fam == FAF_WAVELET_CDF97 || fam == FAF_WAVELET_SYM4)
                            ? 5e-5f : 1e-5f;
                float err = max_abs_re_f32(out, in, n);
                EXPECT_LT(err, tol) << faf_wavelet_name(fam)
                                    << " n=" << n << " levels=" << fwd->levels
                                    << " err=" << err;

                free(in); free(mid); free(out);
                faf_destroy_transform(fwd);
                faf_destroy_transform(inv);
            }
        }
    }
}

TEST_F(WaveletTest, PerfectReconstructionF64) {
    const size_t n = 128;
    for (faf_wavelet_family fam : kFamilies) {
        faf_transform *fwd = test_dwt(fam, n, 4, false, FAF_PREC_FP64);
        faf_transform *inv = test_dwt(fam, n, 4, true, FAF_PREC_FP64);
        ASSERT_NE(fwd, nullptr);
        ASSERT_NE(inv, nullptr);

        double *in = (double*)ALIGNED_ALLOC64(2 * n * sizeof(double));
        double *mid = (double*)ALIGNED_ALLOC64(2 * n * sizeof(double));
        double *out = (double*)ALIGNED_ALLOC64(2 * n * sizeof(double));
        fill_tone_f64(in, n, 5.0);

        ASSERT_EQ(faf_execute_f64(fwd, mid, in), 0);
        ASSERT_EQ(faf_execute_f64(inv, out, mid), 0);
        EXPECT_LT(max_abs_re_f64(out, in, n), 5e-12)
            << faf_wavelet_name(fam);

        free(in); free(mid); free(out);
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }
}

TEST_F(WaveletTest, HaarConstantVanishesInDetails) {
    const size_t n = 64;
    faf_transform *t = test_dwt(FAF_WAVELET_HAAR, n, 3, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) { in[2 * i] = 2.5f; in[2 * i + 1] = 0.0f; }
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);
    EXPECT_LT(detail_energy_f32(out, n, 3), 1e-8f);
    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(WaveletTest, Cdf53ConstantVanishesInDetails) {
    const size_t n = 64;
    faf_transform *t = test_dwt(FAF_WAVELET_CDF53, n, 2, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);
    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) { in[2 * i] = -1.25f; in[2 * i + 1] = 0.0f; }
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);
    EXPECT_LT(detail_energy_f32(out, n, 2), 1e-8f);
    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(WaveletTest, Daubechies4IsNotHaar) {
    const size_t n = 64;
    faf_transform *haar = test_haar(n, 2);
    faf_transform *d4 = test_d4(n, 2);
    ASSERT_NE(haar, nullptr);
    ASSERT_NE(d4, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *h = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *d = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    fill_tone_f32(in, n, 2.0f);

    ASSERT_EQ(faf_execute_f32(haar, h, in), 0);
    ASSERT_EQ(faf_execute_f32(d4, d, in), 0);

    float diff = max_abs_re_f32(h, d, n);
    EXPECT_GT(diff, 1e-3f) << "D4 must not collapse to Haar";

    free(in); free(h); free(d);
    faf_destroy_transform(haar);
    faf_destroy_transform(d4);
}

TEST_F(WaveletTest, RejectsBadArgs) {
    faf_clear_error();
    EXPECT_EQ(test_dwt(FAF_WAVELET_HAAR, 7, 1, false, FAF_PREC_FP32), nullptr);
    EXPECT_STRNE(faf_get_error(), "");

    faf_clear_error();
    EXPECT_EQ(test_dwt(FAF_WAVELET_HAAR, 64, 20, false, FAF_PREC_FP32), nullptr);
    EXPECT_STRNE(faf_get_error(), "");
}

TEST_F(WaveletTest, ChirpDwtMatchesCApi) {
    chirp_register_standard_builtins();
    const size_t n = 64;
    faf_transform *capi = test_dwt(FAF_WAVELET_HAAR, n, 3, false, FAF_PREC_FP32);
    faf_transform *chirp = chirp_compile("(dwt :family haar :size 64 :levels 3)");
    ASSERT_NE(capi, nullptr);
    ASSERT_NE(chirp, nullptr);
    EXPECT_EQ(chirp->n, n);
    EXPECT_EQ(chirp->levels, 3u);

    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *a = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *b = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    fill_tone_f32(in, n, 4.0f);

    ASSERT_EQ(faf_execute_f32(capi, a, in), 0);
    ASSERT_EQ(faf_execute_f32(chirp, b, in), 0);
    EXPECT_LT(max_abs_re_f32(a, b, n), 1e-5f);

    free(in); free(a); free(b);
    faf_destroy_transform(capi);
    faf_destroy_transform(chirp);
    chirp_cleanup();
}

TEST_F(WaveletTest, ChirpDenoiseImprovesSnr) {
    chirp_register_standard_builtins();
    const size_t n = 256;
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (dwt :family cdf97 :size 256 :levels 5)"
        "  (threshold :mode soft :lambda 0.25)"
        "  (idwt :family cdf97 :size 256 :levels 5))"
    );
    ASSERT_NE(t, nullptr);

    float *clean = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *noisy = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    fill_tone_f32(clean, n, 3.0f);

    /* Deterministic "noise": sparse spikes. */
    memcpy(noisy, clean, 2 * n * sizeof(float));
    noisy[2 * 20] += 1.8f;
    noisy[2 * 80] -= 2.1f;
    noisy[2 * 140] += 1.6f;

    auto snr = [&](const float *sig, const float *ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < n; i++) {
            num += (double)ref[2 * i] * ref[2 * i];
            double e = (double)sig[2 * i] - ref[2 * i];
            den += e * e;
        }
        return (den <= 1e-20) ? 200.0 : 10.0 * log10(num / den);
    };

    ASSERT_EQ(faf_execute_f32(t, out, noisy), 0);
    EXPECT_GT(snr(out, clean), snr(noisy, clean));

    free(clean); free(noisy); free(out);
    faf_destroy_transform(t);
    chirp_cleanup();
}

TEST_F(WaveletTest, JitAgreesWithVmHaar) {
    const size_t n = 256;
    faf_transform *t = test_dwt(FAF_WAVELET_HAAR, n, 4, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *vm = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    float *jit = (float*)ALIGNED_ALLOC64(2 * n * sizeof(float));
    fill_tone_f32(in, n, 6.0f);

    ASSERT_EQ(faf_execute_vm(t, vm, in), 0);
    int jret = faf_execute_jit(t, jit, in);
    if (jret == 0) {
        EXPECT_LT(max_abs_re_f32(vm, jit, n), 1e-5f);
    }

    free(in); free(vm); free(jit);
    faf_destroy_transform(t);
}

TEST_F(WaveletTest, TransformNames) {
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_CDF53), "cdf53");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_SYM4), "sym4");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_CDF97), "cdf97");
}
