/**
 * @file test_dwt_backends.cpp
 * @brief Backend L vs F agreement, AUTO rule, custom taps, analysis-only
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <string.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
#include "fastandfourier.h"
#include "faf.h"
#include "chirp.h"
}

#include "faf_test_util.h"

#define ALIGNED_ALLOC64(n) aligned_alloc(64, (((size_t)(n)) + 63u) & ~(size_t)63u)

static double max_abs_re_f64(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[2 * i] - b[2 * i]);
        if (e > m) m = e;
    }
    return m;
}

static double max_abs_f64(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static faf_transform *dwt_backend(faf_wavelet_family fam, size_t n, size_t levels,
                                  bool inverse, faf_dwt_backend backend,
                                  faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.family = fam;
    c.levels = levels;
    c.precision = prec;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    c.dwt_backend = backend;
    return faf_create_dwt(&c);
}

class DwtBackendTest : public ::testing::Test {
protected:
    void SetUp() override { faf_init(); }
};

TEST_F(DwtBackendTest, AutoSelectsLiftForBuiltins) {
    faf_config c = faf_config_init(64);
    c.family = FAF_WAVELET_HAAR;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    faf_transform *t = faf_create_dwt(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.dwt_backend, FAF_DWT_BACKEND_LIFT);
    faf_destroy_transform(t);

    c.family = FAF_WAVELET_CDF97;
    t = faf_create_dwt(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.dwt_backend, FAF_DWT_BACKEND_LIFT);
    faf_destroy_transform(t);
}

TEST_F(DwtBackendTest, AutoSelectsFirForCustomPr) {
    faf_config c = faf_config_init(64);
    c.family = FAF_WAVELET_HAAR;
    c.conv = FAF_CONV_CUSTOM_PR;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    faf_transform *t = faf_create_dwt(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.dwt_backend, FAF_DWT_BACKEND_FIR);
    faf_destroy_transform(t);
}

TEST_F(DwtBackendTest, Cdf53IntPlusFirIsCreateError) {
    faf_config c = faf_config_init(64);
    c.family = FAF_WAVELET_CDF53;
    c.conv = FAF_CONV_CDF53_INT;
    c.dwt_backend = FAF_DWT_BACKEND_FIR;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    EXPECT_EQ(faf_create_dwt(&c), nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "FIR"), nullptr) << err;
}

TEST_F(DwtBackendTest, CustomPrPlusLiftIsCreateError) {
    faf_config c = faf_config_init(64);
    c.conv = FAF_CONV_CUSTOM_PR;
    c.dwt_backend = FAF_DWT_BACKEND_LIFT;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    EXPECT_EQ(faf_create_dwt(&c), nullptr);
}

TEST_F(DwtBackendTest, AnalysisOnlyHasNoInverse) {
    faf_config c = faf_config_init(64);
    c.conv = FAF_CONV_ANALYSIS_ONLY;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    faf_transform *t = faf_create_dwt(&c);
    ASSERT_NE(t, nullptr);
    float h[2] = {0.70710678f, 0.70710678f};
    float g[2] = {0.70710678f, -0.70710678f};
    ASSERT_EQ(faf_dwt_set_taps(t, h, g, 2, nullptr, nullptr, 0), 0);
    EXPECT_EQ(t->cfg.conv, FAF_CONV_ANALYSIS_ONLY);
    EXPECT_EQ(faf_create_inverse(t), nullptr);
    const char *err = faf_get_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(strstr(err, "analysis-only"), nullptr) << err;
    faf_destroy_transform(t);
}

struct BackendFamily {
    faf_wavelet_family fam;
    const char *name;
};

static BackendFamily kAgree[] = {
    {FAF_WAVELET_HAAR, "haar"},
    {FAF_WAVELET_D4,   "d4"},
    {FAF_WAVELET_SYM4, "sym4"},
};

TEST_F(DwtBackendTest, LiftVsFirVsRefOneLevel) {
    const size_t n = 64;
    for (auto spec : kAgree) {
        double x[64], lift[64], fir[64], ref[64];
        for (size_t i = 0; i < n; i++)
            x[i] = sin(2.0 * M_PI * 3.0 * (double)i / (double)n) + 0.1 * (double)i;
        memcpy(lift, x, sizeof(lift));
        memcpy(fir, x, sizeof(fir));
        memcpy(ref, x, sizeof(ref));

        faf_dwt_level_conv_f64(lift, n, spec.fam, FAF_CONV_UNSPEC, 0);

        double h[8], g[8], ht[8], gt[8];
        int lh = 0, ls = 0;
        ASSERT_EQ(faf_dwt_builtin_taps_f64(spec.fam, FAF_CONV_UNSPEC,
                                           h, g, &lh, ht, gt, &ls), 0)
            << spec.name;
        faf_dwt_polyfir_fwd_f64(fir, n, h, g, lh, nullptr);
        faf_dwt_ref_conv_decim_f64(ref, n, h, g, lh);

        EXPECT_LT(max_abs_f64(lift, fir, n), 1e-12) << spec.name << " L vs F";
        EXPECT_LT(max_abs_f64(fir, ref, n), 1e-15) << spec.name << " F vs ref";
    }
}

TEST_F(DwtBackendTest, LiftVsFirExecuteRoundtrip) {
    const size_t n = 128;
    const size_t levels = 4;
    for (auto spec : kAgree) {
        faf_transform *fwd_l = dwt_backend(spec.fam, n, levels, false,
                                           FAF_DWT_BACKEND_LIFT, FAF_PREC_FP64);
        faf_transform *fwd_f = dwt_backend(spec.fam, n, levels, false,
                                           FAF_DWT_BACKEND_FIR, FAF_PREC_FP64);
        ASSERT_NE(fwd_l, nullptr) << spec.name;
        ASSERT_NE(fwd_f, nullptr) << spec.name;
        EXPECT_EQ(fwd_l->cfg.dwt_backend, FAF_DWT_BACKEND_LIFT);
        EXPECT_EQ(fwd_f->cfg.dwt_backend, FAF_DWT_BACKEND_FIR);

        double *in = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
        double *out_l = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
        double *out_f = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            in[2 * i] = sin(2.0 * M_PI * 5.0 * (double)i / (double)n);
            in[2 * i + 1] = 0.0;
        }
        ASSERT_EQ(faf_execute_f64(fwd_l, out_l, in), 0);
        ASSERT_EQ(faf_execute_f64(fwd_f, out_f, in), 0);
        EXPECT_LT(max_abs_re_f64(out_l, out_f, n), 1e-11) << spec.name << " fwd";

        faf_transform *inv_f = faf_create_inverse(fwd_f);
        ASSERT_NE(inv_f, nullptr);
        double *recon = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
        ASSERT_EQ(faf_execute_f64(inv_f, recon, out_f), 0);
        EXPECT_LT(max_abs_re_f64(recon, in, n), 1e-11) << spec.name << " fir recon";

        free(in); free(out_l); free(out_f); free(recon);
        faf_destroy_transform(fwd_l);
        faf_destroy_transform(fwd_f);
        faf_destroy_transform(inv_f);
    }
}

TEST_F(DwtBackendTest, ImpulseAlignmentPeriodic) {
    const size_t n = 32;
    const size_t idxs[] = {0, n - 1};
    for (auto spec : kAgree) {
        for (size_t idx : idxs) {
            faf_transform *l = dwt_backend(spec.fam, n, 1, false,
                                           FAF_DWT_BACKEND_LIFT, FAF_PREC_FP64);
            faf_transform *f = dwt_backend(spec.fam, n, 1, false,
                                           FAF_DWT_BACKEND_FIR, FAF_PREC_FP64);
            ASSERT_NE(l, nullptr);
            ASSERT_NE(f, nullptr);
            double *in = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
            double *ol = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
            double *of = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
            memset(in, 0, 2 * n * sizeof(double));
            in[2 * idx] = 1.0;
            ASSERT_EQ(faf_execute_f64(l, ol, in), 0);
            ASSERT_EQ(faf_execute_f64(f, of, in), 0);
            EXPECT_LT(max_abs_re_f64(ol, of, n), 1e-12)
                << spec.name << " impulse " << idx;
            free(in); free(ol); free(of);
            faf_destroy_transform(l);
            faf_destroy_transform(f);
        }
    }
}

TEST_F(DwtBackendTest, CustomPrHaarLazyRoundtrip) {
    const size_t n = 64;
    faf_config c = faf_config_init(n);
    c.conv = FAF_CONV_CUSTOM_PR;
    c.levels = 3;
    c.precision = FAF_PREC_FP64;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    faf_transform *fwd = faf_create_dwt(&c);
    ASSERT_NE(fwd, nullptr);
    EXPECT_EQ(fwd->cfg.dwt_backend, FAF_DWT_BACKEND_FIR);

    const float h[2] = {1.0f, 1.0f};
    const float g[2] = {1.0f, -1.0f};
    const float ht[2] = {0.5f, 0.5f};
    const float gt[2] = {0.5f, -0.5f};
    ASSERT_EQ(faf_dwt_set_taps(fwd, h, g, 2, ht, gt, 2), 0);

    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->custom_len_syn, 2);

    double *in = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
    double *mid = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
    double *out = (double *)ALIGNED_ALLOC64(2 * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        in[2 * i] = (double)(i + 1);
        in[2 * i + 1] = 0.0;
    }
    ASSERT_EQ(faf_execute_f64(fwd, mid, in), 0);
    ASSERT_EQ(faf_execute_f64(inv, out, mid), 0);
    EXPECT_LT(max_abs_re_f64(out, in, n), 1e-6);

    free(in); free(mid); free(out);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST_F(DwtBackendTest, ChirpBackendFir) {
    faf_transform *t = chirp_compile(
        "(dwt :family haar :size 64 :levels 2 :backend fir)");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(t->cfg.dwt_backend, FAF_DWT_BACKEND_FIR);
    faf_destroy_transform(t);
}

TEST_F(DwtBackendTest, ChirpCustomTapsBind) {
    static const float h[2] = {1.0f, 1.0f};
    static const float g[2] = {1.0f, -1.0f};
    static const float ht[2] = {0.5f, 0.5f};
    static const float gt[2] = {0.5f, -0.5f};
    ASSERT_EQ(chirp_register_vector("h", h, 2), 0);
    ASSERT_EQ(chirp_register_vector("g", g, 2), 0);
    ASSERT_EQ(chirp_register_vector("ht", ht, 2), 0);
    ASSERT_EQ(chirp_register_vector("gt", gt, 2), 0);

    faf_transform *t = chirp_compile(
        "(bind lazy :h h :g g :h-syn ht :g-syn gt)\n"
        "(pipeline "
        "  (dwt :family custom :size 64 :levels 2 :backend fir "
        "       :conv custom-pr :taps lazy)"
        "  (inverse))");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(t->cfg.dwt_backend, FAF_DWT_BACKEND_FIR);
    EXPECT_EQ(t->cfg.conv, FAF_CONV_CUSTOM_PR);
    EXPECT_GT(t->scratch_size, 0u);

    float in[128] = {0}, out[128] = {0};
    for (int i = 0; i < 64; i++) in[2 * i] = (float)(i + 1);
    ASSERT_EQ(faf_execute_f32(t, out, in), 0);
    for (int i = 0; i < 64; i++)
        EXPECT_NEAR(out[2 * i], (float)(i + 1), 1e-4f) << i;
    faf_destroy_transform(t);
    chirp_cleanup();
}

TEST_F(DwtBackendTest, ChirpRegisterTapsDirect) {
    static const float h[2] = {0.70710677f, 0.70710677f};
    static const float g[2] = {0.70710677f, -0.70710677f};
    ASSERT_EQ(chirp_register_taps("haar-ortho-t", h, g, 2, h, g, 2), 0);
    faf_transform *t = chirp_compile(
        "(dwt :family custom :size 32 :levels 1 :taps haar-ortho-t "
        ":conv custom-pr :backend fir)");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_NE(t->custom_h, nullptr);
    faf_destroy_transform(t);
    chirp_cleanup();
}

TEST_F(DwtBackendTest, ChirpCustomWithoutTapsFails) {
    faf_transform *t = chirp_compile(
        "(dwt :family custom :size 32 :conv custom-pr :backend fir)");
    EXPECT_EQ(t, nullptr);
}

TEST_F(DwtBackendTest, ChirpCdf53BackendFirFails) {
    faf_transform *t = chirp_compile(
        "(dwt :family cdf53 :size 64 :backend fir)");
    EXPECT_EQ(t, nullptr);
}

TEST_F(DwtBackendTest, HaarMeanFirMatchesLift) {
    const size_t n = 32;
    double lift[32], fir[32];
    for (size_t i = 0; i < n; i++) lift[i] = fir[i] = (double)i;
    faf_dwt_level_conv_f64(lift, n, FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN, 0);
    double h[8], g[8], ht[8], gt[8];
    int lh = 0, ls = 0;
    ASSERT_EQ(faf_dwt_builtin_taps_f64(FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN,
                                       h, g, &lh, ht, gt, &ls), 0);
    faf_dwt_polyfir_fwd_f64(fir, n, h, g, lh, nullptr);
    EXPECT_LT(max_abs_f64(lift, fir, n), 1e-15);
    faf_dwt_level_conv_f64(lift, n, FAF_WAVELET_HAAR, FAF_CONV_HAAR_MEAN, 1);
    faf_dwt_polyfir_inv_f64(fir, n, ht, gt, ls, nullptr);
    EXPECT_LT(max_abs_f64(lift, fir, n), 1e-15);
}
