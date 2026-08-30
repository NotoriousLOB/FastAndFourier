/**
 * @file test_cwt_lp.cpp
 * @brief CWT filter bank: constructor rejection, LP certification, CQT geometry
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "fastandfourier.h"
}

/* ---- Constructor rejection ---- */

TEST(CWTBank, RejectOddN) {
    faf_cwt_config cfg = faf_cwt_config_init(1023);
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(std::strstr(faf_get_error(), "5-smooth"), nullptr);
}

TEST(CWTBank, RejectNon5Smooth) {
    faf_cwt_config cfg = faf_cwt_config_init(14);  /* 2*7 */
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
}

TEST(CWTBank, RejectN0) {
    faf_cwt_config cfg = faf_cwt_config_init(0);
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
}

TEST(CWTBank, RejectVoicesZero) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.voices = 0;
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(std::strstr(faf_get_error(), "voices"), nullptr);
}

TEST(CWTBank, RejectFminGeFmax) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.fmin = 100.0;
    cfg.fmax = 10.0;
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(std::strstr(faf_get_error(), "fmin >= fmax"), nullptr);
}

TEST(CWTBank, RejectFP16) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP16;
    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
    EXPECT_NE(std::strstr(faf_get_error(), "unsupported"), nullptr);
}

TEST(CWTBank, NullConfig) {
    faf_transform *t = faf_create_cwt(nullptr);
    EXPECT_EQ(t, nullptr);
}

/* ---- LP certification ---- */

TEST(CWTLP, MorseDefaultPasses) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr) << faf_get_error();

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->passed, 1);
    EXPECT_LE(rep->max_abs_dev, 0.05);
    EXPECT_LE(rep->mean_abs_dev, 0.05);
    EXPECT_GT(rep->admissibility_C, 0.0);
    EXPECT_EQ(rep->has_lowpass, 1);
    EXPECT_NEAR(cfg.morse_beta, 20.0, 0.0);
    EXPECT_LE(rep->max_wrap_energy, 1e-3);
    EXPECT_LT(rep->frame_cond, 1.2);

    faf_cwt_report_fprint(stdout, rep);

    faf_destroy_transform(t);
}

TEST(CWTLP, MorseTimeBandwidth180Passes) {
    /* ssqueezepy-style Morse β=60, γ=3 → P²=180, narrower than MATLAB. */
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.morse_gamma = 3.0;
    cfg.morse_beta = 60.0;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr) << faf_get_error();

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->passed, 1);

    faf_destroy_transform(t);
}

TEST(CWTLP, MorletPasses) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MORLET;
    cfg.morlet_mu = 6.0;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;
    cfg.lp_alpha = 0.10;
    cfg.lp_beta = 0.10;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->passed, 1);
    EXPECT_LE(rep->max_abs_dev, 0.10);

    faf_destroy_transform(t);
}

TEST(CWTLP, LinearScalesFail) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.scale_kind = FAF_CWT_SCALE_LINEAR;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;
    cfg.flags &= ~FAF_CWT_FLAG_ALLOW_UNTILED;

    faf_transform *t = faf_create_cwt(&cfg);
    EXPECT_EQ(t, nullptr);
}

TEST(CWTLP, LinearScalesAllowUntiled) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.scale_kind = FAF_CWT_SCALE_LINEAR;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;
    cfg.flags |= FAF_CWT_FLAG_ALLOW_UNTILED;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->passed, 0);

    faf_destroy_transform(t);
}

/* ---- Analyticity / DC / phase ---- */

TEST(CWTLP, WaveletDCIsZero) {
    faf_cwt_config cfg = faf_cwt_config_init(2048);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_LT(rep->max_dc_wavelet, 1e-8);

    faf_destroy_transform(t);
}

TEST(CWTLP, LowpassDCPositive) {
    faf_cwt_config cfg = faf_cwt_config_init(2048);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const double *phi = faf_cwt_phi_f64(t);
    ASSERT_NE(phi, nullptr);
    EXPECT_GT(phi[0], 0.0);

    faf_destroy_transform(t);
}

/* ---- CQT geometry ---- */

TEST(CWTLP, GeometricFreqRatios) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    size_t J = faf_cwt_n_scales(t);
    ASSERT_GT(J, 2u);

    double *freqs = new double[J];
    faf_cwt_freqs(t, freqs, J);

    double expected_ratio = std::pow(2.0, 1.0 / 10.0);
    for (size_t j = 0; j + 1 < J; j++) {
        double ratio = freqs[j + 1] / freqs[j];
        EXPECT_NEAR(ratio, expected_ratio, 1e-12);
    }

    delete[] freqs;
    faf_destroy_transform(t);
}

TEST(CWTLP, AHasNoNaN) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    ASSERT_NE(rep->A, nullptr);

    for (size_t k = 0; k < rep->n_bins; k++) {
        EXPECT_FALSE(std::isnan(rep->A[k])) << "A[" << k << "] is NaN";
    }

    faf_destroy_transform(t);
}

/* ---- Query functions ---- */

TEST(CWTLP, QueryFunctions) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    EXPECT_GT(faf_cwt_n_scales(t), 0u);
    EXPECT_EQ(faf_cwt_n_rows(t), faf_cwt_n_scales(t) + (size_t)faf_cwt_has_lowpass(t));
    EXPECT_EQ(faf_cwt_has_lowpass(t), 1);
    EXPECT_EQ(faf_cwt_n_bins(t), 4096 / 2 + 1);

    double freqs[8];
    int nwritten = faf_cwt_freqs(t, freqs, 8);
    EXPECT_EQ(nwritten, 8);

    faf_destroy_transform(t);
}

TEST(CWTLP, MeyerCreates) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.wavelet = FAF_CWT_WAVELET_MEYER;
    cfg.precision = FAF_PREC_FP64;
    cfg.lp_alpha = 0.15;
    cfg.lp_beta = 0.15;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr) << faf_get_error();

    const double *psi = faf_cwt_psi_f64(t, 0);
    ASSERT_NE(psi, nullptr);
    EXPECT_NEAR(psi[0], 0.0, 1e-15);

    /* Meyer is not the bump placeholder: peak sits near 4π/3 in proto units. */
    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_GT(rep->admissibility_C, 0.0);

    faf_destroy_transform(t);
}

TEST(CWTLP, EnergyCenterDiffersFromPeak) {
    faf_cwt_config peak = faf_cwt_config_init(4096);
    peak.precision = FAF_PREC_FP64;
    peak.morse_beta = 20.0;
    peak.center_kind = FAF_CWT_CENTER_PEAK;

    faf_cwt_config energy = peak;
    energy.center_kind = FAF_CWT_CENTER_ENERGY;

    faf_transform *tp = faf_create_cwt(&peak);
    faf_transform *te = faf_create_cwt(&energy);
    ASSERT_NE(tp, nullptr) << faf_get_error();
    ASSERT_NE(te, nullptr) << faf_get_error();

    size_t J = faf_cwt_n_scales(tp);
    ASSERT_EQ(J, faf_cwt_n_scales(te));
    ASSERT_GT(J, 0u);

    double *sp = new double[J];
    double *se = new double[J];
    ASSERT_GT(faf_cwt_scales(tp, sp, J), 0);
    ASSERT_GT(faf_cwt_scales(te, se, J), 0);
    EXPECT_GT(std::fabs(sp[J / 2] - se[J / 2]), 1e-6);

    delete[] sp;
    delete[] se;
    faf_destroy_transform(tp);
    faf_destroy_transform(te);
}

TEST(CWTBank, NullIcwtConfig) {
    EXPECT_EQ(faf_create_icwt(nullptr, FAF_CWT_INV_DUAL), nullptr);
}

TEST(CWTLP, FP32BankCreates) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP32;

    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    const faf_cwt_lp_report *rep = faf_cwt_bank_report(t);
    ASSERT_NE(rep, nullptr);
    EXPECT_EQ(rep->passed, 1);

    const float *psi0 = faf_cwt_psi_f32(t, 0);
    ASSERT_NE(psi0, nullptr);

    faf_destroy_transform(t);
}
