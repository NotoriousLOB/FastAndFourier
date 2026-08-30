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

TEST(LayoutTest, HermMulDcAndNyquistStayReal) {
    const size_t nb = 5; /* n = 8 */
    float xr[5] = {2, 1, 0, 3, 4};
    float xi[5] = {7, 1, 0, 2, 9}; /* DC/Nyquist imag should be ignored */
    float hr[5] = {3, 1, 0, 0, 5};
    float hi[5] = {8, 0, 0, 1, 6};
    float yr[5], yi[5];
    faf_herm_mul_f32(yr, yi, xr, xi, hr, hi, nb);
    EXPECT_NEAR(yr[0], 6.0f, 1e-6f);  /* 2*3, imag dropped */
    EXPECT_NEAR(yi[0], 0.0f, 1e-6f);
    EXPECT_NEAR(yr[4], 20.0f, 1e-6f); /* 4*5 */
    EXPECT_NEAR(yi[4], 0.0f, 1e-6f);
    EXPECT_NEAR(yr[1], 1.0f, 1e-6f);  /* (1+i)(1+0i) */
    EXPECT_NEAR(yi[1], 1.0f, 1e-6f);
}

TEST(LayoutTest, HermMulConj) {
    float xr[3] = {1, 2, 3}, xi[3] = {0, 4, 0};
    float hr[3] = {5, 1, 7}, hi[3] = {0, 1, 0};
    float yr[3], yi[3];
    faf_herm_mul_conj_f32(yr, yi, xr, xi, hr, hi, 3);
    EXPECT_NEAR(yr[0], 5.0f, 1e-6f);
    EXPECT_NEAR(yi[0], 0.0f, 1e-6f);
    EXPECT_NEAR(yr[1], 2.0f * 1.0f + 4.0f * 1.0f, 1e-6f); /* 6 */
    EXPECT_NEAR(yi[1], 4.0f * 1.0f - 2.0f * 1.0f, 1e-6f); /* 2 */
    EXPECT_NEAR(yr[2], 21.0f, 1e-6f);
    EXPECT_NEAR(yi[2], 0.0f, 1e-6f);
}

static void naive_full_corr(const float *x, const float *h, size_t n,
                            std::vector<float> *y) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    faf_transform *fwd = faf_create_fft(&c);
    faf_transform *inv = faf_create_inverse(fwd);
    std::vector<float> xr(x, x + n), xi(n, 0.0f), hr(h, h + n), hi(n, 0.0f);
    std::vector<float> Xr(n), Xi(n), Hr(n), Hi(n), Yr(n), Yi(n);
    faf_buffer xin = faf_buffer_split(xr.data(), xi.data(), n);
    faf_buffer hin = faf_buffer_split(hr.data(), hi.data(), n);
    faf_buffer X = faf_buffer_split(Xr.data(), Xi.data(), n);
    faf_buffer H = faf_buffer_split(Hr.data(), Hi.data(), n);
    ASSERT_EQ(faf_execute(fwd, &X, &xin), 0);
    ASSERT_EQ(faf_execute(fwd, &H, &hin), 0);
    for (size_t k = 0; k < n; k++) {
        float ar = Xr[k], ai = Xi[k], br = Hr[k], bi = Hi[k];
        Yr[k] = ar * br + ai * bi;
        Yi[k] = ai * br - ar * bi;
    }
    faf_buffer Y = faf_buffer_split(Yr.data(), Yi.data(), n);
    y->assign(n, 0.0f);
    std::vector<float> yim(n, 0.0f);
    faf_buffer out = faf_buffer_split(y->data(), yim.data(), n);
    ASSERT_EQ(faf_execute(inv, &out, &Y), 0);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

static void packed_corr(const float *x, const float *h, size_t n,
                        std::vector<float> *y) {
    const size_t nb = n / 2 + 1;
    faf_config c = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&c);
    faf_transform *inv = faf_create_inverse(fwd);
    std::vector<float> Xr(nb), Xi(nb), Hr(nb), Hi(nb);
    std::vector<float> xv(x, x + n), hv(h, h + n);
    faf_buffer xin = faf_buffer_real(xv.data(), n);
    faf_buffer hin = faf_buffer_real(hv.data(), n);
    faf_buffer Xs = faf_buffer_hermitian(Xr.data(), Xi.data(), nb);
    faf_buffer Hs = faf_buffer_hermitian(Hr.data(), Hi.data(), nb);
    ASSERT_EQ(faf_execute(fwd, &Xs, &xin), 0);
    ASSERT_EQ(faf_execute(fwd, &Hs, &hin), 0);
    EXPECT_NEAR(Xi[0], 0.0f, 1e-5f);
    EXPECT_NEAR(Xi[nb - 1], 0.0f, 1e-5f);
    faf_herm_mul_conj_f32(Xr.data(), Xi.data(), Xr.data(), Xi.data(),
                          Hr.data(), Hi.data(), nb);
    EXPECT_NEAR(Xi[0], 0.0f, 1e-5f);
    EXPECT_NEAR(Xi[nb - 1], 0.0f, 1e-5f);
    y->assign(n, 0.0f);
    faf_buffer out = faf_buffer_real(y->data(), n);
    ASSERT_EQ(faf_execute(inv, &out, &Xs), 0);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST(LayoutTest, PackedCorrMatchesFullSpectrum_DC) {
    const size_t n = 32;
    std::vector<float> x(n, 1.0f), h(n, 1.0f), packed, full;
    packed_corr(x.data(), h.data(), n, &packed);
    naive_full_corr(x.data(), h.data(), n, &full);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(packed[i], full[i], 2e-4f) << i;
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(packed[i], (float)n, 2e-3f) << i;
}

TEST(LayoutTest, PackedCorrMatchesFullSpectrum_Nyquist) {
    const size_t n = 32;
    std::vector<float> x(n), h(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = (i & 1u) ? -1.0f : 1.0f;
        h[i] = x[i];
    }
    std::vector<float> packed, full;
    packed_corr(x.data(), h.data(), n, &packed);
    naive_full_corr(x.data(), h.data(), n, &full);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(packed[i], full[i], 2e-4f) << i;
}

TEST(LayoutTest, PackedCorrMatchesFullSpectrum_Tone) {
    const size_t n = 32;
    const int bin = 5;
    std::vector<float> x(n), h(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = cosf(2.0f * (float)M_PI * (float)bin * (float)i / (float)n);
        h[i] = x[i];
    }
    std::vector<float> packed, full;
    packed_corr(x.data(), h.data(), n, &packed);
    naive_full_corr(x.data(), h.data(), n, &full);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(packed[i], full[i], 2e-4f) << i;
    float peak = packed[0];
    for (size_t i = 1; i < n; i++)
        EXPECT_LT(fabsf(packed[i]), peak + 1e-3f);
}

struct CorrCtx {
    float *Hr;
    float *Hi;
};

static void corr_spectral(float *re, float *im, size_t n_bins, void *ctx) {
    CorrCtx *c = (CorrCtx *)ctx;
    faf_herm_mul_conj_f32(re, im, re, im, c->Hr, c->Hi, n_bins);
}

TEST(LayoutTest, SplitPlaneCorrelatorBuiltin) {
    const size_t n = 32;
    const size_t nb = n / 2 + 1;
    std::vector<float> tmpl(n, 0.0f), x(n, 0.0f);
    for (size_t i = 0; i < 8; i++)
        tmpl[i] = 1.0f;
    for (size_t i = 0; i < 8; i++)
        x[10 + i] = 1.0f;

    faf_config c = faf_config_init(n);
    faf_transform *rf = faf_create_rfft(&c);
    std::vector<float> Hr(nb), Hi(nb);
    faf_buffer tin = faf_buffer_real(tmpl.data(), n);
    faf_buffer hs = faf_buffer_hermitian(Hr.data(), Hi.data(), nb);
    ASSERT_EQ(faf_execute(rf, &hs, &tin), 0);
    faf_destroy_transform(rf);

    CorrCtx ctx{Hr.data(), Hi.data()};
    ASSERT_GE(chirp_register_spectral_ex("corr", corr_spectral, nullptr, &ctx), 0);
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 32) (spectral corr) (irfft))");
    ASSERT_NE(t, nullptr);
    std::vector<float> y(n);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    size_t peak = 0;
    float best = y[0];
    for (size_t i = 1; i < n; i++) {
        if (y[i] > best) { best = y[i]; peak = i; }
    }
    EXPECT_EQ(peak, 10u);
    faf_destroy_transform(t);
    chirp_cleanup();
}
