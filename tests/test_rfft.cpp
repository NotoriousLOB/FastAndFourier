/**
 * @file test_rfft.cpp
 * @brief Correctness tests for first-class real FFT (R2C / C2R)
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "faf_test_util.h"
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static faf_transform *make_rfft(size_t n, bool inverse, faf_precision prec,
                                faf_layout layout = FAF_LAYOUT_HERMITIAN,
                                faf_norm norm = FAF_NORM_NONE) {
    faf_config c = faf_config_init(n);
    c.precision = prec;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.layout = layout;
    c.norm = norm;
    return faf_create_rfft(&c);
}

class RfftF32 : public ::testing::Test {
protected:
    static void naive_dft_real(const float *x, size_t n,
                               std::vector<float> *re,
                               std::vector<float> *im) {
        re->assign(n, 0.0f);
        im->assign(n, 0.0f);
        for (size_t k = 0; k < n; k++) {
            double sr = 0.0, si = 0.0;
            for (size_t t = 0; t < n; t++) {
                double ang = -2.0 * M_PI * (double)k * (double)t / (double)n;
                sr += (double)x[t] * cos(ang);
                si += (double)x[t] * sin(ang);
            }
            (*re)[k] = (float)sr;
            (*im)[k] = (float)si;
        }
    }
};

TEST(RfftApi, DefaultsAndSpectrumLen) {
    faf_config c = faf_config_init(64);
    faf_transform *t = faf_create_rfft(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_RFFT);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_HERMITIAN);
    EXPECT_EQ(t->cfg.norm, FAF_NORM_NONE);
    EXPECT_EQ(t->n, 64u);
    EXPECT_EQ(faf_spectrum_len(t), 33u);
    EXPECT_NE(t->inner, nullptr);
    EXPECT_EQ(t->inner->n, 32u);
    EXPECT_EQ(t->inner->cfg.layout, FAF_LAYOUT_SPLIT);
    EXPECT_EQ(t->inner->cfg.norm, FAF_NORM_NONE);

    faf_transform *inv = faf_create_inverse(t);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->type, FAF_TRANSFORM_IRFFT);
    EXPECT_EQ(inv->cfg.dir, FAF_DIR_INVERSE);
    EXPECT_EQ(inv->cfg.n, 64u);
    EXPECT_EQ(inv->cfg.layout, FAF_LAYOUT_HERMITIAN);
    EXPECT_EQ(faf_spectrum_len(inv), 33u);

    faf_destroy_transform(inv);
    faf_destroy_transform(t);
}

TEST(RfftApi, RejectsOddAndNonPow2) {
    faf_config odd = faf_config_init(15);
    EXPECT_EQ(faf_create_rfft(&odd), nullptr);
    EXPECT_STRNE(faf_get_error(), "");

    faf_config even = faf_config_init(14); /* 2*7, not 5-smooth */
    EXPECT_EQ(faf_create_rfft(&even), nullptr);

    faf_config split = faf_config_init(32);
    split.layout = FAF_LAYOUT_SPLIT;
    EXPECT_EQ(faf_create_rfft(&split), nullptr);

    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 2));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 1024));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 1));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 3));
}

TEST(RfftApi, ExecuteRejectsWrongLayouts) {
    faf_transform *t = make_rfft(32, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);
    float x[32] = {0};
    float re[17] = {0}, im[17] = {0};
    faf_buffer in = faf_buffer_split(re, im, 32);
    faf_buffer out = faf_buffer_hermitian(re, im, 17);
    EXPECT_NE(faf_execute(t, &out, &in), 0);

    in = faf_buffer_real(x, 32);
    EXPECT_EQ(faf_execute(t, &out, &in), 0);
    faf_destroy_transform(t);
}

TEST_F(RfftF32, ImpulseIsFlatHermitian) {
    const size_t n = 64;
    const size_t nb = n / 2 + 1;
    faf_transform *t = make_rfft(n, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);

    std::vector<float> x(n, 0.0f), re(nb), im(nb);
    x[0] = 1.0f;
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_hermitian(re.data(), im.data(), nb);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);

    for (size_t k = 0; k < nb; k++) {
        EXPECT_NEAR(re[k], 1.0f, 1e-5f) << "bin " << k;
        EXPECT_NEAR(im[k], 0.0f, 1e-5f) << "bin " << k;
    }
    faf_destroy_transform(t);
}

TEST_F(RfftF32, CosineEnergyInBinK) {
    const size_t n = 64;
    const size_t nb = n / 2 + 1;
    const int bin = 7;
    faf_transform *t = make_rfft(n, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);

    std::vector<float> x(n), re(nb), im(nb);
    for (size_t i = 0; i < n; i++)
        x[i] = cosf(2.0f * (float)M_PI * (float)bin * (float)i / (float)n);

    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_hermitian(re.data(), im.data(), nb);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);

    EXPECT_NEAR(im[0], 0.0f, 1e-5f);
    EXPECT_NEAR(im[n / 2], 0.0f, 1e-5f);

    float peak = hypotf(re[bin], im[bin]);
    EXPECT_NEAR(peak, (float)n / 2.0f, 1e-3f);
    for (size_t k = 0; k < nb; k++) {
        if (k == (size_t)bin) continue;
        float mag = hypotf(re[k], im[k]);
        EXPECT_LT(mag, 1e-3f) << "leak at bin " << k;
    }
    faf_destroy_transform(t);
}

TEST_F(RfftF32, MatchesC2COnRealInput) {
    const size_t n = 32;
    const size_t nb = n / 2 + 1;
    faf_transform *r2c = make_rfft(n, false, FAF_PREC_FP32);
    faf_config cc = faf_config_init(n);
    cc.layout = FAF_LAYOUT_SPLIT;
    faf_transform *c2c = faf_create_fft(&cc);
    ASSERT_NE(r2c, nullptr);
    ASSERT_NE(c2c, nullptr);

    std::vector<float> x(n), zre(n, 0.0f), zim(n, 0.0f);
    std::vector<float> cre(n), cim(n), rre(nb), rim(nb);
    for (size_t i = 0; i < n; i++) {
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n) +
               0.25f * cosf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n);
        zre[i] = x[i];
    }

    faf_buffer rin = faf_buffer_real(x.data(), n);
    faf_buffer rout = faf_buffer_hermitian(rre.data(), rim.data(), nb);
    faf_buffer cin = faf_buffer_split(zre.data(), zim.data(), n);
    faf_buffer cout = faf_buffer_split(cre.data(), cim.data(), n);
    ASSERT_EQ(faf_execute(r2c, &rout, &rin), 0);
    ASSERT_EQ(faf_execute(c2c, &cout, &cin), 0);

    for (size_t k = 0; k < nb; k++) {
        EXPECT_NEAR(rre[k], cre[k], 1e-4f) << "re bin " << k;
        EXPECT_NEAR(rim[k], cim[k], 1e-4f) << "im bin " << k;
    }
    faf_destroy_transform(r2c);
    faf_destroy_transform(c2c);
}

TEST_F(RfftF32, NaiveDftBins) {
    const size_t n = 16;
    const size_t nb = n / 2 + 1;
    faf_transform *t = make_rfft(n, false, FAF_PREC_FP32);
    ASSERT_NE(t, nullptr);

    std::vector<float> x(n), re(nb), im(nb);
    for (size_t i = 0; i < n; i++)
        x[i] = (float)i - 7.5f;

    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_hermitian(re.data(), im.data(), nb);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);

    std::vector<float> dre, dim;
    naive_dft_real(x.data(), n, &dre, &dim);
    for (size_t k = 0; k < nb; k++) {
        EXPECT_NEAR(re[k], dre[k], 1e-4f) << "re bin " << k;
        EXPECT_NEAR(im[k], dim[k], 1e-4f) << "im bin " << k;
    }
    faf_destroy_transform(t);
}

TEST_F(RfftF32, RoundtripNoneNorm) {
    const size_t sizes[] = {2, 4, 16, 32, 64, 256, 1024, 4096};
    for (size_t n : sizes) {
        faf_transform *fwd = make_rfft(n, false, FAF_PREC_FP32);
        faf_transform *inv = faf_create_inverse(fwd);
        ASSERT_NE(fwd, nullptr) << "n=" << n;
        ASSERT_NE(inv, nullptr) << "n=" << n;

        size_t nb = n / 2 + 1;
        std::vector<float> x(n), y(n), re(nb), im(nb);
        for (size_t i = 0; i < n; i++)
            x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n) +
                   0.3f * cosf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n);

        faf_buffer in = faf_buffer_real(x.data(), n);
        faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), nb);
        faf_buffer out = faf_buffer_real(y.data(), n);
        ASSERT_EQ(faf_execute(fwd, &spec, &in), 0) << "n=" << n;
        ASSERT_EQ(faf_execute(inv, &out, &spec), 0) << "n=" << n;

        float tol = (n >= 1024) ? 2e-4f : 1e-4f;
        for (size_t i = 0; i < n; i++)
            EXPECT_NEAR(y[i], x[i], tol) << "n=" << n << " i=" << i;

        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }
}

TEST(RfftF64, RoundtripNoneNorm) {
    const size_t n = 256;
    const size_t nb = n / 2 + 1;
    faf_transform *fwd = make_rfft(n, false, FAF_PREC_FP64);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);

    std::vector<double> x(n), y(n), re(nb), im(nb);
    for (size_t i = 0; i < n; i++)
        x[i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);

    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), nb);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 1e-12) << "i=" << i;

    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST_F(RfftF32, OrthoRoundtrip) {
    const size_t n = 64;
    const size_t nb = n / 2 + 1;
    faf_transform *fwd = make_rfft(n, false, FAF_PREC_FP32,
                                   FAF_LAYOUT_HERMITIAN, FAF_NORM_ORTHO);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);

    std::vector<float> x(n), y(n), re(nb), im(nb);
    for (size_t i = 0; i < n; i++)
        x[i] = (float)((int)(i % 7) - 3);

    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), nb);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 1e-4f) << "i=" << i;

    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST_F(RfftF32, InterleavedPackedRoundtrip) {
    const size_t n = 32;
    const size_t nb = n / 2 + 1;
    faf_transform *fwd = make_rfft(n, false, FAF_PREC_FP32,
                                   FAF_LAYOUT_INTERLEAVED);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(fwd->cfg.layout, FAF_LAYOUT_INTERLEAVED);

    std::vector<float> x(n), y(n), packed(2 * nb);
    for (size_t i = 0; i < n; i++)
        x[i] = cosf(2.0f * (float)M_PI * 2.0f * (float)i / (float)n);

    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer spec = faf_buffer_interleaved(packed.data(), nb);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    EXPECT_NEAR(packed[1], 0.0f, 1e-5f);           /* DC imag */
    EXPECT_NEAR(packed[2 * (nb - 1) + 1], 0.0f, 1e-5f); /* Nyquist imag */
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 1e-4f) << "i=" << i;

    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST(RfftChirp, CompileStandalone) {
    faf_transform *t = chirp_compile("(rfft :size 64)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_RFFT);
    EXPECT_EQ(t->n, 64u);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_HERMITIAN);

    const size_t n = 64, nb = 33;
    std::vector<float> x(n, 0.0f), re(nb), im(nb);
    x[0] = 1.0f;
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_hermitian(re.data(), im.data(), nb);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    EXPECT_NEAR(re[0], 1.0f, 1e-5f);
    faf_destroy_transform(t);

    faf_transform *inv = chirp_compile("(irfft :size 64 :norm none)");
    ASSERT_NE(inv, nullptr);
    EXPECT_EQ(inv->type, FAF_TRANSFORM_IRFFT);
    faf_destroy_transform(inv);

    faf_transform *pipe = chirp_compile("(pipeline (rfft :size 32))");
    ASSERT_NE(pipe, nullptr);
    EXPECT_EQ(pipe->type, FAF_TRANSFORM_RFFT);
    EXPECT_EQ(pipe->n, 32u);
    faf_destroy_transform(pipe);
}

TEST(RfftChirp, RoundtripPipeline) {
    faf_transform *t = chirp_compile("(pipeline (rfft :size 64) (irfft))");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_PIPELINE);

    const size_t n = 64;
    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 1e-4f) << "i=" << i;
    faf_destroy_transform(t);
}

TEST(RfftChirp, MixedWithFftIrRejected) {
    faf_transform *t = chirp_compile("(pipeline (rfft :size 64) twiddle)");
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(faf_get_error(), "");
}

TEST(RfftChirp, NormLayoutPrecision) {
    faf_transform *t = chirp_compile(
        "(rfft :size 32 :norm ortho :layout hermitian :precision f32)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->cfg.norm, FAF_NORM_ORTHO);
    EXPECT_EQ(t->cfg.layout, FAF_LAYOUT_HERMITIAN);
    EXPECT_EQ(t->precision, FAF_PREC_FP32);

    const size_t n = 32, nb = 17;
    std::vector<float> x(n, 0.0f), re(nb), im(nb);
    x[0] = 1.0f;
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_hermitian(re.data(), im.data(), nb);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    float s = 1.0f / sqrtf((float)n);
    EXPECT_NEAR(re[0], s, 1e-5f);
    EXPECT_NEAR(im[0], 0.0f, 1e-5f);
    EXPECT_NEAR(im[nb - 1], 0.0f, 1e-5f);
    faf_destroy_transform(t);

    faf_transform *il = chirp_compile("(rfft :size 16 :layout interleaved)");
    ASSERT_NE(il, nullptr);
    EXPECT_EQ(il->cfg.layout, FAF_LAYOUT_INTERLEAVED);
    faf_destroy_transform(il);

    faf_transform *d = chirp_compile("(rfft :size 16 :precision f64)");
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->precision, FAF_PREC_FP64);
    faf_destroy_transform(d);
}

TEST(RfftChirp, RejectsJpeg2000Norm) {
    faf_transform *t = chirp_compile("(rfft :size 32 :norm jpeg2000)");
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(faf_get_error(), "");

    faf_transform *lazy = chirp_compile("(rfft :size 32 :norm lazy)");
    EXPECT_EQ(lazy, nullptr);
}

TEST(RfftChirp, InverseInheritsNormLayout) {
    faf_transform *pipe = chirp_compile(
        "(pipeline (rfft :size 64 :norm ortho :layout hermitian) (inverse))");
    ASSERT_NE(pipe, nullptr);
    EXPECT_EQ(pipe->type, FAF_TRANSFORM_PIPELINE);
    const size_t n = 64;
    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(pipe, &out, &in), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 2e-4f) << i;
    faf_destroy_transform(pipe);
}
