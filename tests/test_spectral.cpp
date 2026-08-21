/**
 * @file test_spectral.cpp
 * @brief Fourier-domain C callbacks and fused R2C pipelines
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct NotchCtx {
    size_t bin;
};

static void notch_bin(float *re, float *im, size_t n_bins, void *ctx) {
    size_t k = ((NotchCtx *)ctx)->bin;
    if (k < n_bins) {
        re[k] = 0.0f;
        im[k] = 0.0f;
    }
}

class SpectralTest : public ::testing::Test {
protected:
    NotchCtx notch_{};
    void TearDown() override { chirp_cleanup(); }
};

TEST_F(SpectralTest, RegisterAndNotchBin) {
    const size_t n = 64;
    const size_t nb = n / 2 + 1;
    notch_.bin = 5;
    ASSERT_GE(chirp_register_spectral("notch", notch_bin, &notch_), 0);

    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 64) (spectral notch) (irfft))");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_PIPELINE);

    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = cosf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n) +
               0.4f * cosf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    }
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);

    /* Reconstruct: bin 5 gone, bin 3 remains. */
    faf_config rc = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&rc);
    ASSERT_NE(fwd, nullptr);
    std::vector<float> re(nb), im(nb);
    faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), nb);
    faf_buffer yb = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &yb), 0);
    EXPECT_NEAR(hypotf(re[5], im[5]), 0.0f, 1e-3f);
    EXPECT_GT(hypotf(re[3], im[3]), 5.0f);

    EXPECT_NE(t->jit_cache, nullptr) << "fused spectral pipeline should JIT";

    faf_destroy_transform(fwd);
    faf_destroy_transform(t);
}

TEST_F(SpectralTest, UnaryStillWorks) {
    chirp_register_standard_builtins();
    faf_transform *t = chirp_compile("(sin)");
    ASSERT_NE(t, nullptr);
    float in[128] = {0}, out[128] = {0};
    for (size_t i = 0; i < t->n; i++) {
        in[2 * i] = (float)i * 0.1f;
        in[2 * i + 1] = 0.0f;
    }
    EXPECT_EQ(faf_execute_f32(t, out, in), 0);
    EXPECT_NEAR(out[0], sinf(0.0f), 1e-5f);
    EXPECT_NEAR(out[2], sinf(0.1f), 1e-5f);
    faf_destroy_transform(t);
}

TEST_F(SpectralTest, Bandpass) {
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 32) (bandpass :lo 4 :hi 4) (irfft))");
    ASSERT_NE(t, nullptr);
    const size_t n = 32;
    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = cosf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n) +
               cosf(2.0f * (float)M_PI * 7.0f * (float)i / (float)n);
    }
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);

    faf_config rc = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&rc);
    std::vector<float> re(n / 2 + 1), im(n / 2 + 1);
    faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), n / 2 + 1);
    faf_buffer yb = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &yb), 0);
    EXPECT_GT(hypotf(re[4], im[4]), 8.0f);
    EXPECT_NEAR(hypotf(re[7], im[7]), 0.0f, 1e-3f);
    faf_destroy_transform(fwd);
    faf_destroy_transform(t);
}

TEST_F(SpectralTest, MulSpectrumConvolve) {
    const size_t n = 32;
    const size_t nb = n / 2 + 1;
    faf_config c = faf_config_init(n);
    faf_transform *rf = faf_create_rfft(&c);
    ASSERT_NE(rf, nullptr);

    std::vector<float> h(n, 0.0f), Hr(nb), Hi(nb);
    h[0] = 1.0f; /* identity kernel */
    faf_buffer hin = faf_buffer_real(h.data(), n);
    faf_buffer hs = faf_buffer_hermitian(Hr.data(), Hi.data(), nb);
    ASSERT_EQ(faf_execute(rf, &hs, &hin), 0);
    faf_destroy_transform(rf);

    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 32) (mul-spectrum) (irfft))");
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(chirp_bind(t, "H", Hr.data(), Hi.data(), nb), 0);

    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 2.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 1e-4f) << "i=" << i;
    faf_destroy_transform(t);
}

TEST_F(SpectralTest, ConjPipeline) {
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 16) conj (irfft))");
    ASSERT_NE(t, nullptr);
    const size_t n = 16;
    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++)
        x[i] = (float)((int)i - 8);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    /* conj in frequency is time-reversal of a real signal (x[0] fixed). */
    EXPECT_NEAR(y[0], x[0], 1e-4f);
    for (size_t i = 1; i < n; i++)
        EXPECT_NEAR(y[i], x[n - i], 1e-4f) << "i=" << i;
    faf_destroy_transform(t);
}
