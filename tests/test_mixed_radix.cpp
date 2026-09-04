/**
 * @file test_mixed_radix.cpp
 * @brief Mixed-radix 2/3/4/5 FFT correctness vs naive DFT
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "faf.h"
#include "chirp.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void naive_dft(const float *xr, const float *xi, size_t n,
                      std::vector<float> *yr, std::vector<float> *yi,
                      int inverse) {
    yr->assign(n, 0.0f);
    yi->assign(n, 0.0f);
    double sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (size_t t = 0; t < n; t++) {
            double ang = sign * 2.0 * M_PI * (double)k * (double)t / (double)n;
            double wr = cos(ang), wi = sin(ang);
            double xr_t = xr[t], xi_t = xi ? xi[t] : 0.0;
            sr += xr_t * wr - xi_t * wi;
            si += xr_t * wi + xi_t * wr;
        }
        (*yr)[k] = (float)sr;
        (*yi)[k] = (float)si;
    }
}

class MixedRadixTest : public ::testing::Test {
protected:
    static faf_transform *make_fft(size_t n, bool inverse) {
        faf_config c = faf_config_init(n);
        c.layout = FAF_LAYOUT_SPLIT;
        c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
        c.norm = FAF_NORM_NONE;
        return faf_create_fft(&c);
    }
};

TEST_F(MixedRadixTest, Factorizer) {
    EXPECT_TRUE(faf_is_5_smooth(1));
    EXPECT_TRUE(faf_is_5_smooth(12));
    EXPECT_TRUE(faf_is_5_smooth(4000));
    EXPECT_FALSE(faf_is_5_smooth(7));
    EXPECT_FALSE(faf_is_5_smooth(14));
    EXPECT_EQ(faf_next_5_smooth(7), 8u);
    EXPECT_EQ(faf_next_5_smooth(11), 12u);

    int fac[16], nf = 0;
    ASSERT_EQ(faf_factor_5smooth(12, fac, &nf), 0);
    /* 12 = 4 * 3 */
    ASSERT_EQ(nf, 2);
    EXPECT_EQ(fac[0], 4);
    EXPECT_EQ(fac[1], 3);
}

TEST_F(MixedRadixTest, MatchesDft) {
    const size_t sizes[] = {12, 15, 20, 30, 48, 60, 192};
    for (size_t n : sizes) {
        faf_transform *t = make_fft(n, false);
        ASSERT_NE(t, nullptr) << "n=" << n;

        std::vector<float> xr(n), xi(n, 0.0f), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sinf(2.0f * (float)M_PI * 2.0f * (float)i / (float)n) +
                    0.3f * (float)((int)(i % 5) - 2);

        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;

        std::vector<float> dr, di;
        naive_dft(xr.data(), xi.data(), n, &dr, &di, 0);
        float tol = (n >= 100) ? 2e-3f : 5e-4f;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], dr[k], tol) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], di[k], tol) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(MixedRadixTest, LargerSizesMatchDftBins) {
    const size_t sizes[] = {1000, 1920, 4000};
    for (size_t n : sizes) {
        faf_transform *t = make_fft(n, false);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<float> xr(n, 0.0f), xi(n, 0.0f), yr(n), yi(n);
        xr[0] = 1.0f;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], 1.0f, 2e-3f) << "n=" << n << " k=" << k;
            EXPECT_NEAR(yi[k], 0.0f, 2e-3f) << "n=" << n << " k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(MixedRadixTest, InverseRoundtrip) {
    const size_t n = 60;
    faf_transform *fwd = make_fft(n, false);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    std::vector<float> xr(n), xi(n, 0.0f), sr(n), si(n), yr(n), yi(n);
    for (size_t i = 0; i < n; i++)
        xr[i] = cosf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
    faf_buffer spec = faf_buffer_split(sr.data(), si.data(), n);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(yr[i], xr[i], 1e-4f) << i;
        EXPECT_NEAR(yi[i], 0.0f, 1e-4f) << i;
    }
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST_F(MixedRadixTest, Pow2Unchanged) {
    const size_t sizes[] = {16, 32, 64, 256, 1024};
    for (size_t n : sizes) {
        faf_transform *t = make_fft(n, false);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<float> xr(n), xi(n, 0.0f), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        std::vector<float> dr, di;
        naive_dft(xr.data(), xi.data(), n, &dr, &di, 0);
        float tol = (n >= 256) ? 3e-3f : 5e-4f;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], dr[k], tol) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], di[k], tol) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(MixedRadixTest, RejectsPrime) {
    /* 23: prime, 22 = 2·11 is not 7-smooth → not Rader, not 5-smooth. */
    faf_config c = faf_config_init(23);
    EXPECT_EQ(faf_create_fft(&c), nullptr);
    EXPECT_STRNE(faf_get_error(), "");
}

TEST_F(MixedRadixTest, ChirpSize60) {
    faf_transform *t = chirp_compile("(fft :size 60 :layout split)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 60u);
    std::vector<float> xr(60, 0.0f), xi(60, 0.0f), yr(60), yi(60);
    xr[0] = 1.0f;
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 60);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 60);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t k = 0; k < 60; k++)
        EXPECT_NEAR(yr[k], 1.0f, 1e-4f) << k;
    faf_destroy_transform(t);
}

TEST_F(MixedRadixTest, RfftEvenFiveSmooth) {
    const size_t n = 60;
    faf_config c = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&c);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    size_t nb = n / 2 + 1;
    std::vector<float> x(n), y(n), re(nb), im(nb);
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x.data(), n);
    faf_buffer spec = faf_buffer_hermitian(re.data(), im.data(), nb);
    faf_buffer out = faf_buffer_real(y.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++)
        EXPECT_NEAR(y[i], x[i], 2e-4f) << i;
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}
