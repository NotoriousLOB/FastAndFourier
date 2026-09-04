/**
 * @file test_sizes.cpp
 * @brief Size policy: 5-smooth FFT/RFFT, dyadic DWT, opt-in Bluestein
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "faf.h"

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

TEST(SizePolicy, FftAcceptsFiveSmooth) {
    const size_t sizes[] = {3840, 4000, 4050, 4096};
    for (size_t n : sizes) {
        EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, n)) << n;
        faf_config c = faf_config_init(n);
        c.layout = FAF_LAYOUT_SPLIT;
        faf_transform *t = faf_create_fft(&c);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<float> xr(n, 0.0f), xi(n, 0.0f), yr(n), yi(n);
        xr[0] = 1.0f;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << n;
        EXPECT_NEAR(yr[0], 1.0f, 2e-3f) << n;
        faf_destroy_transform(t);
    }
}

TEST(SizePolicy, FftRejectsNonFiveSmoothWithoutFlag) {
    /* 7 is a codelet; 11 is Rader. 14 and 23 stay illegal without Bluestein. */
    faf_config c14 = faf_config_init(14);
    EXPECT_EQ(faf_create_fft(&c14), nullptr);
    EXPECT_STRNE(faf_get_error(), "");

    faf_config c23 = faf_config_init(23);
    EXPECT_EQ(faf_create_fft(&c23), nullptr);

    faf_config c4097 = faf_config_init(4097);
    EXPECT_EQ(faf_create_fft(&c4097), nullptr);
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 14));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 23));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 4097));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 7));
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 11));
}

TEST(SizePolicy, RfftRejectsOddIncludingFiveSmooth) {
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 15)); /* 3*5 */
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 9));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 25));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_RFFT, 27));
    faf_config c = faf_config_init(15);
    EXPECT_EQ(faf_create_rfft(&c), nullptr);
}

TEST(SizePolicy, DwtRejects3840) {
    faf_config c = faf_config_init(3840);
    c.family = FAF_WAVELET_HAAR;
    EXPECT_EQ(faf_create_dwt(&c), nullptr);
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_HAAR, 3840));

    faf_transform *t = chirp_compile("(dwt :family haar :size 3840)");
    EXPECT_EQ(t, nullptr);
    EXPECT_STRNE(faf_get_error(), "");
}

TEST(SizePolicy, RecommendedSize3000) {
    size_t n = faf_get_recommended_size(FAF_TRANSFORM_FFT, 3000);
    EXPECT_GE(n, 3000u);
    EXPECT_TRUE(faf_is_5_smooth(n));
    EXPECT_EQ(n, 3000u); /* 3000 = 2^3 · 3 · 5^3 */
    size_t nr = faf_get_recommended_size(FAF_TRANSFORM_RFFT, 3000);
    EXPECT_GE(nr, 3000u);
    EXPECT_EQ(nr % 2u, 0u);
    EXPECT_TRUE(faf_is_5_smooth(nr));
}

TEST(SizePolicy, ChirpFft3840) {
    faf_transform *t = chirp_compile("(fft :size 3840 :layout split)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 3840u);
    faf_destroy_transform(t);

    faf_transform *r = chirp_compile("(rfft :size 3840 :layout hermitian)");
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->n, 3840u);
    faf_destroy_transform(r);
}

static faf_transform *make_bluestein(size_t n, bool inverse, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    c.precision = prec;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.flags = FAF_FLAG_BLUESTEIN;
    return faf_create_fft(&c);
}

TEST(Bluestein, MatchesNaiveDft) {
    const size_t sizes[] = {23, 47, 307};
    for (size_t n : sizes) {
        faf_transform *t = make_bluestein(n, false, FAF_PREC_FP32);
        ASSERT_NE(t, nullptr) << "n=" << n;
        EXPECT_NE(t->flags & FAF_FLAG_BLUESTEIN, 0u);
        EXPECT_NE(t->inner, nullptr);
        EXPECT_GE(t->inner->n, 2u * n - 1u);
        EXPECT_GT(t->scratch_size, 0u);

        std::vector<float> xr(n), xi(n, 0.0f), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sinf(2.0f * (float)M_PI * 2.0f * (float)i / (float)n) +
                    0.2f * (float)((int)(i % 5) - 2);

        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;

        std::vector<float> dr, di;
        naive_dft(xr.data(), xi.data(), n, &dr, &di, 0);
        /* Bound ~ O(ε n log n); f32 ε≈1e-7, n=307 → a few 1e-4. */
        float tol = (n >= 200) ? 5e-3f : 1e-3f;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], dr[k], tol) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], di[k], tol) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST(Bluestein, Roundtrip) {
    const size_t n = 47;
    faf_transform *fwd = make_bluestein(n, false, FAF_PREC_FP32);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    EXPECT_NE(inv->flags & FAF_FLAG_BLUESTEIN, 0u);

    std::vector<float> xr(n), xi(n, 0.0f), sr(n), si(n), yr(n), yi(n);
    for (size_t i = 0; i < n; i++)
        xr[i] = cosf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
    faf_buffer spec = faf_buffer_split(sr.data(), si.data(), n);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(yr[i], xr[i], 2e-4f) << i;
        EXPECT_NEAR(yi[i], 0.0f, 2e-4f) << i;
    }
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST(Bluestein, RfftRefused) {
    faf_config c = faf_config_init(64);
    c.flags = FAF_FLAG_BLUESTEIN;
    EXPECT_EQ(faf_create_rfft(&c), nullptr);
    EXPECT_STRNE(faf_get_error(), "");

    faf_config odd = faf_config_init(307);
    odd.flags = FAF_FLAG_BLUESTEIN;
    EXPECT_EQ(faf_create_rfft(&odd), nullptr);
}

TEST(Bluestein, DwtRefused) {
    faf_config c = faf_config_init(64);
    c.family = FAF_WAVELET_HAAR;
    c.flags = FAF_FLAG_BLUESTEIN;
    EXPECT_EQ(faf_create_dwt(&c), nullptr);
}

TEST(Bluestein, InterleavedExecute) {
    const size_t n = 23;
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_INTERLEAVED;
    c.flags = FAF_FLAG_BLUESTEIN;
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);
    std::vector<float> in(2 * n, 0.0f), out(2 * n, 0.0f);
    in[0] = 1.0f;
    ASSERT_EQ(faf_execute_f32(t, out.data(), in.data()), 0);
    for (size_t k = 0; k < n; k++) {
        EXPECT_NEAR(out[2 * k], 1.0f, 1e-4f) << k;
        EXPECT_NEAR(out[2 * k + 1], 0.0f, 1e-4f) << k;
    }
    faf_destroy_transform(t);
}

TEST(Bluestein, ChirpKeyword) {
    faf_transform *no = chirp_compile("(fft :size 23)");
    EXPECT_EQ(no, nullptr);

    faf_transform *t = chirp_compile("(fft :size 23 :bluestein :layout split)");
    ASSERT_NE(t, nullptr);
    EXPECT_NE(t->flags & FAF_FLAG_BLUESTEIN, 0u);
    EXPECT_EQ(t->n, 23u);

    std::vector<float> xr(23, 0.0f), xi(23, 0.0f), yr(23), yi(23);
    xr[0] = 1.0f;
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 23);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 23);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t k = 0; k < 23; k++) {
        EXPECT_NEAR(yr[k], 1.0f, 1e-4f) << k;
        EXPECT_NEAR(yi[k], 0.0f, 1e-4f) << k;
    }
    faf_destroy_transform(t);

    faf_transform *r = chirp_compile("(rfft :size 32 :bluestein)");
    EXPECT_EQ(r, nullptr);
}

TEST(Bluestein, FourZeroNineSeven) {
    faf_config c = faf_config_init(4097);
    EXPECT_EQ(faf_create_fft(&c), nullptr);
    c.flags = FAF_FLAG_BLUESTEIN;
    c.layout = FAF_LAYOUT_SPLIT;
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);
    std::vector<float> xr(4097, 0.0f), xi(4097, 0.0f), yr(4097), yi(4097);
    xr[0] = 1.0f;
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 4097);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 4097);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    EXPECT_NEAR(yr[0], 1.0f, 5e-3f);
    EXPECT_NEAR(yr[100], 1.0f, 5e-3f);
    faf_destroy_transform(t);
}