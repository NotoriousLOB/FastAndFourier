/**
 * @file test_rader.cpp
 * @brief Rader primes, small 3/5/7 codelets, MEASURE, 7-smooth inner FFT
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

static void naive_dft_f64(const double *xr, const double *xi, size_t n,
                          std::vector<double> *yr, std::vector<double> *yi,
                          int inverse) {
    yr->assign(n, 0.0);
    yi->assign(n, 0.0);
    double sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (size_t t = 0; t < n; t++) {
            double ang = sign * 2.0 * M_PI * (double)k * (double)t / (double)n;
            double wr = cos(ang), wi = sin(ang);
            sr += xr[t] * wr - xi[t] * wi;
            si += xr[t] * wi + xi[t] * wr;
        }
        (*yr)[k] = sr;
        (*yi)[k] = si;
    }
}

static faf_transform *make_split(size_t n, bool inverse, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.norm = FAF_NORM_NONE;
    c.precision = prec;
    return faf_create_fft(&c);
}

TEST(Codelet, SmallLeavesAreCodelets) {
    for (size_t n : {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 12u}) {
        EXPECT_TRUE(faf_fft_is_codelet_size(n)) << n;
        faf_transform *t = make_split(n, false, FAF_PREC_FP64);
        ASSERT_NE(t, nullptr) << "n=" << n;
        EXPECT_EQ(t->execute_func, faf_fft_kernel_execute) << n;
        std::vector<double> xr(n, 0.0), xi(n, 0.0), yr(n), yi(n);
        xr[0] = 1.0;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << n;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], 1.0, 1e-12) << "n=" << n << " k=" << k;
            EXPECT_NEAR(yi[k], 0.0, 1e-12) << "n=" << n << " k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST(Codelet, SevenIsNotSevenSmoothPolicy) {
    EXPECT_TRUE(faf_is_size_supported(FAF_TRANSFORM_FFT, 7));
    EXPECT_FALSE(faf_is_5_smooth(7));
    EXPECT_TRUE(faf_is_7_smooth(7));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 14));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 21));
    EXPECT_FALSE(faf_is_size_supported(FAF_TRANSFORM_FFT, 49));
}

TEST(Rader, EligiblePrimes) {
    EXPECT_TRUE(faf_rader_eligible(11));
    EXPECT_TRUE(faf_rader_eligible(13));
    EXPECT_TRUE(faf_rader_eligible(17));
    EXPECT_TRUE(faf_rader_eligible(19));
    EXPECT_TRUE(faf_rader_eligible(29));  /* 28 = 4·7 */
    EXPECT_TRUE(faf_rader_eligible(31));
    EXPECT_TRUE(faf_rader_eligible(37));
    EXPECT_FALSE(faf_rader_eligible(7));  /* codelet, n < 11 */
    EXPECT_FALSE(faf_rader_eligible(23)); /* 22 = 2·11 */
    EXPECT_FALSE(faf_rader_eligible(12));
}

TEST(Rader, MatchesNaiveDft) {
    const size_t primes[] = {11, 13, 17, 19, 29, 31, 37};
    for (size_t n : primes) {
        faf_transform *t = make_split(n, false, FAF_PREC_FP64);
        ASSERT_NE(t, nullptr) << "n=" << n;
        EXPECT_EQ(t->execute_func, faf_fft_rader_execute) << n;
        std::vector<double> xr(n), xi(n, 0.0), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sin(2.0 * M_PI * 2.0 * (double)i / (double)n) +
                    0.1 * (double)((int)(i % 5) - 2);
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << n;
        std::vector<double> dr, di;
        naive_dft_f64(xr.data(), xi.data(), n, &dr, &di, 0);
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], dr[k], 1e-10) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], di[k], 1e-10) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST(Rader, InverseRoundtrip) {
    const size_t n = 31;
    faf_transform *fwd = make_split(n, false, FAF_PREC_FP64);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    std::vector<double> xr(n), xi(n, 0.0), sr(n), si(n), yr(n), yi(n);
    for (size_t i = 0; i < n; i++)
        xr[i] = cos(2.0 * M_PI * 4.0 * (double)i / (double)n);
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
    faf_buffer spec = faf_buffer_split(sr.data(), si.data(), n);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(yr[i], xr[i], 1e-10) << i;
        EXPECT_NEAR(yi[i], 0.0, 1e-10) << i;
    }
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST(Rader, ChirpPrime) {
    faf_transform *t = chirp_compile("(fft :size 17 :layout split)");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "");
    EXPECT_EQ(t->execute_func, faf_fft_rader_execute);
    std::vector<float> xr(17, 0.0f), xi(17, 0.0f), yr(17), yi(17);
    xr[0] = 1.0f;
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 17);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 17);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t k = 0; k < 17; k++)
        EXPECT_NEAR(yr[k], 1.0f, 1e-5f) << k;
    faf_destroy_transform(t);
}

TEST(Rader, InnerSevenSmooth) {
    int fac[16], nf = 0;
    ASSERT_EQ(faf_factor_7smooth(28, fac, &nf), 0);
    ASSERT_GE(nf, 2);
    EXPECT_EQ(fac[0], 4);
    EXPECT_EQ(fac[nf - 1], 7);
    EXPECT_EQ(faf_factor_5smooth(28, fac, &nf), -1);
}

TEST(Measure, OptInDoesNotCrashAndAgrees) {
    faf_config c = faf_config_init(64);
    c.layout = FAF_LAYOUT_SPLIT;
    c.norm = FAF_NORM_NONE;
    c.precision = FAF_PREC_FP64;
    c.flags = FAF_FLAG_MEASURE;
    faf_transform *t = faf_create_fft(&c);
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(t->execute_func == faf_fft_sr_dif_execute ||
                t->execute_func == faf_fft_dit_execute);
    std::vector<double> xr(64), xi(64, 0.0), yr(64), yi(64);
    for (size_t i = 0; i < 64; i++)
        xr[i] = sin(2.0 * M_PI * 3.0 * (double)i / 64.0);
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 64);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 64);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    std::vector<double> dr, di;
    naive_dft_f64(xr.data(), xi.data(), 64, &dr, &di, 0);
    for (size_t k = 0; k < 64; k++) {
        EXPECT_NEAR(yr[k], dr[k], 1e-10) << k;
        EXPECT_NEAR(yi[k], di[k], 1e-10) << k;
    }
    faf_destroy_transform(t);
}

TEST(Measure, ChirpKeyword) {
    faf_transform *t = chirp_compile("(fft :size 64 :layout split :measure)");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "");
    EXPECT_NE(t->execute_func, nullptr);
    faf_destroy_transform(t);
}
