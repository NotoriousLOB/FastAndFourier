/**
 * @file test_fft_kernels.cpp
 * @brief Small-N codelets, split-radix DIF, execute scratch, NEON vs scalar
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "faf.h"
#include "chirp.h"

#include <cmath>
#include <cstring>
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
            double xt = xr[t], xit = xi ? xi[t] : 0.0;
            sr += xt * wr - xit * wi;
            si += xt * wi + xit * wr;
        }
        (*yr)[k] = sr;
        (*yi)[k] = si;
    }
}

static faf_transform *make_split_fft(size_t n, bool inverse, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.norm = FAF_NORM_NONE;
    c.precision = prec;
    return faf_create_fft(&c);
}

static void fill_tone_f32(std::vector<float> *re, std::vector<float> *im,
                          size_t n, int k) {
    re->assign(n, 0.0f);
    im->assign(n, 0.0f);
    for (size_t i = 0; i < n; i++) {
        double ang = 2.0 * M_PI * (double)k * (double)i / (double)n;
        (*re)[i] = (float)cos(ang);
        (*im)[i] = (float)sin(ang);
    }
}

class FftKernel : public ::testing::Test {};

TEST_F(FftKernel, DispatchIsCodelet) {
    for (size_t n : {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 12u}) {
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP32);
        ASSERT_NE(t, nullptr) << "n=" << n;
        EXPECT_EQ(t->execute_func, faf_fft_kernel_execute) << "n=" << n;
        EXPECT_EQ(t->code, nullptr);
        EXPECT_EQ(t->n_inst, 0u);
        EXPECT_NE(t->scratch, nullptr);
        faf_destroy_transform(t);
    }
    faf_transform *t16 = make_split_fft(16, false, FAF_PREC_FP32);
    ASSERT_NE(t16, nullptr);
    EXPECT_EQ(t16->execute_func, faf_fft_sr_dif_execute);
    EXPECT_EQ(t16->code, nullptr);
    faf_destroy_transform(t16);
}

TEST_F(FftKernel, N2N4N8_impulse_f32) {
    for (size_t n : {2u, 4u, 8u}) {
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP32);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<float> xr(n, 0.0f), xi(n, 0.0f), yr(n), yi(n);
        xr[0] = 1.0f;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], 1.0f, 1e-5f) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], 0.0f, 1e-5f) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(FftKernel, N2N4N8_tone_f32) {
    for (size_t n : {2u, 4u, 8u}) {
        int bin = (n == 2) ? 1 : 1;
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP32);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<float> xr, xi, yr(n), yi(n);
        fill_tone_f32(&xr, &xi, n, bin);
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        for (size_t k = 0; k < n; k++) {
            float ere = (k == (size_t)bin) ? (float)n : 0.0f;
            EXPECT_NEAR(yr[k], ere, 2e-5f) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], 0.0f, 2e-5f) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(FftKernel, N2N4N8_roundtrip_f32) {
    for (size_t n : {2u, 4u, 8u}) {
        faf_transform *fwd = make_split_fft(n, false, FAF_PREC_FP32);
        faf_transform *inv = faf_create_inverse(fwd);
        ASSERT_NE(fwd, nullptr);
        ASSERT_NE(inv, nullptr);
        std::vector<float> xr(n), xi(n, 0.0f), sr(n), si(n), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sinf(2.0f * (float)M_PI * (float)i / (float)n) + 0.25f * (float)i;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer spec = faf_buffer_split(sr.data(), si.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
        ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
        for (size_t i = 0; i < n; i++) {
            EXPECT_NEAR(yr[i], xr[i], 1e-5f) << "n=" << n << " i=" << i;
            EXPECT_NEAR(yi[i], 0.0f, 1e-5f) << "n=" << n << " i=" << i;
        }
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }
}

TEST_F(FftKernel, N2N4N8_impulse_f64) {
    for (size_t n : {2u, 4u, 8u}) {
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP64);
        ASSERT_NE(t, nullptr) << "n=" << n;
        std::vector<double> xr(n, 0.0), xi(n, 0.0), yr(n), yi(n);
        xr[0] = 1.0;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], 1.0, 1e-12) << "n=" << n << " k=" << k;
            EXPECT_NEAR(yi[k], 0.0, 1e-12) << "n=" << n << " k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST_F(FftKernel, DirectKernelMatchesNaive) {
    const size_t n = 8;
    std::vector<float> re(n), im(n);
    for (size_t i = 0; i < n; i++) {
        re[i] = sinf(0.3f * (float)i) + 0.1f * (float)i;
        im[i] = cosf(0.2f * (float)i);
    }
    std::vector<double> xr(n), xi(n);
    for (size_t i = 0; i < n; i++) {
        xr[i] = re[i];
        xi[i] = im[i];
    }
    std::vector<double> dr, di;
    naive_dft_f64(xr.data(), xi.data(), n, &dr, &di, 0);
    faf_fft_kernel_8_f32(re.data(), im.data(), 0);
    for (size_t k = 0; k < n; k++) {
        EXPECT_NEAR(re[k], (float)dr[k], 2e-5f) << "re k=" << k;
        EXPECT_NEAR(im[k], (float)di[k], 2e-5f) << "im k=" << k;
    }
}

TEST_F(FftKernel, ChirpSize8HitsKernel) {
    faf_transform *t = chirp_compile("(fft :size 8 :layout split)");
    ASSERT_NE(t, nullptr) << (faf_get_error() ? faf_get_error() : "none");
    EXPECT_EQ(t->execute_func, faf_fft_kernel_execute);
    std::vector<float> xr(8, 0.0f), xi(8, 0.0f), yr(8), yi(8);
    xr[0] = 1.0f;
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), 8);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), 8);
    ASSERT_EQ(faf_execute(t, &out, &in), 0);
    for (size_t k = 0; k < 8; k++)
        EXPECT_NEAR(yr[k], 1.0f, 1e-5f) << k;
    faf_destroy_transform(t);
}

TEST(SplitRadix, MatchesNaiveDft) {
    const size_t sizes[] = {16, 32, 64, 256, 1024};
    for (size_t n : sizes) {
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP64);
        ASSERT_NE(t, nullptr) << "n=" << n;
        EXPECT_EQ(t->execute_func, faf_fft_sr_dif_execute) << "n=" << n;
        std::vector<double> xr(n), xi(n, 0.0), yr(n), yi(n);
        for (size_t i = 0; i < n; i++)
            xr[i] = sin(2.0 * M_PI * 3.0 * (double)i / (double)n) +
                    0.15 * (double)((int)(i % 7) - 3);
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0) << "n=" << n;
        /* Full naive DFT is O(n^2); 1024 is ~1e6 ops, fine. */
        std::vector<double> dr, di;
        naive_dft_f64(xr.data(), xi.data(), n, &dr, &di, 0);
        double tol = (n >= 256) ? 2e-9 : 1e-10;
        for (size_t k = 0; k < n; k++) {
            EXPECT_NEAR(yr[k], dr[k], tol) << "n=" << n << " re k=" << k;
            EXPECT_NEAR(yi[k], di[k], tol) << "n=" << n << " im k=" << k;
        }
        faf_destroy_transform(t);
    }
}

TEST(SplitRadix, InverseRoundtrip) {
    const size_t n = 64;
    faf_transform *fwd = make_split_fft(n, false, FAF_PREC_FP64);
    faf_transform *inv = faf_create_inverse(fwd);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(inv, nullptr);
    std::vector<double> xr(n), xi(n, 0.0), sr(n), si(n), yr(n), yi(n);
    for (size_t i = 0; i < n; i++)
        xr[i] = cos(2.0 * M_PI * 5.0 * (double)i / (double)n);
    faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
    faf_buffer spec = faf_buffer_split(sr.data(), si.data(), n);
    faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
    ASSERT_EQ(faf_execute(fwd, &spec, &in), 0);
    ASSERT_EQ(faf_execute(inv, &out, &spec), 0);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(yr[i], xr[i], 1e-12) << i;
        EXPECT_NEAR(yi[i], 0.0, 1e-12) << i;
    }
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

TEST(Execute, NoAllocOnHotPath) {
    for (size_t n : {8u, 64u, 256u}) {
        faf_transform *t = make_split_fft(n, false, FAF_PREC_FP32);
        ASSERT_NE(t, nullptr) << "n=" << n;
        ASSERT_NE(t->scratch, nullptr);
        size_t before = t->scratch_size;
        EXPECT_GT(before, 0u);
        std::vector<float> xr(n, 0.0f), xi(n, 0.0f), yr(n), yi(n);
        xr[0] = 1.0f;
        faf_buffer in = faf_buffer_split(xr.data(), xi.data(), n);
        faf_buffer out = faf_buffer_split(yr.data(), yi.data(), n);
        ASSERT_EQ(faf_execute(t, &out, &in), 0);
        EXPECT_EQ(t->scratch_size, before) << "n=" << n;
        /* Interleaved convenience path also uses create-time scratch. */
        faf_config c = faf_config_init(n);
        c.layout = FAF_LAYOUT_INTERLEAVED;
        faf_transform *ti = faf_create_fft(&c);
        ASSERT_NE(ti, nullptr);
        size_t ibefore = ti->scratch_size;
        std::vector<float> xin(2 * n, 0.0f), xout(2 * n);
        xin[0] = 1.0f;
        ASSERT_EQ(faf_execute_f32(ti, xout.data(), xin.data()), 0);
        EXPECT_EQ(ti->scratch_size, ibefore) << "interleaved n=" << n;
        faf_destroy_transform(ti);
        faf_destroy_transform(t);
    }
}

static void scalar_radix2_f32(float *re, float *im, size_t n, size_t group,
                              size_t stride, size_t tw_step,
                              const float *tw, size_t ntw) {
    size_t half = group / 2;
    if (half == 0 || group == 0) return;
    size_t ngroups = n / (group * stride);
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group * stride;
        for (size_t r = 0; r < half; r++) {
            size_t i1 = base + r * stride;
            size_t i2 = i1 + half * stride;
            float ar = re[i1], ai = im[i1];
            float br = re[i2], bi = im[i2];
            if (tw) {
                size_t idx = r * tw_step;
                if (idx < ntw) {
                    float wr = tw[2 * idx], wi = tw[2 * idx + 1];
                    float tr = br * wr - bi * wi;
                    float ti = br * wi + bi * wr;
                    br = tr;
                    bi = ti;
                }
            }
            re[i1] = ar + br;
            im[i1] = ai + bi;
            re[i2] = ar - br;
            im[i2] = ai - bi;
        }
    }
}

TEST(FftStage, NeonAgreesWithScalar) {
    const size_t n = 64;
    const size_t group = 32;
    std::vector<float> tw(n);
    faf_gen_twiddles_f32(tw.data(), n, false);
    std::vector<float> re(n), im(n), re_s(n), im_s(n);
    for (size_t i = 0; i < n; i++) {
        re[i] = sinf(0.17f * (float)i);
        im[i] = cosf(0.11f * (float)i);
        re_s[i] = re[i];
        im_s[i] = im[i];
    }
    faf_fft_stage_split_f32(re.data(), im.data(), n, (uint32_t)group, 1u, 1u,
                            tw.data(), n / 2, 0);
    scalar_radix2_f32(re_s.data(), im_s.data(), n, group, 1, 1,
                      tw.data(), n / 2);
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(re[i], re_s[i], 1e-6f) << "re i=" << i;
        EXPECT_NEAR(im[i], im_s[i], 1e-6f) << "im i=" << i;
    }
}
