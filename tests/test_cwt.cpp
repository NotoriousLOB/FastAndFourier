/**
 * @file test_cwt.cpp
 * @brief CWT forward execute, inverse, reconstruction, layout guards
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

extern "C" {
#include "fastandfourier.h"
#include "chirp.h"
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void *alloc64(size_t bytes) {
    return aligned_alloc(64, ((bytes + 63) / 64) * 64);
}

/* ---- Forward execute basic ---- */

TEST(CWTExec, ForwardImpulse) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;


    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(t);
    ASSERT_GT(n_rows, 0u);

    double *in_data = (double *)alloc64(n * sizeof(double));
    double *out_data = (double *)alloc64(n_rows * n * sizeof(double));

    memset(in_data, 0, n * sizeof(double));
    in_data[0] = 1.0;

    faf_buffer in_buf = faf_buffer_real(in_data, n);
    faf_buffer out_buf;
    memset(&out_buf, 0, sizeof(out_buf));
    out_buf.re = out_data;
    out_buf.n = n;
    out_buf.layout = FAF_LAYOUT_REAL;

    int ret = faf_execute(t, &out_buf, &in_buf);
    EXPECT_EQ(ret, 0);

    double total_energy = 0.0;
    for (size_t i = 0; i < n_rows * n; i++)
        total_energy += out_data[i] * out_data[i];
    EXPECT_GT(total_energy, 0.0);

    free(in_data);
    free(out_data);
    faf_destroy_transform(t);
}

TEST(CWTExec, ForwardFP32) {
    faf_cwt_config cfg = faf_cwt_config_init(1024);
    cfg.precision = FAF_PREC_FP32;


    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    size_t n = 1024;
    size_t n_rows = faf_cwt_n_rows(t);

    float *in_data = (float *)alloc64(n * sizeof(float));
    float *out_data = (float *)alloc64(n_rows * n * sizeof(float));
    memset(in_data, 0, n * sizeof(float));
    in_data[0] = 1.0f;

    faf_buffer in_buf = faf_buffer_real(in_data, n);
    faf_buffer out_buf;
    memset(&out_buf, 0, sizeof(out_buf));
    out_buf.re = out_data;
    out_buf.n = n;
    out_buf.layout = FAF_LAYOUT_REAL;

    EXPECT_EQ(faf_execute(t, &out_buf, &in_buf), 0);

    free(in_data);
    free(out_data);
    faf_destroy_transform(t);
}

/* ---- Layout guards ---- */

TEST(CWTExec, RejectAlias) {
    faf_cwt_config cfg = faf_cwt_config_init(1024);
    cfg.precision = FAF_PREC_FP32;


    faf_transform *t = faf_create_cwt(&cfg);
    ASSERT_NE(t, nullptr);

    float *data = (float *)alloc64(1024 * sizeof(float));
    faf_buffer buf = faf_buffer_real(data, 1024);
    EXPECT_NE(faf_execute(t, &buf, &buf), 0);

    free(data);
    faf_destroy_transform(t);
}

/* ---- Dual inverse reconstruction ---- */

static double rel_l2_error(const double *a, const double *b, size_t n) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        err += d * d;
        norm += b[i] * b[i];
    }
    return (norm > 0.0) ? std::sqrt(err / norm) : std::sqrt(err);
}

TEST(CWTRecon, DefaultL1DualImpulse) {
    /* Advertised path: default L1 bank + faf_create_inverse (dual). */
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr) << faf_get_error();
    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr) << faf_get_error();

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    double *x = (double *)alloc64(n * sizeof(double));
    double *w = (double *)alloc64(n_rows * n * sizeof(double));
    double *y = (double *)alloc64(n * sizeof(double));

    memset(x, 0, n * sizeof(double));
    x[0] = 1.0;

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = rel_l2_error(y, x, n);
    EXPECT_LE(err, 1e-5) << "Default L1 + dual inverse impulse error: " << err;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

TEST(CWTRecon, L1OneIntegralSine) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;
    cfg.norm = FAF_CWT_NORM_L1;

    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr) << faf_get_error();
    faf_transform *icwt = faf_create_icwt(&cfg, FAF_CWT_INV_L1);
    ASSERT_NE(icwt, nullptr) << faf_get_error();

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    std::vector<double> freqs(faf_cwt_n_scales(cwt));
    faf_cwt_freqs(cwt, freqs.data(), freqs.size());
    double f_test = freqs[freqs.size() / 2];

    double *x = (double *)alloc64(n * sizeof(double));
    double *w = (double *)alloc64(n_rows * n * sizeof(double));
    double *y = (double *)alloc64(n * sizeof(double));

    for (size_t i = 0; i < n; i++)
        x[i] = std::sin(2.0 * M_PI * f_test * (double)i);

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = rel_l2_error(y, x, n);
    EXPECT_LE(err, 0.25) << "L1 one-integral sine error: " << err;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

TEST(CWTRecon, DualImpulseAt0) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;
    cfg.norm = FAF_CWT_NORM_L2;


    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr);
    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr);

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    double *x = (double *)alloc64(n * sizeof(double));
    double *w = (double *)alloc64(n_rows * n * sizeof(double));
    double *y = (double *)alloc64(n * sizeof(double));

    memset(x, 0, n * sizeof(double));
    x[0] = 1.0;

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = rel_l2_error(y, x, n);
    EXPECT_LE(err, 1e-5) << "Dual inverse impulse@0 error: " << err;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

TEST(CWTRecon, DualImpulseAtN4) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;
    cfg.norm = FAF_CWT_NORM_L2;


    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr);
    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr);

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    double *x = (double *)alloc64(n * sizeof(double));
    double *w = (double *)alloc64(n_rows * n * sizeof(double));
    double *y = (double *)alloc64(n * sizeof(double));

    memset(x, 0, n * sizeof(double));
    x[n / 4] = 1.0;

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = rel_l2_error(y, x, n);
    EXPECT_LE(err, 1e-5) << "Dual inverse impulse@n/4 error: " << err;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

TEST(CWTRecon, DualSine) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP64;
    cfg.norm = FAF_CWT_NORM_L2;


    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr);
    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr);

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    std::vector<double> freqs(faf_cwt_n_scales(cwt));
    faf_cwt_freqs(cwt, freqs.data(), freqs.size());
    double f_test = freqs[freqs.size() / 2];

    double *x = (double *)alloc64(n * sizeof(double));
    double *w = (double *)alloc64(n_rows * n * sizeof(double));
    double *y = (double *)alloc64(n * sizeof(double));

    for (size_t i = 0; i < n; i++)
        x[i] = std::sin(2.0 * M_PI * f_test * (double)i);

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = rel_l2_error(y, x, n);
    EXPECT_LE(err, 1e-5) << "Dual inverse sine error: " << err;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

/* ---- FP32 vs FP64 parity ---- */

TEST(CWTRecon, FP32DualSine) {
    faf_cwt_config cfg = faf_cwt_config_init(4096);
    cfg.precision = FAF_PREC_FP32;
    cfg.norm = FAF_CWT_NORM_L2;


    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr);
    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr);

    size_t n = 4096;
    size_t n_rows = faf_cwt_n_rows(cwt);

    std::vector<double> freqs(faf_cwt_n_scales(cwt));
    faf_cwt_freqs(cwt, freqs.data(), freqs.size());
    double f_test = freqs[freqs.size() / 2];

    float *x = (float *)alloc64(n * sizeof(float));
    float *w = (float *)alloc64(n_rows * n * sizeof(float));
    float *y = (float *)alloc64(n * sizeof(float));

    for (size_t i = 0; i < n; i++)
        x[i] = (float)std::sin(2.0 * M_PI * f_test * (double)i);

    faf_buffer x_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;

    ASSERT_EQ(faf_execute(cwt, &w_buf, &x_buf), 0);

    faf_buffer y_buf = faf_buffer_real(y, n);
    ASSERT_EQ(faf_execute(icwt, &y_buf, &w_buf), 0);

    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)y[i] - (double)x[i];
        err += d * d;
        norm += (double)x[i] * (double)x[i];
    }
    double rel = (norm > 0.0) ? std::sqrt(err / norm) : std::sqrt(err);
    EXPECT_LE(rel, 1e-4) << "FP32 dual inverse sine error: " << rel;

    free(x);
    free(w);
    free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

/* ---- Transform name ---- */

TEST(CWTExec, TransformNames) {
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_CWT), "cwt");
    EXPECT_STREQ(faf_transform_name(FAF_TRANSFORM_ICWT), "icwt");
}

/* ---- create_inverse ---- */

TEST(CWTExec, CreateInverse) {
    faf_cwt_config cfg = faf_cwt_config_init(1024);
    cfg.precision = FAF_PREC_FP64;
    cfg.norm = FAF_CWT_NORM_L2;

    faf_transform *cwt = faf_create_cwt(&cfg);
    ASSERT_NE(cwt, nullptr);
    EXPECT_EQ(cwt->type, FAF_TRANSFORM_CWT);

    faf_transform *icwt = faf_create_inverse(cwt);
    ASSERT_NE(icwt, nullptr);
    EXPECT_EQ(icwt->type, FAF_TRANSFORM_ICWT);

    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}

/* ---- Chirp DSL integration ---- */

TEST(CWTChirp, StandaloneMorse) {
    faf_transform *t = chirp_compile("(cwt :n 4096 :wavelet morse)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_CWT);

    size_t n = 4096;
    float *in = (float *)alloc64(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        in[i] = sinf(2.0f * (float)M_PI * 10.0f * (float)i / (float)n);

    size_t n_rows = faf_cwt_n_rows(t);
    float *out = (float *)alloc64(n_rows * n * sizeof(float));

    faf_buffer in_buf = faf_buffer_real(in, n);
    faf_buffer out_buf;
    memset(&out_buf, 0, sizeof(out_buf));
    out_buf.re = out;
    out_buf.n = n;
    out_buf.layout = FAF_LAYOUT_REAL;

    EXPECT_EQ(faf_execute(t, &out_buf, &in_buf), 0);

    double energy = 0.0;
    for (size_t i = 0; i < n_rows * n; i++)
        energy += (double)out[i] * (double)out[i];
    EXPECT_GT(energy, 0.0);

    free(in);
    free(out);
    faf_destroy_transform(t);
}

TEST(CWTChirp, WithKeywords) {
    faf_transform *t = chirp_compile(
        "(cwt :n 1024 :wavelet morlet :mu 6.0 :voices 12 :precision f64)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_CWT);
    faf_destroy_transform(t);
}

TEST(CWTChirp, StandaloneICWT) {
    faf_transform *t = chirp_compile("(icwt :n 1024 :norm l2)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_ICWT);
    faf_destroy_transform(t);
}

TEST(CWTChirp, PipelineRejected) {
    faf_transform *t = chirp_compile("(pipeline (cwt :n 1024) sin)");
    EXPECT_EQ(t, nullptr);
}

TEST(CWTChirp, MissingNRejected) {
    faf_transform *t = chirp_compile("(cwt :wavelet morse)");
    EXPECT_EQ(t, nullptr);
}

TEST(CWTChirp, UnknownWaveletRejected) {
    faf_transform *t = chirp_compile("(cwt :n 1024 :wavelet bogus)");
    EXPECT_EQ(t, nullptr);
}
