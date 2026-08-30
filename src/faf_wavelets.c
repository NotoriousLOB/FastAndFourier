/**
 * @file faf_wavelets.c
 * @brief Lifting-scheme and filter-bank discrete wavelet transforms
 */

#include "faf.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
static int name_eq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

/* Orthonormal Haar scale: 1/sqrt(2) */
#define FAF_INV_SQRT2_F  0.7071067811865476f
#define FAF_INV_SQRT2    0.70710678118654752440

/* CDF 9/7 Sweldens lifting constants */
#define FAF_CDF97_A  (-1.586134342059924)
#define FAF_CDF97_B  (-0.052980118572961)
#define FAF_CDF97_G  ( 0.882911075530934)
#define FAF_CDF97_D  ( 0.443506852043971)
#define FAF_CDF97_K  ( 1.149604398860241)

/* Daubechies D4 / db2 analysis low-pass */
static const float D4_H_F32[4] = {
    0.4829629131445341f,
    0.8365163037378079f,
    0.2241438680420134f,
    -0.1294095225512604f
};
static const double D4_H_F64[4] = {
    0.4829629131445341,
    0.8365163037378079,
    0.2241438680420134,
    -0.1294095225512604
};

/* Symlet-4 analysis low-pass (8-tap) */
static const float SYM4_H_F32[8] = {
    -0.07576571478927333f,
    -0.02963552764599851f,
     0.49761866763201545f,
     0.8037387518059161f,
     0.29785779560527736f,
    -0.09921954357684722f,
    -0.012603967262037833f,
     0.0322231006040427f
};
static const double SYM4_H_F64[8] = {
    -0.07576571478927333,
    -0.02963552764599851,
     0.49761866763201545,
     0.8037387518059161,
     0.29785779560527736,
    -0.09921954357684722,
    -0.012603967262037833,
     0.0322231006040427
};

static size_t wrap_idx(ptrdiff_t i, size_t n) {
    ptrdiff_t m = (ptrdiff_t)n;
    ptrdiff_t r = i % m;
    if (r < 0) r += m;
    return (size_t)r;
}

const char* faf_wavelet_name(faf_wavelet_family family) {
    switch (family) {
        case FAF_WAVELET_HAAR:  return "haar";
        case FAF_WAVELET_D4:    return "d4";
        case FAF_WAVELET_CDF53: return "cdf53";
        case FAF_WAVELET_CDF97: return "cdf97";
        case FAF_WAVELET_SYM4:  return "sym4";
        default: return "unknown";
    }
}

int faf_wavelet_taps(faf_wavelet_family family) {
    switch (family) {
        case FAF_WAVELET_HAAR:  return 2;
        case FAF_WAVELET_D4:    return 4;
        case FAF_WAVELET_CDF53: return 5;
        case FAF_WAVELET_CDF97: return 9;
        case FAF_WAVELET_SYM4:  return 8;
        default: return 0;
    }
}

int faf_wavelet_from_name(const char *name, faf_wavelet_family *out) {
    if (!name || !out) return -1;
    if (name_eq(name, "haar") || name_eq(name, "db1")) {
        *out = FAF_WAVELET_HAAR;
        return 0;
    }
    if (name_eq(name, "d4") || name_eq(name, "db2") ||
        name_eq(name, "daubechies4")) {
        *out = FAF_WAVELET_D4;
        return 0;
    }
    if (name_eq(name, "db4")) {
        /* dbN has 2N taps: db4 is 8-tap, not the 4-tap D4/db2 family. */
        return -1;
    }
    if (name_eq(name, "cdf53") || name_eq(name, "legall")) {
        *out = FAF_WAVELET_CDF53;
        return 0;
    }
    if (name_eq(name, "cdf97")) {
        *out = FAF_WAVELET_CDF97;
        return 0;
    }
    if (name_eq(name, "sym4") || name_eq(name, "symlet4")) {
        *out = FAF_WAVELET_SYM4;
        return 0;
    }
    return -1;
}

const char* faf_convention_name(faf_wavelet_convention conv) {
    switch (conv) {
        case FAF_CONV_UNSPEC:        return "unspec";
        case FAF_CONV_HAAR_ORTHO:    return "haar-ortho";
        case FAF_CONV_HAAR_LAZY:     return "haar-lazy";
        case FAF_CONV_HAAR_MEAN:     return "haar-mean";
        case FAF_CONV_D4_ORTHO:      return "d4-ortho";
        case FAF_CONV_SYM4_ORTHO:    return "sym4-ortho";
        case FAF_CONV_CDF53_INT:     return "cdf53-int";
        case FAF_CONV_CDF53_ENERGY:  return "cdf53-energy";
        case FAF_CONV_CDF97_JPEG:    return "cdf97-jpeg";
        case FAF_CONV_CDF97_ORTHO:   return "cdf97-ortho";
        case FAF_CONV_CUSTOM_PR:     return "custom-pr";
        case FAF_CONV_ANALYSIS_ONLY: return "analysis-only";
        default: return "unknown";
    }
}

int faf_convention_from_name(const char *name, faf_wavelet_convention *out) {
    if (!name || !out) return -1;
    if (name_eq(name, "haar-ortho") || name_eq(name, "haar_ortho"))
        { *out = FAF_CONV_HAAR_ORTHO; return 0; }
    if (name_eq(name, "haar-lazy") || name_eq(name, "haar_lazy"))
        { *out = FAF_CONV_HAAR_LAZY; return 0; }
    if (name_eq(name, "haar-mean") || name_eq(name, "haar_mean"))
        { *out = FAF_CONV_HAAR_MEAN; return 0; }
    if (name_eq(name, "d4-ortho") || name_eq(name, "d4_ortho"))
        { *out = FAF_CONV_D4_ORTHO; return 0; }
    if (name_eq(name, "sym4-ortho") || name_eq(name, "sym4_ortho"))
        { *out = FAF_CONV_SYM4_ORTHO; return 0; }
    if (name_eq(name, "cdf53-int") || name_eq(name, "cdf53_int"))
        { *out = FAF_CONV_CDF53_INT; return 0; }
    if (name_eq(name, "cdf53-energy") || name_eq(name, "cdf53_energy"))
        { *out = FAF_CONV_CDF53_ENERGY; return 0; }
    if (name_eq(name, "cdf97-jpeg") || name_eq(name, "cdf97_jpeg"))
        { *out = FAF_CONV_CDF97_JPEG; return 0; }
    if (name_eq(name, "cdf97-ortho") || name_eq(name, "cdf97_ortho"))
        { *out = FAF_CONV_CDF97_ORTHO; return 0; }
    if (name_eq(name, "custom-pr") || name_eq(name, "custom_pr"))
        { *out = FAF_CONV_CUSTOM_PR; return 0; }
    if (name_eq(name, "analysis-only") || name_eq(name, "analysis_only"))
        { *out = FAF_CONV_ANALYSIS_ONLY; return 0; }
    return -1;
}

faf_wavelet_convention faf_convention_default(faf_wavelet_family family) {
    switch (family) {
        case FAF_WAVELET_HAAR:  return FAF_CONV_HAAR_ORTHO;
        case FAF_WAVELET_D4:    return FAF_CONV_D4_ORTHO;
        case FAF_WAVELET_CDF53: return FAF_CONV_CDF53_INT;
        case FAF_WAVELET_CDF97: return FAF_CONV_CDF97_JPEG;
        case FAF_WAVELET_SYM4:  return FAF_CONV_SYM4_ORTHO;
        default:                return FAF_CONV_UNSPEC;
    }
}

int faf_validate_convention(faf_wavelet_family family, faf_wavelet_convention conv) {
    if (conv == FAF_CONV_UNSPEC || conv == FAF_CONV_CUSTOM_PR ||
        conv == FAF_CONV_ANALYSIS_ONLY)
        return 0;
    switch (conv) {
        case FAF_CONV_HAAR_ORTHO:
        case FAF_CONV_HAAR_LAZY:
        case FAF_CONV_HAAR_MEAN:
            return (family == FAF_WAVELET_HAAR) ? 0 : -1;
        case FAF_CONV_D4_ORTHO:
            return (family == FAF_WAVELET_D4) ? 0 : -1;
        case FAF_CONV_SYM4_ORTHO:
            return (family == FAF_WAVELET_SYM4) ? 0 : -1;
        case FAF_CONV_CDF53_INT:
        case FAF_CONV_CDF53_ENERGY:
            return (family == FAF_WAVELET_CDF53) ? 0 : -1;
        case FAF_CONV_CDF97_JPEG:
        case FAF_CONV_CDF97_ORTHO:
            return (family == FAF_WAVELET_CDF97) ? 0 : -1;
        default:
            return -1;
    }
}

/* -------------------------------------------------------------------------- */
/* Haar                                                                       */
/* -------------------------------------------------------------------------- */

static void haar_fwd_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        float a = x[2 * i];
        float b = x[2 * i + 1];
        tmp[i] = (a + b) * FAF_INV_SQRT2_F;
        tmp[half + i] = (a - b) * FAF_INV_SQRT2_F;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_inv_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        float a = x[i];
        float d = x[half + i];
        tmp[2 * i]     = (a + d) * FAF_INV_SQRT2_F;
        tmp[2 * i + 1] = (a - d) * FAF_INV_SQRT2_F;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_fwd_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        double a = x[2 * i];
        double b = x[2 * i + 1];
        tmp[i] = (a + b) * FAF_INV_SQRT2;
        tmp[half + i] = (a - b) * FAF_INV_SQRT2;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

static void haar_inv_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        double a = x[i];
        double d = x[half + i];
        tmp[2 * i]     = (a + d) * FAF_INV_SQRT2;
        tmp[2 * i + 1] = (a - d) * FAF_INV_SQRT2;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

/* -------------------------------------------------------------------------- */
/* Haar lazy: no scale factor, divide by 2 on recon                           */
/* -------------------------------------------------------------------------- */

static void haar_lazy_fwd_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[i]        = x[2 * i] + x[2 * i + 1];
        tmp[half + i] = x[2 * i] - x[2 * i + 1];
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_lazy_inv_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[2 * i]     = (x[i] + x[half + i]) * 0.5f;
        tmp[2 * i + 1] = (x[i] - x[half + i]) * 0.5f;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_lazy_fwd_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[i]        = x[2 * i] + x[2 * i + 1];
        tmp[half + i] = x[2 * i] - x[2 * i + 1];
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

static void haar_lazy_inv_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[2 * i]     = (x[i] + x[half + i]) * 0.5;
        tmp[2 * i + 1] = (x[i] - x[half + i]) * 0.5;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

/* -------------------------------------------------------------------------- */
/* Haar mean: LP = (a+b)/2, HP = a-b                                         */
/* -------------------------------------------------------------------------- */

static void haar_mean_fwd_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[i]        = (x[2 * i] + x[2 * i + 1]) * 0.5f;
        tmp[half + i] = x[2 * i] - x[2 * i + 1];
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_mean_inv_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[2 * i]     = x[i] + x[half + i] * 0.5f;
        tmp[2 * i + 1] = x[i] - x[half + i] * 0.5f;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void haar_mean_fwd_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[i]        = (x[2 * i] + x[2 * i + 1]) * 0.5;
        tmp[half + i] = x[2 * i] - x[2 * i + 1];
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

static void haar_mean_inv_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t i = 0; i < half; i++) {
        tmp[2 * i]     = x[i] + x[half + i] * 0.5;
        tmp[2 * i + 1] = x[i] - x[half + i] * 0.5;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

/* -------------------------------------------------------------------------- */
/* Orthonormal filter bank (D4, Sym4)                                         */
/* -------------------------------------------------------------------------- */

static void filt_fwd_f32(float *x, size_t n, const float *h, int taps) {
    size_t half = n / 2;
    float *tmp = (float*)malloc(n * sizeof(float));
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        float a = 0.0f, d = 0.0f;
        for (int i = 0; i < taps; i++) {
            float s = x[(2 * k + (size_t)i) % n];
            float g = ((i & 1) ? -1.0f : 1.0f) * h[taps - 1 - i];
            a += h[i] * s;
            d += g * s;
        }
        tmp[k] = a;
        tmp[half + k] = d;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void filt_inv_f32(float *x, size_t n, const float *h, int taps) {
    size_t half = n / 2;
    float *tmp = (float*)calloc(n, sizeof(float));
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        float a = x[k];
        float d = x[half + k];
        for (int i = 0; i < taps; i++) {
            float g = ((i & 1) ? -1.0f : 1.0f) * h[taps - 1 - i];
            tmp[(2 * k + (size_t)i) % n] += h[i] * a + g * d;
        }
    }
    memcpy(x, tmp, n * sizeof(float));
    free(tmp);
}

static void filt_fwd_f64(double *x, size_t n, const double *h, int taps) {
    size_t half = n / 2;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        double a = 0.0, d = 0.0;
        for (int i = 0; i < taps; i++) {
            double s = x[(2 * k + (size_t)i) % n];
            double g = ((i & 1) ? -1.0 : 1.0) * h[taps - 1 - i];
            a += h[i] * s;
            d += g * s;
        }
        tmp[k] = a;
        tmp[half + k] = d;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

static void filt_inv_f64(double *x, size_t n, const double *h, int taps) {
    size_t half = n / 2;
    double *tmp = (double*)calloc(n, sizeof(double));
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        double a = x[k];
        double d = x[half + k];
        for (int i = 0; i < taps; i++) {
            double g = ((i & 1) ? -1.0 : 1.0) * h[taps - 1 - i];
            tmp[(2 * k + (size_t)i) % n] += h[i] * a + g * d;
        }
    }
    memcpy(x, tmp, n * sizeof(double));
    free(tmp);
}

/* -------------------------------------------------------------------------- */
/* CDF 5/3 lifting                                                            */
/* -------------------------------------------------------------------------- */

static void cdf53_fwd_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *s = (float*)malloc(n * sizeof(float));
    if (!s) return;
    float *d = s + half;
    for (size_t i = 0; i < half; i++) {
        s[i] = x[2 * i];
        d[i] = x[2 * i + 1];
    }
    for (size_t i = 0; i < half; i++) {
        d[i] -= 0.5f * (s[i] + s[(i + 1) % half]);
    }
    for (size_t i = 0; i < half; i++) {
        s[i] += 0.25f * (d[wrap_idx((ptrdiff_t)i - 1, half)] + d[i]);
    }
    memcpy(x, s, n * sizeof(float));
    free(s);
}

static void cdf53_inv_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *s = (float*)malloc(n * sizeof(float));
    if (!s) return;
    float *d = s + half;
    memcpy(s, x, n * sizeof(float));
    for (size_t i = 0; i < half; i++) {
        s[i] -= 0.25f * (d[wrap_idx((ptrdiff_t)i - 1, half)] + d[i]);
    }
    for (size_t i = 0; i < half; i++) {
        d[i] += 0.5f * (s[i] + s[(i + 1) % half]);
    }
    for (size_t i = 0; i < half; i++) {
        x[2 * i] = s[i];
        x[2 * i + 1] = d[i];
    }
    free(s);
}

static void cdf53_fwd_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *s = (double*)malloc(n * sizeof(double));
    if (!s) return;
    double *d = s + half;
    for (size_t i = 0; i < half; i++) {
        s[i] = x[2 * i];
        d[i] = x[2 * i + 1];
    }
    for (size_t i = 0; i < half; i++) {
        d[i] -= 0.5 * (s[i] + s[(i + 1) % half]);
    }
    for (size_t i = 0; i < half; i++) {
        s[i] += 0.25 * (d[wrap_idx((ptrdiff_t)i - 1, half)] + d[i]);
    }
    memcpy(x, s, n * sizeof(double));
    free(s);
}

static void cdf53_inv_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *s = (double*)malloc(n * sizeof(double));
    if (!s) return;
    double *d = s + half;
    memcpy(s, x, n * sizeof(double));
    for (size_t i = 0; i < half; i++) {
        s[i] -= 0.25 * (d[wrap_idx((ptrdiff_t)i - 1, half)] + d[i]);
    }
    for (size_t i = 0; i < half; i++) {
        d[i] += 0.5 * (s[i] + s[(i + 1) % half]);
    }
    for (size_t i = 0; i < half; i++) {
        x[2 * i] = s[i];
        x[2 * i + 1] = d[i];
    }
    free(s);
}

/* -------------------------------------------------------------------------- */
/* CDF 9/7 lifting                                                            */
/* -------------------------------------------------------------------------- */

#define LIFT_PAIR_F32(pred_c, upd_c) \
    do { \
        for (size_t i = 0; i < half; i++) \
            d[i] += (float)(pred_c) * (s[i] + s[(i + 1) % half]); \
        for (size_t i = 0; i < half; i++) \
            s[i] += (float)(upd_c) * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]); \
    } while (0)

#define UNLIFT_PAIR_F32(pred_c, upd_c) \
    do { \
        for (size_t i = 0; i < half; i++) \
            s[i] -= (float)(upd_c) * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]); \
        for (size_t i = 0; i < half; i++) \
            d[i] -= (float)(pred_c) * (s[i] + s[(i + 1) % half]); \
    } while (0)

static void cdf97_fwd_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *s = (float*)malloc(n * sizeof(float));
    if (!s) return;
    float *d = s + half;
    for (size_t i = 0; i < half; i++) {
        s[i] = x[2 * i];
        d[i] = x[2 * i + 1];
    }
    LIFT_PAIR_F32(FAF_CDF97_A, FAF_CDF97_B);
    LIFT_PAIR_F32(FAF_CDF97_G, FAF_CDF97_D);
    for (size_t i = 0; i < half; i++) {
        s[i] *= (float)FAF_CDF97_K;
        d[i] /= (float)FAF_CDF97_K;
    }
    memcpy(x, s, n * sizeof(float));
    free(s);
}

static void cdf97_inv_f32(float *x, size_t n) {
    size_t half = n / 2;
    float *s = (float*)malloc(n * sizeof(float));
    if (!s) return;
    float *d = s + half;
    memcpy(s, x, n * sizeof(float));
    for (size_t i = 0; i < half; i++) {
        s[i] /= (float)FAF_CDF97_K;
        d[i] *= (float)FAF_CDF97_K;
    }
    UNLIFT_PAIR_F32(FAF_CDF97_G, FAF_CDF97_D);
    UNLIFT_PAIR_F32(FAF_CDF97_A, FAF_CDF97_B);
    for (size_t i = 0; i < half; i++) {
        x[2 * i] = s[i];
        x[2 * i + 1] = d[i];
    }
    free(s);
}

static void cdf97_fwd_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *s = (double*)malloc(n * sizeof(double));
    if (!s) return;
    double *d = s + half;
    for (size_t i = 0; i < half; i++) {
        s[i] = x[2 * i];
        d[i] = x[2 * i + 1];
    }
    for (size_t i = 0; i < half; i++)
        d[i] += FAF_CDF97_A * (s[i] + s[(i + 1) % half]);
    for (size_t i = 0; i < half; i++)
        s[i] += FAF_CDF97_B * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]);
    for (size_t i = 0; i < half; i++)
        d[i] += FAF_CDF97_G * (s[i] + s[(i + 1) % half]);
    for (size_t i = 0; i < half; i++)
        s[i] += FAF_CDF97_D * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]);
    for (size_t i = 0; i < half; i++) {
        s[i] *= FAF_CDF97_K;
        d[i] /= FAF_CDF97_K;
    }
    memcpy(x, s, n * sizeof(double));
    free(s);
}

static void cdf97_inv_f64(double *x, size_t n) {
    size_t half = n / 2;
    double *s = (double*)malloc(n * sizeof(double));
    if (!s) return;
    double *d = s + half;
    memcpy(s, x, n * sizeof(double));
    for (size_t i = 0; i < half; i++) {
        s[i] /= FAF_CDF97_K;
        d[i] *= FAF_CDF97_K;
    }
    for (size_t i = 0; i < half; i++)
        s[i] -= FAF_CDF97_D * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]);
    for (size_t i = 0; i < half; i++)
        d[i] -= FAF_CDF97_G * (s[i] + s[(i + 1) % half]);
    for (size_t i = 0; i < half; i++)
        s[i] -= FAF_CDF97_B * (d[i] + d[wrap_idx((ptrdiff_t)i - 1, half)]);
    for (size_t i = 0; i < half; i++)
        d[i] -= FAF_CDF97_A * (s[i] + s[(i + 1) % half]);
    for (size_t i = 0; i < half; i++) {
        x[2 * i] = s[i];
        x[2 * i + 1] = d[i];
    }
    free(s);
}

/* -------------------------------------------------------------------------- */
/* Public level kernels                                                       */
/* -------------------------------------------------------------------------- */

void faf_dwt_level_conv_f32(float *x, size_t n,
                            faf_wavelet_family family,
                            faf_wavelet_convention conv, int inverse) {
    if (!x || n < 2 || (n & 1u)) return;

    if (family == FAF_WAVELET_HAAR) {
        faf_wavelet_convention c = (conv == FAF_CONV_UNSPEC) ? FAF_CONV_HAAR_ORTHO : conv;
        switch (c) {
            case FAF_CONV_HAAR_LAZY:
                if (inverse) haar_lazy_inv_f32(x, n); else haar_lazy_fwd_f32(x, n);
                return;
            case FAF_CONV_HAAR_MEAN:
                if (inverse) haar_mean_inv_f32(x, n); else haar_mean_fwd_f32(x, n);
                return;
            default:
                if (inverse) haar_inv_f32(x, n); else haar_fwd_f32(x, n);
                return;
        }
    }

    switch (family) {
        case FAF_WAVELET_D4:
            if (inverse) filt_inv_f32(x, n, D4_H_F32, 4); else filt_fwd_f32(x, n, D4_H_F32, 4);
            break;
        case FAF_WAVELET_CDF53:
            if (inverse) cdf53_inv_f32(x, n); else cdf53_fwd_f32(x, n);
            break;
        case FAF_WAVELET_CDF97:
            if (inverse) cdf97_inv_f32(x, n); else cdf97_fwd_f32(x, n);
            break;
        case FAF_WAVELET_SYM4:
            if (inverse) filt_inv_f32(x, n, SYM4_H_F32, 8); else filt_fwd_f32(x, n, SYM4_H_F32, 8);
            break;
        default:
            break;
    }
}

void faf_dwt_level_conv_f64(double *x, size_t n,
                            faf_wavelet_family family,
                            faf_wavelet_convention conv, int inverse) {
    if (!x || n < 2 || (n & 1u)) return;

    if (family == FAF_WAVELET_HAAR) {
        faf_wavelet_convention c = (conv == FAF_CONV_UNSPEC) ? FAF_CONV_HAAR_ORTHO : conv;
        switch (c) {
            case FAF_CONV_HAAR_LAZY:
                if (inverse) haar_lazy_inv_f64(x, n); else haar_lazy_fwd_f64(x, n);
                return;
            case FAF_CONV_HAAR_MEAN:
                if (inverse) haar_mean_inv_f64(x, n); else haar_mean_fwd_f64(x, n);
                return;
            default:
                if (inverse) haar_inv_f64(x, n); else haar_fwd_f64(x, n);
                return;
        }
    }

    switch (family) {
        case FAF_WAVELET_D4:
            if (inverse) filt_inv_f64(x, n, D4_H_F64, 4); else filt_fwd_f64(x, n, D4_H_F64, 4);
            break;
        case FAF_WAVELET_CDF53:
            if (inverse) cdf53_inv_f64(x, n); else cdf53_fwd_f64(x, n);
            break;
        case FAF_WAVELET_CDF97:
            if (inverse) cdf97_inv_f64(x, n); else cdf97_fwd_f64(x, n);
            break;
        case FAF_WAVELET_SYM4:
            if (inverse) filt_inv_f64(x, n, SYM4_H_F64, 8); else filt_fwd_f64(x, n, SYM4_H_F64, 8);
            break;
        default:
            break;
    }
}

void faf_dwt_level_f32(float *x, size_t n, faf_wavelet_family family, int inverse) {
    faf_dwt_level_conv_f32(x, n, family, FAF_CONV_UNSPEC, inverse);
}

void faf_dwt_level_f64(double *x, size_t n, faf_wavelet_family family, int inverse) {
    faf_dwt_level_conv_f64(x, n, family, FAF_CONV_UNSPEC, inverse);
}

void faf_threshold_band_f32(float *x, size_t start, size_t end, int mode, float lambda) {
    if (!x || start >= end) return;
    float lam = lambda < 0.0f ? -lambda : lambda;
    for (size_t i = start; i < end; i++) {
        float v = x[i];
        float av = v < 0.0f ? -v : v;
        if (av <= lam) {
            x[i] = 0.0f;
        } else if (mode == (int)FAF_THRESH_SOFT) {
            x[i] = (v > 0.0f) ? (v - lam) : (v + lam);
        }
    }
}

void faf_threshold_band_f64(double *x, size_t start, size_t end, int mode, double lambda) {
    if (!x || start >= end) return;
    double lam = lambda < 0.0 ? -lambda : lambda;
    for (size_t i = start; i < end; i++) {
        double v = x[i];
        double av = v < 0.0 ? -v : v;
        if (av <= lam) {
            x[i] = 0.0;
        } else if (mode == (int)FAF_THRESH_SOFT) {
            x[i] = (v > 0.0) ? (v - lam) : (v + lam);
        }
    }
}

size_t faf_fft_stage_count(size_t n) {
    size_t bits = 0;
    size_t temp = n;
    while (temp > 1) { temp >>= 1; bits++; }
    return bits;
}

void faf_emit_fft_stages(faf_transform *t, size_t n, size_t *idx) {
    size_t bits = faf_fft_stage_count(n);
    for (size_t stage = 0; stage < bits; stage++) {
        size_t group_size = 2u << stage;
        size_t twiddle_step = n / group_size;
        if (*idx >= 65536) return;
        t->code[(*idx)++] = (faf_inst){
            .packed = FAF_PACK_INST(FAF_FFT_STAGE, 0, 0),
            .a0 = (uint32_t)group_size,
            .a1 = 1,
            .a2 = (uint32_t)twiddle_step
        };
    }
}
