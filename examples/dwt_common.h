/**
 * @file dwt_common.h
 * @brief Shared helpers for Chirp DWT examples
 */
#ifndef FAF_DWT_COMMON_H
#define FAF_DWT_COMMON_H

#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float *dwt_alloc(size_t n) {
    size_t bytes = ((2 * n * sizeof(float)) + 63u) & ~(size_t)63u;
    float *p = (float*)aligned_alloc(64, bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

static inline void dwt_fill_tone(float *buf, size_t n, float freq, float amp) {
    for (size_t i = 0; i < n; i++) {
        buf[2 * i] = amp * sinf(2.0f * (float)M_PI * freq * (float)i / (float)n);
        buf[2 * i + 1] = 0.0f;
    }
}

static inline float dwt_max_abs(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(a[2 * i] - b[2 * i]);
        if (e > m) m = e;
    }
    return m;
}

static inline float dwt_snr_db(const float *sig, const float *ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = ref[2 * i];
        double e = (double)sig[2 * i] - r;
        num += r * r;
        den += e * e;
    }
    if (den < 1e-20) return 200.0f;
    return (float)(10.0 * log10(num / den));
}

static inline float dwt_energy(const float *buf, size_t start, size_t end) {
    float e = 0.0f;
    for (size_t i = start; i < end; i++) e += buf[2 * i] * buf[2 * i];
    return e;
}

static inline int dwt_run_pr(faf_wavelet_family fam, size_t n, size_t levels,
                             const float *in, float *mid, float *out, float *err) {
    faf_transform *fwd = faf_create_dwt(fam, n, levels, false, FAF_PREC_FP32, 0);
    faf_transform *inv = faf_create_dwt(fam, n, levels, true, FAF_PREC_FP32, 0);
    if (!fwd || !inv) {
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
        return -1;
    }
    if (faf_execute_f32(fwd, mid, in) != 0 || faf_execute_f32(inv, out, mid) != 0) {
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
        return -1;
    }
    *err = dwt_max_abs(out, in, n);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    return 0;
}

#endif
