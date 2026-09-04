/**
 * @file faf_fft_stage.c
 * @brief Mixed-radix FFT stage kernels (radix 2/3/4/5/7)
 *
 * Legacy radix-2 stages keep a1 == 1 (group size in a0). Mixed-radix
 * stages set a1 to 3, 4, 5, or 7. Radix-7 is for Rader inner FFTs
 * (7-smooth n-1), not a public 7-smooth size policy.
 */

#include "faf.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RSQ3_2  0.86602540378443864676  /* sqrt(3)/2 */
#define C2PI5   0.30901699437494742410  /* cos(2π/5) */
#define C4PI5  -0.80901699437494742410  /* cos(4π/5) */
#define S2PI5   0.95105651629515357212  /* sin(2π/5) */
#define S4PI5   0.58778525229247312917  /* sin(4π/5) */

int faf_is_5_smooth(size_t n) {
    if (n == 0) return 0;
    while ((n % 2u) == 0) n /= 2u;
    while ((n % 3u) == 0) n /= 3u;
    while ((n % 5u) == 0) n /= 5u;
    return n == 1;
}

int faf_is_7_smooth(size_t n) {
    if (n == 0) return 0;
    while ((n % 2u) == 0) n /= 2u;
    while ((n % 3u) == 0) n /= 3u;
    while ((n % 5u) == 0) n /= 5u;
    while ((n % 7u) == 0) n /= 7u;
    return n == 1;
}

int faf_is_prime(size_t n) {
    if (n < 2) return 0;
    if ((n % 2u) == 0) return n == 2;
    for (size_t d = 3; d * d <= n; d += 2u)
        if ((n % d) == 0) return 0;
    return 1;
}

int faf_rader_eligible(size_t n) {
    return n >= 11 && faf_is_prime(n) && faf_is_7_smooth(n - 1);
}

static size_t modpow_sz(size_t b, size_t e, size_t m) {
    uint64_t r = 1, base = (uint64_t)b % (uint64_t)m;
    while (e) {
        if (e & 1u)
            r = (r * base) % (uint64_t)m;
        base = (base * base) % (uint64_t)m;
        e >>= 1;
    }
    return (size_t)r;
}

size_t faf_primitive_root(size_t p) {
    if (!faf_is_prime(p) || p < 3) return 0;
    size_t phi = p - 1;
    size_t fac[16];
    int nf = 0;
    size_t m = phi;
    if ((m % 2u) == 0) {
        fac[nf++] = 2;
        while ((m % 2u) == 0) m /= 2u;
    }
    for (size_t d = 3; d * d <= m; d += 2u) {
        if ((m % d) == 0) {
            fac[nf++] = d;
            while ((m % d) == 0) m /= d;
        }
    }
    if (m > 1) fac[nf++] = m;
    for (size_t g = 2; g < p; g++) {
        int ok = 1;
        for (int i = 0; i < nf; i++) {
            if (modpow_sz(g, phi / fac[i], p) == 1) {
                ok = 0;
                break;
            }
        }
        if (ok) return g;
    }
    return 0;
}

int faf_factor_7smooth(size_t n, int *factors, int *n_factors) {
    int nf = 0;
    if (!factors || !n_factors || n == 0) return -1;
    while ((n % 4u) == 0 && nf < 16) { factors[nf++] = 4; n /= 4u; }
    while ((n % 2u) == 0 && nf < 16) { factors[nf++] = 2; n /= 2u; }
    while ((n % 3u) == 0 && nf < 16) { factors[nf++] = 3; n /= 3u; }
    while ((n % 5u) == 0 && nf < 16) { factors[nf++] = 5; n /= 5u; }
    while ((n % 7u) == 0 && nf < 16) { factors[nf++] = 7; n /= 7u; }
    *n_factors = nf;
    return n == 1 ? 0 : -1;
}

int faf_factor_5smooth(size_t n, int *factors, int *n_factors) {
    int rc = faf_factor_7smooth(n, factors, n_factors);
    if (rc != 0) return rc;
    for (int i = 0; i < *n_factors; i++)
        if (factors[i] == 7) return -1;
    return 0;
}

size_t faf_digit_reverse(size_t i, const int *factors, int n_factors) {
    size_t rev = 0;
    for (int k = 0; k < n_factors; k++) {
        size_t f = (size_t)factors[k];
        if (f == 0) break;
        rev = rev * f + (i % f);
        i /= f;
    }
    return rev;
}

size_t faf_next_5_smooth(size_t min) {
    if (min <= 1) return 1;
    if (faf_is_5_smooth(min)) return min;
    size_t cap = min * 2u + 32u;
    if (cap < min) cap = (size_t)-1;
    for (size_t n = min + 1; n < cap; n++) {
        if (faf_is_5_smooth(n)) return n;
    }
    return dsir_next_power_of_2(min);
}

void faf_digitrev_split_f32(float *re, float *im, size_t n) {
    int fac[16], nf = 0;
    if (faf_factor_7smooth(n, fac, &nf) != 0) return;
    for (size_t i = 0; i < n; i++) {
        size_t j = faf_digit_reverse(i, fac, nf);
        if (j > i) {
            float tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr;    im[j] = ti;
        }
    }
}

void faf_digitrev_split_f64(double *re, double *im, size_t n) {
    int fac[16], nf = 0;
    if (faf_factor_7smooth(n, fac, &nf) != 0) return;
    for (size_t i = 0; i < n; i++) {
        size_t j = faf_digit_reverse(i, fac, nf);
        if (j > i) {
            double tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr;    im[j] = ti;
        }
    }
}

static void twiddle_mul_f32(float *br, float *bi, const float *tw, size_t idx,
                            size_t ntw) {
    if (!tw || idx >= ntw) return;
    float wr = tw[2 * idx], wi = tw[2 * idx + 1];
    float r = *br, i = *bi;
    *br = r * wr - i * wi;
    *bi = r * wi + i * wr;
}

static void twiddle_mul_f64(double *br, double *bi, const double *tw, size_t idx,
                            size_t ntw) {
    if (!tw || idx >= ntw) return;
    double wr = tw[2 * idx], wi = tw[2 * idx + 1];
    double r = *br, i = *bi;
    *br = r * wr - i * wi;
    *bi = r * wi + i * wr;
}

static void radix2_split_f32(float *re, float *im, size_t n, size_t group,
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
            if (i1 >= n || i2 >= n) continue;
            float ar = re[i1], ai = im[i1];
            float br = re[i2], bi = im[i2];
            twiddle_mul_f32(&br, &bi, tw, r * tw_step, ntw);
            re[i1] = ar + br; im[i1] = ai + bi;
            re[i2] = ar - br; im[i2] = ai - bi;
        }
    }
}

static void radix3_split_f32(float *re, float *im, size_t n, size_t group,
                             size_t tw_step, const float *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 3;
    if (m == 0) return;
    size_t ngroups = n / group;
    float us = inverse ? (float)RSQ3_2 : -(float)RSQ3_2;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            size_t i1 = i0 + m;
            size_t i2 = i0 + 2 * m;
            float a0r = re[i0], a0i = im[i0];
            float a1r = re[i1], a1i = im[i1];
            float a2r = re[i2], a2i = im[i2];
            twiddle_mul_f32(&a1r, &a1i, tw, j * tw_step, ntw);
            twiddle_mul_f32(&a2r, &a2i, tw, j * 2 * tw_step, ntw);
            float sr = a1r + a2r, si = a1i + a2i;
            float dr = a1r - a2r, di = a1i - a2i;
            re[i0] = a0r + sr;
            im[i0] = a0i + si;
            float cr = a0r - 0.5f * sr;
            float ci = a0i - 0.5f * si;
            /* ±i * (dr, di) = (∓di, ±dr) with us = ±√3/2 */
            re[i1] = cr - us * di;
            im[i1] = ci + us * dr;
            re[i2] = cr + us * di;
            im[i2] = ci - us * dr;
        }
    }
}

static void radix4_split_f32(float *re, float *im, size_t n, size_t group,
                             size_t tw_step, const float *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 4;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            size_t i1 = i0 + m, i2 = i0 + 2 * m, i3 = i0 + 3 * m;
            float a0r = re[i0], a0i = im[i0];
            float a1r = re[i1], a1i = im[i1];
            float a2r = re[i2], a2i = im[i2];
            float a3r = re[i3], a3i = im[i3];
            twiddle_mul_f32(&a1r, &a1i, tw, j * tw_step, ntw);
            twiddle_mul_f32(&a2r, &a2i, tw, j * 2 * tw_step, ntw);
            twiddle_mul_f32(&a3r, &a3i, tw, j * 3 * tw_step, ntw);
            float t0r = a0r + a2r, t0i = a0i + a2i;
            float t1r = a0r - a2r, t1i = a0i - a2i;
            float t2r = a1r + a3r, t2i = a1i + a3i;
            float t3r = a1r - a3r, t3i = a1i - a3i;
            re[i0] = t0r + t2r; im[i0] = t0i + t2i;
            re[i2] = t0r - t2r; im[i2] = t0i - t2i;
            if (!inverse) {
                /* y1 = t1 - i t3, y3 = t1 + i t3 */
                re[i1] = t1r + t3i; im[i1] = t1i - t3r;
                re[i3] = t1r - t3i; im[i3] = t1i + t3r;
            } else {
                re[i1] = t1r - t3i; im[i1] = t1i + t3r;
                re[i3] = t1r + t3i; im[i3] = t1i - t3r;
            }
        }
    }
}

static void dft5_f32(float *yr, float *yi,
                     float a0r, float a0i, float a1r, float a1i,
                     float a2r, float a2i, float a3r, float a3i,
                     float a4r, float a4i, int inverse) {
    float sign = inverse ? 1.0f : -1.0f;
    float w1r = (float)C2PI5, w1i = sign * (float)S2PI5;
    float w2r = (float)C4PI5, w2i = sign * (float)S4PI5;
    float w3r = (float)C4PI5, w3i = -sign * (float)S4PI5;
    float w4r = (float)C2PI5, w4i = -sign * (float)S2PI5;
    float xr[5] = { a0r, a1r, a2r, a3r, a4r };
    float xi[5] = { a0i, a1i, a2i, a3i, a4i };
    float wr[5] = { 1.0f, w1r, w2r, w3r, w4r };
    float wi[5] = { 0.0f, w1i, w2i, w3i, w4i };
    for (int k = 0; k < 5; k++) {
        float sr = 0.0f, si = 0.0f;
        for (int j = 0; j < 5; j++) {
            int e = (j * k) % 5;
            sr += xr[j] * wr[e] - xi[j] * wi[e];
            si += xr[j] * wi[e] + xi[j] * wr[e];
        }
        yr[k] = sr;
        yi[k] = si;
    }
}

static void radix5_split_f32(float *re, float *im, size_t n, size_t group,
                             size_t tw_step, const float *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 5;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            float ar[5], ai[5], yr[5], yi[5];
            for (int p = 0; p < 5; p++) {
                size_t ix = i0 + (size_t)p * m;
                ar[p] = re[ix];
                ai[p] = im[ix];
                if (p > 0)
                    twiddle_mul_f32(&ar[p], &ai[p], tw, j * (size_t)p * tw_step,
                                    ntw);
            }
            dft5_f32(yr, yi, ar[0], ai[0], ar[1], ai[1], ar[2], ai[2],
                     ar[3], ai[3], ar[4], ai[4], inverse);
            for (int p = 0; p < 5; p++) {
                size_t ix = i0 + (size_t)p * m;
                re[ix] = yr[p];
                im[ix] = yi[p];
            }
        }
    }
}

static void dft7_f32(float *yr, float *yi,
                     const float *ar, const float *ai, int inverse) {
    float sign = inverse ? 1.0f : -1.0f;
    const float c1 = 0.62348980185873353052f, s1 = 0.78183148246802980870f;
    const float c2 = -0.22252093395631440428f, s2 = 0.97492791218182360701f;
    const float c3 = -0.90096886790241912623f, s3 = 0.43388373911755812047f;
    float wr[7] = { 1.0f, c1, c2, c3, c3, c2, c1 };
    float wi[7] = { 0.0f, sign * s1, sign * s2, sign * s3,
                    -sign * s3, -sign * s2, -sign * s1 };
    for (int k = 0; k < 7; k++) {
        float sr = 0.0f, si = 0.0f;
        for (int j = 0; j < 7; j++) {
            int e = (j * k) % 7;
            sr += ar[j] * wr[e] - ai[j] * wi[e];
            si += ar[j] * wi[e] + ai[j] * wr[e];
        }
        yr[k] = sr;
        yi[k] = si;
    }
}

static void radix7_split_f32(float *re, float *im, size_t n, size_t group,
                             size_t tw_step, const float *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 7;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            float ar[7], ai[7], yr[7], yi[7];
            for (int p = 0; p < 7; p++) {
                size_t ix = i0 + (size_t)p * m;
                ar[p] = re[ix];
                ai[p] = im[ix];
                if (p > 0)
                    twiddle_mul_f32(&ar[p], &ai[p], tw, j * (size_t)p * tw_step,
                                    ntw);
            }
            dft7_f32(yr, yi, ar, ai, inverse);
            for (int p = 0; p < 7; p++) {
                size_t ix = i0 + (size_t)p * m;
                re[ix] = yr[p];
                im[ix] = yi[p];
            }
        }
    }
}

void faf_fft_stage_split_f32(float *re, float *im, size_t n,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             const float *tw, size_t ntw, int inverse) {
    if (!re || !im || n == 0 || a0 == 0) return;
    if (a1 == 3) {
        radix3_split_f32(re, im, n, a0, a2, tw, ntw, inverse);
        return;
    }
    if (a1 == 4) {
        radix4_split_f32(re, im, n, a0, a2, tw, ntw, inverse);
        return;
    }
    if (a1 == 5) {
        radix5_split_f32(re, im, n, a0, a2, tw, ntw, inverse);
        return;
    }
    if (a1 == 7) {
        radix7_split_f32(re, im, n, a0, a2, tw, ntw, inverse);
        return;
    }
    /* a1 == 1 (or 2): legacy / mixed radix-2 */
    size_t stride = (a1 == 0) ? 1 : ((a1 == 2) ? 1 : a1);
    if (a1 == 1 || a1 == 2)
        stride = 1;
#ifdef FAF_ARCH_AARCH64
    if (stride == 1) {
        faf_radix2_split_neon_f32(re, im, n, a0, stride, a2, tw, ntw);
        return;
    }
#endif
    radix2_split_f32(re, im, n, a0, stride, a2, tw, ntw);
}

/* ---- FP64 copies ---- */

static void radix2_split_f64(double *re, double *im, size_t n, size_t group,
                             size_t stride, size_t tw_step,
                             const double *tw, size_t ntw) {
    size_t half = group / 2;
    if (half == 0 || group == 0) return;
    size_t ngroups = n / (group * stride);
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group * stride;
        for (size_t r = 0; r < half; r++) {
            size_t i1 = base + r * stride;
            size_t i2 = i1 + half * stride;
            if (i1 >= n || i2 >= n) continue;
            double ar = re[i1], ai = im[i1];
            double br = re[i2], bi = im[i2];
            twiddle_mul_f64(&br, &bi, tw, r * tw_step, ntw);
            re[i1] = ar + br; im[i1] = ai + bi;
            re[i2] = ar - br; im[i2] = ai - bi;
        }
    }
}

static void radix3_split_f64(double *re, double *im, size_t n, size_t group,
                             size_t tw_step, const double *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 3;
    if (m == 0) return;
    size_t ngroups = n / group;
    double us = inverse ? RSQ3_2 : -RSQ3_2;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j, i1 = i0 + m, i2 = i0 + 2 * m;
            double a0r = re[i0], a0i = im[i0];
            double a1r = re[i1], a1i = im[i1];
            double a2r = re[i2], a2i = im[i2];
            twiddle_mul_f64(&a1r, &a1i, tw, j * tw_step, ntw);
            twiddle_mul_f64(&a2r, &a2i, tw, j * 2 * tw_step, ntw);
            double sr = a1r + a2r, si = a1i + a2i;
            double dr = a1r - a2r, di = a1i - a2i;
            re[i0] = a0r + sr; im[i0] = a0i + si;
            double cr = a0r - 0.5 * sr, ci = a0i - 0.5 * si;
            re[i1] = cr - us * di; im[i1] = ci + us * dr;
            re[i2] = cr + us * di; im[i2] = ci - us * dr;
        }
    }
}

static void radix4_split_f64(double *re, double *im, size_t n, size_t group,
                             size_t tw_step, const double *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 4;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            size_t i1 = i0 + m, i2 = i0 + 2 * m, i3 = i0 + 3 * m;
            double a0r = re[i0], a0i = im[i0];
            double a1r = re[i1], a1i = im[i1];
            double a2r = re[i2], a2i = im[i2];
            double a3r = re[i3], a3i = im[i3];
            twiddle_mul_f64(&a1r, &a1i, tw, j * tw_step, ntw);
            twiddle_mul_f64(&a2r, &a2i, tw, j * 2 * tw_step, ntw);
            twiddle_mul_f64(&a3r, &a3i, tw, j * 3 * tw_step, ntw);
            double t0r = a0r + a2r, t0i = a0i + a2i;
            double t1r = a0r - a2r, t1i = a0i - a2i;
            double t2r = a1r + a3r, t2i = a1i + a3i;
            double t3r = a1r - a3r, t3i = a1i - a3i;
            re[i0] = t0r + t2r; im[i0] = t0i + t2i;
            re[i2] = t0r - t2r; im[i2] = t0i - t2i;
            if (!inverse) {
                re[i1] = t1r + t3i; im[i1] = t1i - t3r;
                re[i3] = t1r - t3i; im[i3] = t1i + t3r;
            } else {
                re[i1] = t1r - t3i; im[i1] = t1i + t3r;
                re[i3] = t1r + t3i; im[i3] = t1i - t3r;
            }
        }
    }
}

static void dft5_f64(double *yr, double *yi,
                     double a0r, double a0i, double a1r, double a1i,
                     double a2r, double a2i, double a3r, double a3i,
                     double a4r, double a4i, int inverse) {
    double sign = inverse ? 1.0 : -1.0;
    double wr[5] = { 1.0, C2PI5, C4PI5, C4PI5, C2PI5 };
    double wi[5] = { 0.0, sign * S2PI5, sign * S4PI5, -sign * S4PI5,
                     -sign * S2PI5 };
    double xr[5] = { a0r, a1r, a2r, a3r, a4r };
    double xi[5] = { a0i, a1i, a2i, a3i, a4i };
    for (int k = 0; k < 5; k++) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < 5; j++) {
            int e = (j * k) % 5;
            sr += xr[j] * wr[e] - xi[j] * wi[e];
            si += xr[j] * wi[e] + xi[j] * wr[e];
        }
        yr[k] = sr;
        yi[k] = si;
    }
}

static void radix5_split_f64(double *re, double *im, size_t n, size_t group,
                             size_t tw_step, const double *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 5;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            double ar[5], ai[5], yr[5], yi[5];
            for (int p = 0; p < 5; p++) {
                size_t ix = i0 + (size_t)p * m;
                ar[p] = re[ix];
                ai[p] = im[ix];
                if (p > 0)
                    twiddle_mul_f64(&ar[p], &ai[p], tw, j * (size_t)p * tw_step,
                                    ntw);
            }
            dft5_f64(yr, yi, ar[0], ai[0], ar[1], ai[1], ar[2], ai[2],
                     ar[3], ai[3], ar[4], ai[4], inverse);
            for (int p = 0; p < 5; p++) {
                size_t ix = i0 + (size_t)p * m;
                re[ix] = yr[p];
                im[ix] = yi[p];
            }
        }
    }
}

static void dft7_f64(double *yr, double *yi,
                     const double *ar, const double *ai, int inverse) {
    double sign = inverse ? 1.0 : -1.0;
    const double c1 = 0.62348980185873353052, s1 = 0.78183148246802980870;
    const double c2 = -0.22252093395631440428, s2 = 0.97492791218182360701;
    const double c3 = -0.90096886790241912623, s3 = 0.43388373911755812047;
    double wr[7] = { 1.0, c1, c2, c3, c3, c2, c1 };
    double wi[7] = { 0.0, sign * s1, sign * s2, sign * s3,
                     -sign * s3, -sign * s2, -sign * s1 };
    for (int k = 0; k < 7; k++) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < 7; j++) {
            int e = (j * k) % 7;
            sr += ar[j] * wr[e] - ai[j] * wi[e];
            si += ar[j] * wi[e] + ai[j] * wr[e];
        }
        yr[k] = sr;
        yi[k] = si;
    }
}

static void radix7_split_f64(double *re, double *im, size_t n, size_t group,
                             size_t tw_step, const double *tw, size_t ntw,
                             int inverse) {
    size_t m = group / 7;
    if (m == 0) return;
    size_t ngroups = n / group;
    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group;
        for (size_t j = 0; j < m; j++) {
            size_t i0 = base + j;
            double ar[7], ai[7], yr[7], yi[7];
            for (int p = 0; p < 7; p++) {
                size_t ix = i0 + (size_t)p * m;
                ar[p] = re[ix];
                ai[p] = im[ix];
                if (p > 0)
                    twiddle_mul_f64(&ar[p], &ai[p], tw, j * (size_t)p * tw_step,
                                    ntw);
            }
            dft7_f64(yr, yi, ar, ai, inverse);
            for (int p = 0; p < 7; p++) {
                size_t ix = i0 + (size_t)p * m;
                re[ix] = yr[p];
                im[ix] = yi[p];
            }
        }
    }
}

void faf_fft_stage_split_f64(double *re, double *im, size_t n,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             const double *tw, size_t ntw, int inverse) {
    if (!re || !im || n == 0 || a0 == 0) return;
    if (a1 == 3) { radix3_split_f64(re, im, n, a0, a2, tw, ntw, inverse); return; }
    if (a1 == 4) { radix4_split_f64(re, im, n, a0, a2, tw, ntw, inverse); return; }
    if (a1 == 5) { radix5_split_f64(re, im, n, a0, a2, tw, ntw, inverse); return; }
    if (a1 == 7) { radix7_split_f64(re, im, n, a0, a2, tw, ntw, inverse); return; }
    size_t stride = (a1 == 0) ? 1 : ((a1 == 2) ? 1 : a1);
    if (a1 == 1 || a1 == 2)
        stride = 1;
#ifdef FAF_ARCH_AARCH64
    if (stride == 1) {
        faf_radix2_split_neon_f64(re, im, n, a0, stride, a2, tw, ntw);
        return;
    }
#endif
    radix2_split_f64(re, im, n, a0, stride, a2, tw, ntw);
}

void faf_fft_stage_interleaved_f32(float *regs, size_t n,
                                   uint32_t a0, uint32_t a1, uint32_t a2,
                                   const float *tw, size_t ntw, int inverse) {
    if (!regs || n == 0) return;
    /* Borrow scratch from the register file's imaginary plane layout:
     * deinterleave into a pair of temp buffers on the heap. n is the
     * transform length, so 2n floats of scratch is fine at create sizes. */
    float *re = (float *)malloc(n * sizeof(float));
    float *im = (float *)malloc(n * sizeof(float));
    if (!re || !im) { free(re); free(im); return; }
    for (size_t i = 0; i < n; i++) {
        re[i] = regs[2 * i];
        im[i] = regs[2 * i + 1];
    }
    faf_fft_stage_split_f32(re, im, n, a0, a1, a2, tw, ntw, inverse);
    for (size_t i = 0; i < n; i++) {
        regs[2 * i]     = re[i];
        regs[2 * i + 1] = im[i];
    }
    free(re);
    free(im);
}

void faf_fft_stage_interleaved_f64(double *regs, size_t n,
                                   uint32_t a0, uint32_t a1, uint32_t a2,
                                   const double *tw, size_t ntw, int inverse) {
    if (!regs || n == 0) return;
    double *re = (double *)malloc(n * sizeof(double));
    double *im = (double *)malloc(n * sizeof(double));
    if (!re || !im) { free(re); free(im); return; }
    for (size_t i = 0; i < n; i++) {
        re[i] = regs[2 * i];
        im[i] = regs[2 * i + 1];
    }
    faf_fft_stage_split_f64(re, im, n, a0, a1, a2, tw, ntw, inverse);
    for (size_t i = 0; i < n; i++) {
        regs[2 * i]     = re[i];
        regs[2 * i + 1] = im[i];
    }
    free(re);
    free(im);
}
