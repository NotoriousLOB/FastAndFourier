/**
 * @file faf_fft_rader.c
 * @brief Rader prime-length FFT: cyclic convolution of n−1 via two FFTs
 *
 * Eligible when n ≥ 11 is prime and n−1 is 7-smooth. Inner length n−1
 * may contain a factor of 7; that is an internal size, not a public
 * 7-smooth policy. Bluestein remains the flag for everything else.
 */

#include "faf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int faf_fft_init_rader(faf_transform *t) {
    if (!t || !faf_rader_eligible(t->n)) {
        faf_set_error("Rader: n must be prime ≥ 11 with 7-smooth n-1");
        return -1;
    }
    if (t->precision != FAF_PREC_FP32 && t->precision != FAF_PREC_FP64) {
        faf_set_error("Rader supports FP32 and FP64 only");
        return -1;
    }

    size_t n = t->n;
    size_t m = n - 1;
    size_t g = faf_primitive_root(n);
    if (g == 0) {
        faf_set_error("Rader: no primitive root for %zu", n);
        return -1;
    }
    size_t ginv = 1;
    {
        /* g^{n-2} ≡ g^{-1} (mod n) */
        uint64_t r = 1, base = (uint64_t)g, e = n - 2, mod = (uint64_t)n;
        while (e) {
            if (e & 1u) r = (r * base) % mod;
            base = (base * base) % mod;
            e >>= 1;
        }
        ginv = (size_t)r;
    }

    int inverse = (t->cfg.dir == FAF_DIR_INVERSE) ||
                  (t->type == FAF_TRANSFORM_IFFT);
    size_t elem = (t->precision == FAF_PREC_FP64) ? sizeof(double)
                                                   : sizeof(float);

    faf_config ic = faf_config_init(m);
    ic.precision = t->precision;
    ic.layout = FAF_LAYOUT_SPLIT;
    ic.norm = FAF_NORM_NONE;
    ic.dir = FAF_DIR_FORWARD;
    ic.backend = t->cfg.backend;
    t->inner = faf_create_fft_ex(&ic, 1);
    if (!t->inner) return -1;
    ic.dir = FAF_DIR_INVERSE;
    t->inner_inv = faf_create_fft_ex(&ic, 1);
    if (!t->inner_inv) return -1;

    /* twiddles[2]: rader_in[m] then rader_out[m] */
    size_t idx_bytes = 2 * m * sizeof(int);
    idx_bytes = (idx_bytes + 63u) & ~(size_t)63u;
    t->twiddles[2] = aligned_alloc(64, idx_bytes);
    if (!t->twiddles[2]) {
        faf_set_error("Rader: index alloc failed");
        return -1;
    }
    memset(t->twiddles[2], 0, idx_bytes);
    t->twiddle_sizes[2] = 2 * m;
    int *rin = (int *)t->twiddles[2];
    int *rout = rin + m;
    size_t p = 1, q = 1;
    for (size_t j = 0; j < m; j++) {
        rin[j] = (int)p;
        rout[j] = (int)q;
        p = (size_t)(((uint64_t)p * (uint64_t)g) % (uint64_t)n);
        q = (size_t)(((uint64_t)q * (uint64_t)ginv) % (uint64_t)n);
    }

    size_t b_bytes = m * elem;
    b_bytes = (b_bytes + 63u) & ~(size_t)63u;
    t->twiddles[0] = aligned_alloc(64, b_bytes);
    t->twiddles[1] = aligned_alloc(64, b_bytes);
    if (!t->twiddles[0] || !t->twiddles[1]) {
        faf_set_error("Rader: B alloc failed");
        return -1;
    }
    memset(t->twiddles[0], 0, b_bytes);
    memset(t->twiddles[1], 0, b_bytes);
    t->twiddle_sizes[0] = m;
    t->twiddle_sizes[1] = m;

    /* work: m re + m im */
    size_t work_bytes = 2 * m * elem;
    work_bytes = (work_bytes + 63u) & ~(size_t)63u;
    t->twiddles[3] = aligned_alloc(64, work_bytes);
    if (!t->twiddles[3]) {
        faf_set_error("Rader: work alloc failed");
        return -1;
    }
    memset(t->twiddles[3], 0, work_bytes);
    t->twiddle_sizes[3] = 2 * m;

    double sign = inverse ? 1.0 : -1.0;
    if (t->precision == FAF_PREC_FP64) {
        double *bre = (double *)t->twiddles[0];
        double *bim = (double *)t->twiddles[1];
        for (size_t j = 0; j < m; j++) {
            double ang = sign * 2.0 * M_PI * (double)rout[j] / (double)n;
            bre[j] = cos(ang);
            bim[j] = sin(ang);
        }
        if (faf_execute_split_f64(t->inner, bre, bim, bre, bim) != 0)
            return -1;
    } else {
        float *bre = (float *)t->twiddles[0];
        float *bim = (float *)t->twiddles[1];
        for (size_t j = 0; j < m; j++) {
            double ang = sign * 2.0 * M_PI * (double)rout[j] / (double)n;
            bre[j] = (float)cos(ang);
            bim[j] = (float)sin(ang);
        }
        if (faf_execute_split_f32(t->inner, bre, bim, bre, bim) != 0)
            return -1;
    }

    t->execute_func = faf_fft_rader_execute;
    return 0;
}

int faf_fft_rader_execute(const faf_transform *t,
                          void *out_re, void *out_im,
                          const void *in_re, const void *in_im) {
    if (!t || !t->inner || !t->inner_inv || !t->twiddles[0] ||
        !t->twiddles[1] || !t->twiddles[2] || !t->twiddles[3])
        return -1;

    size_t n = t->n;
    size_t m = n - 1;
    const int *rin = (const int *)t->twiddles[2];
    const int *rout = rin + m;

    if (t->precision == FAF_PREC_FP64) {
        const double *xr = (const double *)in_re;
        const double *xi = (const double *)in_im;
        double *yr = (double *)out_re;
        double *yi = (double *)out_im;
        double *ar = (double *)t->twiddles[3];
        double *ai = ar + m;
        const double *br = (const double *)t->twiddles[0];
        const double *bi = (const double *)t->twiddles[1];

        double x0r = xr[0], x0i = xi[0];
        double dcr = 0.0, dci = 0.0;
        for (size_t i = 0; i < n; i++) {
            dcr += xr[i];
            dci += xi[i];
        }
        for (size_t j = 0; j < m; j++) {
            size_t s = (size_t)rin[j];
            ar[j] = xr[s];
            ai[j] = xi[s];
        }
        if (faf_execute_split_f64(t->inner, ar, ai, ar, ai) != 0)
            return -1;
        for (size_t j = 0; j < m; j++) {
            double rr = ar[j], ii = ai[j];
            ar[j] = rr * br[j] - ii * bi[j];
            ai[j] = rr * bi[j] + ii * br[j];
        }
        if (faf_execute_split_f64(t->inner_inv, ar, ai, ar, ai) != 0)
            return -1;
        double scale = 1.0 / (double)m;
        /* Inverse inner does not apply 1/m (execute_split, NORM_NONE). */
        if (out_re != in_re) memcpy(yr, xr, n * sizeof(double));
        if (out_im != in_im) memcpy(yi, xi, n * sizeof(double));
        yr[0] = dcr;
        yi[0] = dci;
        for (size_t j = 0; j < m; j++) {
            size_t d = (size_t)rout[j];
            yr[d] = x0r + ar[j] * scale;
            yi[d] = x0i + ai[j] * scale;
        }
        return 0;
    }

    const float *xr = (const float *)in_re;
    const float *xi = (const float *)in_im;
    float *yr = (float *)out_re;
    float *yi = (float *)out_im;
    float *ar = (float *)t->twiddles[3];
    float *ai = ar + m;
    const float *br = (const float *)t->twiddles[0];
    const float *bi = (const float *)t->twiddles[1];

    float x0r = xr[0], x0i = xi[0];
    float dcr = 0.0f, dci = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dcr += xr[i];
        dci += xi[i];
    }
    for (size_t j = 0; j < m; j++) {
        size_t s = (size_t)rin[j];
        ar[j] = xr[s];
        ai[j] = xi[s];
    }
    if (faf_execute_split_f32(t->inner, ar, ai, ar, ai) != 0)
        return -1;
    for (size_t j = 0; j < m; j++) {
        float rr = ar[j], ii = ai[j];
        ar[j] = rr * br[j] - ii * bi[j];
        ai[j] = rr * bi[j] + ii * br[j];
    }
    if (faf_execute_split_f32(t->inner_inv, ar, ai, ar, ai) != 0)
        return -1;
    float scale = 1.0f / (float)m;
    if (out_re != in_re) memcpy(yr, xr, n * sizeof(float));
    if (out_im != in_im) memcpy(yi, xi, n * sizeof(float));
    yr[0] = dcr;
    yi[0] = dci;
    for (size_t j = 0; j < m; j++) {
        size_t d = (size_t)rout[j];
        yr[d] = x0r + ar[j] * scale;
        yi[d] = x0i + ai[j] * scale;
    }
    return 0;
}
