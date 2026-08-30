/**
 * @file faf_dwt_fir.c
 * @brief Polyphase FIR discrete wavelet transform backend
 *
 * Mallat analysis with periodic extension:
 *   a[k] = sum_i h[i] x[(2k + i) mod n]
 *   d[k] = sum_i g[i] x[(2k + i) mod n]
 * Inverse is the adjoint synthesis filter bank. This matches the existing
 * D4 / Sym4 QMF kernels in faf_wavelets.c (not MATLAB's `dwt(..., 'per')`
 * delay convention). Naive "convolve n then keep evens" is the same sum;
 * faf_dwt_ref_conv_decim_* is the named test-only alias.
 */

#include "faf.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define FAF_INV_SQRT2    0.70710678118654752440
#define FAF_FIR_MAX_STACK_TAPS 64

static const double D4_H[4] = {
    0.4829629131445341,
    0.8365163037378079,
    0.2241438680420134,
    -0.1294095225512604
};

static const double SYM4_H[8] = {
    -0.07576571478927333,
    -0.02963552764599851,
     0.49761866763201545,
     0.8037387518059161,
     0.29785779560527736,
    -0.09921954357684722,
    -0.012603967262037833,
     0.0322231006040427
};

static void qmf_from_h(double *g, const double *h, int n) {
    for (int i = 0; i < n; i++)
        g[i] = ((i & 1) ? -1.0 : 1.0) * h[n - 1 - i];
}

static int copy_taps_f64(double *h, double *g, int *len_hg,
                         double *ht, double *gt, int *len_syn,
                         const double *hs, const double *gs, int n,
                         const double *hts, const double *gts, int ns) {
    if (!h || !g || !len_hg || n < 1) return -1;
    for (int i = 0; i < n; i++) {
        h[i] = hs[i];
        g[i] = gs[i];
    }
    *len_hg = n;
    if (ht && gt && len_syn && hts && gts && ns >= 1) {
        for (int i = 0; i < ns; i++) {
            ht[i] = hts[i];
            gt[i] = gts[i];
        }
        *len_syn = ns;
    } else if (len_syn) {
        *len_syn = 0;
    }
    return 0;
}

int faf_dwt_builtin_taps_f64(faf_wavelet_family fam, faf_wavelet_convention conv,
                             double *h, double *g, int *len_hg,
                             double *ht, double *gt, int *len_syn) {
    double gs[8];
    if (fam == FAF_WAVELET_HAAR) {
        faf_wavelet_convention c = (conv == FAF_CONV_UNSPEC)
                                   ? FAF_CONV_HAAR_ORTHO : conv;
        if (c == FAF_CONV_HAAR_LAZY) {
            const double hs[2] = {1.0, 1.0};
            const double gsv[2] = {1.0, -1.0};
            const double htv[2] = {0.5, 0.5};
            const double gtv[2] = {0.5, -0.5};
            return copy_taps_f64(h, g, len_hg, ht, gt, len_syn,
                                 hs, gsv, 2, htv, gtv, 2);
        }
        if (c == FAF_CONV_HAAR_MEAN) {
            const double hs[2] = {0.5, 0.5};
            const double gsv[2] = {1.0, -1.0};
            const double htv[2] = {1.0, 1.0};
            const double gtv[2] = {0.5, -0.5};
            return copy_taps_f64(h, g, len_hg, ht, gt, len_syn,
                                 hs, gsv, 2, htv, gtv, 2);
        }
        {
            const double hs[2] = {FAF_INV_SQRT2, FAF_INV_SQRT2};
            const double gsv[2] = {FAF_INV_SQRT2, -FAF_INV_SQRT2};
            return copy_taps_f64(h, g, len_hg, ht, gt, len_syn,
                                 hs, gsv, 2, hs, gsv, 2);
        }
    }
    if (fam == FAF_WAVELET_D4 &&
        (conv == FAF_CONV_UNSPEC || conv == FAF_CONV_D4_ORTHO)) {
        qmf_from_h(gs, D4_H, 4);
        return copy_taps_f64(h, g, len_hg, ht, gt, len_syn,
                             D4_H, gs, 4, D4_H, gs, 4);
    }
    if (fam == FAF_WAVELET_SYM4 &&
        (conv == FAF_CONV_UNSPEC || conv == FAF_CONV_SYM4_ORTHO)) {
        qmf_from_h(gs, SYM4_H, 8);
        return copy_taps_f64(h, g, len_hg, ht, gt, len_syn,
                             SYM4_H, gs, 8, SYM4_H, gs, 8);
    }
    return -1;
}

int faf_dwt_builtin_taps_f32(faf_wavelet_family fam, faf_wavelet_convention conv,
                             float *h, float *g, int *len_hg,
                             float *ht, float *gt, int *len_syn) {
    double hd[8], gd[8], htd[8], gtd[8];
    int lh = 0, ls = 0;
    if (faf_dwt_builtin_taps_f64(fam, conv, hd, gd, &lh, htd, gtd, &ls) != 0)
        return -1;
    if (!h || !g || !len_hg || lh < 1) return -1;
    for (int i = 0; i < lh; i++) {
        h[i] = (float)hd[i];
        g[i] = (float)gd[i];
    }
    *len_hg = lh;
    if (ht && gt && len_syn && ls >= 1) {
        for (int i = 0; i < ls; i++) {
            ht[i] = (float)htd[i];
            gt[i] = (float)gtd[i];
        }
        *len_syn = ls;
    } else if (len_syn) {
        *len_syn = 0;
    }
    return 0;
}

void faf_dwt_polyfir_fwd_f32(float *x, size_t n,
                              const float *h, const float *g, int len,
                              float *work) {
    if (!x || !h || !g || n < 2 || (n & 1u) || len < 1) return;
    size_t half = n / 2;
    float *heap = NULL;
    float *tmp = work;
    if (!tmp) {
        heap = (float *)malloc(n * sizeof(float));
        tmp = heap;
    }
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        float a = 0.0f, d = 0.0f;
        for (int i = 0; i < len; i++) {
            float s = x[(2 * k + (size_t)i) % n];
            a += h[i] * s;
            d += g[i] * s;
        }
        tmp[k] = a;
        tmp[half + k] = d;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(heap);
}

void faf_dwt_polyfir_inv_f32(float *x, size_t n,
                              const float *ht, const float *gt, int len,
                              float *work) {
    if (!x || !ht || !gt || n < 2 || (n & 1u) || len < 1) return;
    size_t half = n / 2;
    float *heap = NULL;
    float *tmp = work;
    if (!tmp) {
        heap = (float *)calloc(n, sizeof(float));
        tmp = heap;
    } else {
        memset(tmp, 0, n * sizeof(float));
    }
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        float a = x[k];
        float d = x[half + k];
        for (int i = 0; i < len; i++)
            tmp[(2 * k + (size_t)i) % n] += ht[i] * a + gt[i] * d;
    }
    memcpy(x, tmp, n * sizeof(float));
    free(heap);
}

void faf_dwt_polyfir_fwd_f64(double *x, size_t n,
                              const double *h, const double *g, int len,
                              double *work) {
    if (!x || !h || !g || n < 2 || (n & 1u) || len < 1) return;
    size_t half = n / 2;
    double *heap = NULL;
    double *tmp = work;
    if (!tmp) {
        heap = (double *)malloc(n * sizeof(double));
        tmp = heap;
    }
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        double a = 0.0, d = 0.0;
        for (int i = 0; i < len; i++) {
            double s = x[(2 * k + (size_t)i) % n];
            a += h[i] * s;
            d += g[i] * s;
        }
        tmp[k] = a;
        tmp[half + k] = d;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(heap);
}

void faf_dwt_polyfir_inv_f64(double *x, size_t n,
                              const double *ht, const double *gt, int len,
                              double *work) {
    if (!x || !ht || !gt || n < 2 || (n & 1u) || len < 1) return;
    size_t half = n / 2;
    double *heap = NULL;
    double *tmp = work;
    if (!tmp) {
        heap = (double *)calloc(n, sizeof(double));
        tmp = heap;
    } else {
        memset(tmp, 0, n * sizeof(double));
    }
    if (!tmp) return;
    for (size_t k = 0; k < half; k++) {
        double a = x[k];
        double d = x[half + k];
        for (int i = 0; i < len; i++)
            tmp[(2 * k + (size_t)i) % n] += ht[i] * a + gt[i] * d;
    }
    memcpy(x, tmp, n * sizeof(double));
    free(heap);
}

void faf_dwt_ref_conv_decim_f32(float *x, size_t n,
                                 const float *h, const float *g, int len) {
    faf_dwt_polyfir_fwd_f32(x, n, h, g, len, NULL);
}

void faf_dwt_ref_conv_decim_f64(double *x, size_t n,
                                 const double *h, const double *g, int len) {
    faf_dwt_polyfir_fwd_f64(x, n, h, g, len, NULL);
}

static float *fir_work_f32(const faf_transform *t, size_t n) {
    if (t && t->scratch && t->scratch_size >= n * sizeof(float))
        return (float *)t->scratch;
    return NULL;
}

static double *fir_work_f64(const faf_transform *t, size_t n) {
    if (t && t->scratch && t->scratch_size >= n * sizeof(double))
        return (double *)t->scratch;
    return NULL;
}

void faf_dwt_apply_level_f32(float *x, size_t n, const faf_transform *t,
                             faf_wavelet_family fam,
                             faf_wavelet_convention conv, int inverse) {
    if (!x || n < 2) return;
    if (t && t->cfg.dwt_backend == FAF_DWT_BACKEND_FIR) {
        float *work = fir_work_f32(t, n);
        if (t->custom_h && t->custom_g) {
            if (inverse) {
                if (t->custom_ht && t->custom_gt)
                    faf_dwt_polyfir_inv_f32(x, n, t->custom_ht, t->custom_gt,
                                            t->custom_len_syn, work);
            } else {
                faf_dwt_polyfir_fwd_f32(x, n, t->custom_h, t->custom_g,
                                        t->custom_len_hg, work);
            }
            return;
        }
        float h[8], g[8], ht[8], gt[8];
        int lh = 0, ls = 0;
        if (faf_dwt_builtin_taps_f32(fam, conv, h, g, &lh, ht, gt, &ls) == 0) {
            if (inverse)
                faf_dwt_polyfir_inv_f32(x, n, ht, gt, ls, work);
            else
                faf_dwt_polyfir_fwd_f32(x, n, h, g, lh, work);
            return;
        }
    }
    faf_dwt_level_conv_f32(x, n, fam, conv, inverse);
}

void faf_dwt_apply_level_f64(double *x, size_t n, const faf_transform *t,
                             faf_wavelet_family fam,
                             faf_wavelet_convention conv, int inverse) {
    if (!x || n < 2) return;
    if (t && t->cfg.dwt_backend == FAF_DWT_BACKEND_FIR) {
        double *work = fir_work_f64(t, n);
        if (t->custom_h && t->custom_g) {
            int len = inverse ? t->custom_len_syn : t->custom_len_hg;
            const float *ha = inverse ? t->custom_ht : t->custom_h;
            const float *ga = inverse ? t->custom_gt : t->custom_g;
            if (!ha || !ga || len < 1) return;
            double hs[FAF_FIR_MAX_STACK_TAPS];
            double gs[FAF_FIR_MAX_STACK_TAPS];
            double *h = hs;
            double *g = gs;
            int heap = 0;
            if (len > FAF_FIR_MAX_STACK_TAPS) {
                h = (double *)malloc((size_t)len * 2 * sizeof(double));
                if (!h) return;
                g = h + len;
                heap = 1;
            }
            for (int i = 0; i < len; i++) {
                h[i] = (double)ha[i];
                g[i] = (double)ga[i];
            }
            if (inverse)
                faf_dwt_polyfir_inv_f64(x, n, h, g, len, work);
            else
                faf_dwt_polyfir_fwd_f64(x, n, h, g, len, work);
            if (heap) free(h);
            return;
        }
        double h[8], g[8], ht[8], gt[8];
        int lh = 0, ls = 0;
        if (faf_dwt_builtin_taps_f64(fam, conv, h, g, &lh, ht, gt, &ls) == 0) {
            if (inverse)
                faf_dwt_polyfir_inv_f64(x, n, ht, gt, ls, work);
            else
                faf_dwt_polyfir_fwd_f64(x, n, h, g, lh, work);
            return;
        }
    }
    faf_dwt_level_conv_f64(x, n, fam, conv, inverse);
}
