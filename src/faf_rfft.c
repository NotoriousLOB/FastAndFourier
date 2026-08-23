/**
 * @file faf_rfft.c
 * @brief Real-to-complex FFT via even/odd packing (Kiss / Sorensen)
 *
 * A length-n real FFT (n even) is an n/2 complex FFT of
 *   z[k] = x[2k] + i x[2k+1]
 * followed by a super-twiddle post-pass that produces the packed
 * Hermitian spectrum of n/2+1 bins. Inverse is the reverse.
 *
 * The inner C2C is unscaled (execute_split, not faf_execute) so the
 * outer apply_fft_norm can use the real length n.
 */

#include "faf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int alloc_scratch_bytes(faf_transform *t, size_t bytes) {
    if (bytes == 0) return 0;
    bytes = (bytes + 63u) & ~(size_t)63u;
    t->scratch = aligned_alloc(64, bytes);
    if (!t->scratch) {
        faf_set_error("Failed to allocate R2C scratch");
        return -1;
    }
    t->scratch_size = bytes;
    memset(t->scratch, 0, bytes);
    return 0;
}

/* Kiss super-twiddles: phase = ±π ((i+1)/ncfft + 1/2), i = 0 .. ncfft/2-1 */
static void gen_super_twiddles_f32(float *tw, size_t ncfft, int inverse) {
    size_t ntw = ncfft / 2;
    for (size_t i = 0; i < ntw; i++) {
        double phase = -M_PI * (((double)(i + 1) / (double)ncfft) + 0.5);
        if (inverse) phase = -phase;
        tw[2 * i]     = (float)cos(phase);
        tw[2 * i + 1] = (float)sin(phase);
    }
}

static void gen_super_twiddles_f64(double *tw, size_t ncfft, int inverse) {
    size_t ntw = ncfft / 2;
    for (size_t i = 0; i < ntw; i++) {
        double phase = -M_PI * (((double)(i + 1) / (double)ncfft) + 0.5);
        if (inverse) phase = -phase;
        tw[2 * i]     = cos(phase);
        tw[2 * i + 1] = sin(phase);
    }
}

/*
 * Forward post-pass: packed even/odd FFT (nc complex) -> Hermitian n/2+1.
 * Matches kiss_fftr() for floating point (C_FIXDIV is a no-op; HALF_OF = *0.5).
 */
static void rfft_post_f32(float *xr, float *xi,
                          const float *fr, const float *fi,
                          const float *stw, size_t nc) {
    float tdc_r = fr[0];
    float tdc_i = fi[0];
    xr[0]  = tdc_r + tdc_i;
    xi[0]  = 0.0f;
    xr[nc] = tdc_r - tdc_i;
    xi[nc] = 0.0f;

    for (size_t k = 1; k <= nc / 2; k++) {
        float fpk_r = fr[k];
        float fpk_i = fi[k];
        float fpnk_r = fr[nc - k];
        float fpnk_i = -fi[nc - k];
        float f1_r = fpk_r + fpnk_r;
        float f1_i = fpk_i + fpnk_i;
        float f2_r = fpk_r - fpnk_r;
        float f2_i = fpk_i - fpnk_i;
        float tw_r = stw[2 * (k - 1)];
        float tw_i = stw[2 * (k - 1) + 1];
        float wr = f2_r * tw_r - f2_i * tw_i;
        float wi = f2_r * tw_i + f2_i * tw_r;
        xr[k]      = 0.5f * (f1_r + wr);
        xi[k]      = 0.5f * (f1_i + wi);
        xr[nc - k] = 0.5f * (f1_r - wr);
        xi[nc - k] = 0.5f * (wi - f1_i);
    }
}

static void rfft_post_f64(double *xr, double *xi,
                          const double *fr, const double *fi,
                          const double *stw, size_t nc) {
    double tdc_r = fr[0];
    double tdc_i = fi[0];
    xr[0]  = tdc_r + tdc_i;
    xi[0]  = 0.0;
    xr[nc] = tdc_r - tdc_i;
    xi[nc] = 0.0;

    for (size_t k = 1; k <= nc / 2; k++) {
        double fpk_r = fr[k];
        double fpk_i = fi[k];
        double fpnk_r = fr[nc - k];
        double fpnk_i = -fi[nc - k];
        double f1_r = fpk_r + fpnk_r;
        double f1_i = fpk_i + fpnk_i;
        double f2_r = fpk_r - fpnk_r;
        double f2_i = fpk_i - fpnk_i;
        double tw_r = stw[2 * (k - 1)];
        double tw_i = stw[2 * (k - 1) + 1];
        double wr = f2_r * tw_r - f2_i * tw_i;
        double wi = f2_r * tw_i + f2_i * tw_r;
        xr[k]      = 0.5 * (f1_r + wr);
        xi[k]      = 0.5 * (f1_i + wi);
        xr[nc - k] = 0.5 * (f1_r - wr);
        xi[nc - k] = 0.5 * (wi - f1_i);
    }
}

/* Inverse pre-pass: Hermitian n/2+1 -> packed even/odd spectrum (nc complex). */
static void irfft_pre_f32(float *fr, float *fi,
                          const float *xr, const float *xi,
                          const float *stw, size_t nc) {
    fr[0] = xr[0] + xr[nc];
    fi[0] = xr[0] - xr[nc];

    for (size_t k = 1; k <= nc / 2; k++) {
        float fk_r = xr[k];
        float fk_i = xi[k];
        float fn_r = xr[nc - k];
        float fn_i = -xi[nc - k];
        float fek_r = fk_r + fn_r;
        float fek_i = fk_i + fn_i;
        float tmp_r = fk_r - fn_r;
        float tmp_i = fk_i - fn_i;
        float tw_r = stw[2 * (k - 1)];
        float tw_i = stw[2 * (k - 1) + 1];
        float fok_r = tmp_r * tw_r - tmp_i * tw_i;
        float fok_i = tmp_r * tw_i + tmp_i * tw_r;
        fr[k]      = fek_r + fok_r;
        fi[k]      = fek_i + fok_i;
        fr[nc - k] = fek_r - fok_r;
        fi[nc - k] = -(fek_i - fok_i);
    }
}

static void irfft_pre_f64(double *fr, double *fi,
                          const double *xr, const double *xi,
                          const double *stw, size_t nc) {
    fr[0] = xr[0] + xr[nc];
    fi[0] = xr[0] - xr[nc];

    for (size_t k = 1; k <= nc / 2; k++) {
        double fk_r = xr[k];
        double fk_i = xi[k];
        double fn_r = xr[nc - k];
        double fn_i = -xi[nc - k];
        double fek_r = fk_r + fn_r;
        double fek_i = fk_i + fn_i;
        double tmp_r = fk_r - fn_r;
        double tmp_i = fk_i - fn_i;
        double tw_r = stw[2 * (k - 1)];
        double tw_i = stw[2 * (k - 1) + 1];
        double fok_r = tmp_r * tw_r - tmp_i * tw_i;
        double fok_i = tmp_r * tw_i + tmp_i * tw_r;
        fr[k]      = fek_r + fok_r;
        fi[k]      = fek_i + fok_i;
        fr[nc - k] = fek_r - fok_r;
        fi[nc - k] = -(fek_i - fok_i);
    }
}

faf_transform* faf_create_rfft(const faf_config *cfg) {
    if (!cfg) {
        faf_set_error("faf_create_rfft: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.n < 2 || (c.n & 1u)) {
        faf_set_error("R2C size must be even and >= 2, got %zu", c.n);
        return NULL;
    }
    if (!faf_is_5_smooth(c.n)) {
        faf_set_error("R2C size must be even and 5-smooth, got %zu; nearest is %zu",
                      c.n, faf_get_recommended_size(FAF_TRANSFORM_RFFT, c.n));
        return NULL;
    }
    if (c.precision != FAF_PREC_FP32 && c.precision != FAF_PREC_FP64) {
        faf_set_error("R2C supports FP32 and FP64 only");
        return NULL;
    }
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_HERMITIAN;
    if (c.norm == FAF_NORM_DEFAULT)
        c.norm = FAF_NORM_NONE;
    if (c.layout != FAF_LAYOUT_HERMITIAN &&
        c.layout != FAF_LAYOUT_INTERLEAVED) {
        faf_set_error("R2C layout must be hermitian or interleaved, got '%s'",
                      faf_layout_name(c.layout));
        return NULL;
    }
    if (c.norm == FAF_NORM_LAZY || c.norm == FAF_NORM_JPEG2000) {
        faf_set_error("R2C does not support norm '%s'", faf_norm_name(c.norm));
        return NULL;
    }

    faf_transform *t = calloc(1, sizeof(faf_transform));
    if (!t) {
        faf_set_error("Failed to allocate transform");
        return NULL;
    }

    int inverse = (c.dir == FAF_DIR_INVERSE);
    t->type = inverse ? FAF_TRANSFORM_IRFFT : FAF_TRANSFORM_RFFT;
    t->cfg = c;
    t->n = c.n;
    t->precision = c.precision;
    t->flags = c.flags | FAF_FLAG_REAL;
    if (inverse)
        t->flags |= FAF_FLAG_INVERSE;

    faf_config ic = faf_config_init(c.n / 2);
    ic.precision = c.precision;
    ic.layout = FAF_LAYOUT_SPLIT;
    ic.norm = FAF_NORM_NONE;
    ic.dir = c.dir;
    ic.backend = c.backend;
    t->inner = faf_create_fft(&ic);
    if (!t->inner) {
        faf_destroy_transform(t);
        return NULL;
    }

    size_t nc = c.n / 2;
    size_t ntw = nc / 2;
    size_t elem = (c.precision == FAF_PREC_FP64) ? sizeof(double)
                                                 : sizeof(float);
    if (ntw > 0) {
        t->twiddles[1] = malloc(ntw * 2 * elem);
        if (!t->twiddles[1]) {
            faf_set_error("Failed to allocate R2C super-twiddles");
            faf_destroy_transform(t);
            return NULL;
        }
        t->twiddle_sizes[1] = ntw;
        if (c.precision == FAF_PREC_FP64)
            gen_super_twiddles_f64((double *)t->twiddles[1], nc, inverse);
        else
            gen_super_twiddles_f32((float *)t->twiddles[1], nc, inverse);
    }

    /* pack (n/2 split) + inner FFT (n/2 split) + herm workspace (n/2+1 split) */
    size_t nbins = nc + 1;
    size_t n_elem = (nc * 4) + (nbins * 2);
    if (alloc_scratch_bytes(t, n_elem * elem) != 0) {
        faf_destroy_transform(t);
        return NULL;
    }
    return t;
}

static int rfft_check_buffers(const faf_transform *t,
                              const faf_buffer *out,
                              const faf_buffer *in) {
    if (!t->inner) {
        faf_set_error("R2C transform is missing inner FFT");
        return -1;
    }
    if (!in->re || !out->re) {
        faf_set_error("execute buffers must provide re");
        return -1;
    }
    int inverse = (t->type == FAF_TRANSFORM_IRFFT) ||
                  (t->cfg.dir == FAF_DIR_INVERSE);
    size_t n = t->n;
    size_t nbins = n / 2 + 1;
    faf_layout spec_layout = t->cfg.layout;

    if (!inverse) {
        if (in->layout != FAF_LAYOUT_REAL) {
            faf_set_error("rfft input must be REAL, got %s",
                          faf_layout_name(in->layout));
            return -1;
        }
        if (in->n != n) {
            faf_set_error("rfft input n=%zu, expected %zu", in->n, n);
            return -1;
        }
        if (out->layout != spec_layout) {
            faf_set_error("rfft output layout (%s) does not match transform (%s)",
                          faf_layout_name(out->layout),
                          faf_layout_name(spec_layout));
            return -1;
        }
        if (out->n != nbins) {
            faf_set_error("rfft output n=%zu, expected %zu bins", out->n, nbins);
            return -1;
        }
        if (spec_layout == FAF_LAYOUT_HERMITIAN && !out->im) {
            faf_set_error("hermitian rfft output requires im");
            return -1;
        }
    } else {
        if (in->layout != spec_layout) {
            faf_set_error("irfft input layout (%s) does not match transform (%s)",
                          faf_layout_name(in->layout),
                          faf_layout_name(spec_layout));
            return -1;
        }
        if (in->n != nbins) {
            faf_set_error("irfft input n=%zu, expected %zu bins", in->n, nbins);
            return -1;
        }
        if (spec_layout == FAF_LAYOUT_HERMITIAN && !in->im) {
            faf_set_error("hermitian irfft input requires im");
            return -1;
        }
        if (out->layout != FAF_LAYOUT_REAL) {
            faf_set_error("irfft output must be REAL, got %s",
                          faf_layout_name(out->layout));
            return -1;
        }
        if (out->n != n) {
            faf_set_error("irfft output n=%zu, expected %zu", out->n, n);
            return -1;
        }
    }
    return 0;
}

static int rfft_execute_f32(const faf_transform *t,
                            faf_buffer *out,
                            const faf_buffer *in) {
    size_t n = t->n;
    size_t nc = n / 2;
    size_t nbins = nc + 1;
    float *scratch = (float *)t->scratch;
    float *pack_re = scratch;
    float *pack_im = pack_re + nc;
    float *fft_re  = pack_im + nc;
    float *fft_im  = fft_re + nc;
    float *hr      = fft_im + nc;
    float *hi      = hr + nbins;
    const float *stw = (const float *)t->twiddles[1];
    int inverse = (t->type == FAF_TRANSFORM_IRFFT) ||
                  (t->cfg.dir == FAF_DIR_INVERSE);

    if (!inverse) {
        const float *x = (const float *)in->re;
        for (size_t k = 0; k < nc; k++) {
            pack_re[k] = x[2 * k];
            pack_im[k] = x[2 * k + 1];
        }
        if (faf_execute_split_f32(t->inner, fft_re, fft_im,
                                  pack_re, pack_im) != 0)
            return -1;
        rfft_post_f32(hr, hi, fft_re, fft_im, stw, nc);
        if (out->layout == FAF_LAYOUT_INTERLEAVED) {
            float *dst = (float *)out->re;
            for (size_t k = 0; k < nbins; k++) {
                dst[2 * k]     = hr[k];
                dst[2 * k + 1] = hi[k];
            }
        } else {
            memcpy(out->re, hr, nbins * sizeof(float));
            memcpy(out->im, hi, nbins * sizeof(float));
        }
        return 0;
    }

    if (in->layout == FAF_LAYOUT_INTERLEAVED) {
        const float *src = (const float *)in->re;
        for (size_t k = 0; k < nbins; k++) {
            hr[k] = src[2 * k];
            hi[k] = src[2 * k + 1];
        }
    } else {
        memcpy(hr, in->re, nbins * sizeof(float));
        memcpy(hi, in->im, nbins * sizeof(float));
    }
    irfft_pre_f32(fft_re, fft_im, hr, hi, stw, nc);
    if (faf_execute_split_f32(t->inner, pack_re, pack_im,
                              fft_re, fft_im) != 0)
        return -1;
    float *y = (float *)out->re;
    for (size_t k = 0; k < nc; k++) {
        y[2 * k]     = pack_re[k];
        y[2 * k + 1] = pack_im[k];
    }
    return 0;
}

static int rfft_execute_f64(const faf_transform *t,
                            faf_buffer *out,
                            const faf_buffer *in) {
    size_t n = t->n;
    size_t nc = n / 2;
    size_t nbins = nc + 1;
    double *scratch = (double *)t->scratch;
    double *pack_re = scratch;
    double *pack_im = pack_re + nc;
    double *fft_re  = pack_im + nc;
    double *fft_im  = fft_re + nc;
    double *hr      = fft_im + nc;
    double *hi      = hr + nbins;
    const double *stw = (const double *)t->twiddles[1];
    int inverse = (t->type == FAF_TRANSFORM_IRFFT) ||
                  (t->cfg.dir == FAF_DIR_INVERSE);

    if (!inverse) {
        const double *x = (const double *)in->re;
        for (size_t k = 0; k < nc; k++) {
            pack_re[k] = x[2 * k];
            pack_im[k] = x[2 * k + 1];
        }
        if (faf_execute_split_f64(t->inner, fft_re, fft_im,
                                  pack_re, pack_im) != 0)
            return -1;
        rfft_post_f64(hr, hi, fft_re, fft_im, stw, nc);
        if (out->layout == FAF_LAYOUT_INTERLEAVED) {
            double *dst = (double *)out->re;
            for (size_t k = 0; k < nbins; k++) {
                dst[2 * k]     = hr[k];
                dst[2 * k + 1] = hi[k];
            }
        } else {
            memcpy(out->re, hr, nbins * sizeof(double));
            memcpy(out->im, hi, nbins * sizeof(double));
        }
        return 0;
    }

    if (in->layout == FAF_LAYOUT_INTERLEAVED) {
        const double *src = (const double *)in->re;
        for (size_t k = 0; k < nbins; k++) {
            hr[k] = src[2 * k];
            hi[k] = src[2 * k + 1];
        }
    } else {
        memcpy(hr, in->re, nbins * sizeof(double));
        memcpy(hi, in->im, nbins * sizeof(double));
    }
    irfft_pre_f64(fft_re, fft_im, hr, hi, stw, nc);
    if (faf_execute_split_f64(t->inner, pack_re, pack_im,
                              fft_re, fft_im) != 0)
        return -1;
    double *y = (double *)out->re;
    for (size_t k = 0; k < nc; k++) {
        y[2 * k]     = pack_re[k];
        y[2 * k + 1] = pack_im[k];
    }
    return 0;
}

int faf_rfft_execute(const faf_transform *t, faf_buffer *out,
                     const faf_buffer *in) {
    if (rfft_check_buffers(t, out, in) != 0)
        return -1;
    if (t->precision == FAF_PREC_FP64)
        return rfft_execute_f64(t, out, in);
    return rfft_execute_f32(t, out, in);
}
