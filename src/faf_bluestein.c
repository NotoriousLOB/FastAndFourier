/**
 * @file faf_bluestein.c
 * @brief Opt-in Bluestein (chirp-z) FFT for non-5-smooth n
 *
 * Lowers to two length-M mixed-radix FFTs plus chirp multiplies.
 * M = next 5-smooth ≥ 2n−1. Scratch and chirp tables are allocated
 * at create; execute does not malloc.
 */

#include "faf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t bluestein_M(size_t n) {
    size_t need = 2u * n - 1u;
    return faf_next_5_smooth(need);
}

static int alloc_scratch_bytes(faf_transform *t, size_t bytes) {
    if (bytes == 0) return 0;
    bytes = (bytes + 63u) & ~(size_t)63u;
    t->scratch = aligned_alloc(64, bytes);
    if (!t->scratch) {
        faf_set_error("Failed to allocate Bluestein scratch");
        return -1;
    }
    t->scratch_size = bytes;
    memset(t->scratch, 0, bytes);
    return 0;
}

/* φ[k] = exp(sign * i π k² / n), sign = -1 forward / +1 inverse */
static void gen_chirp_f32(float *re, float *im, size_t n, int inverse) {
    double s = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n; k++) {
        double ang = s * M_PI * ((double)k * (double)k) / (double)n;
        re[k] = (float)cos(ang);
        im[k] = (float)sin(ang);
    }
}

static void gen_chirp_f64(double *re, double *im, size_t n, int inverse) {
    double s = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n; k++) {
        double ang = s * M_PI * ((double)k * (double)k) / (double)n;
        re[k] = cos(ang);
        im[k] = sin(ang);
    }
}

int faf_fft_init_bluestein(faf_transform *t) {
    if (!t || t->n < 1) {
        faf_set_error("Bluestein: n must be >= 1");
        return -1;
    }
    if (t->precision != FAF_PREC_FP32 && t->precision != FAF_PREC_FP64) {
        faf_set_error("Bluestein supports FP32 and FP64 only");
        return -1;
    }

    size_t n = t->n;
    size_t M = bluestein_M(n);
    int inverse = (t->cfg.dir == FAF_DIR_INVERSE) ||
                  (t->type == FAF_TRANSFORM_IFFT);

    faf_config ic = faf_config_init(M);
    ic.precision = t->precision;
    ic.layout = FAF_LAYOUT_SPLIT;
    ic.norm = FAF_NORM_NONE;
    ic.dir = FAF_DIR_FORWARD;
    ic.backend = t->cfg.backend;
    t->inner = faf_create_fft(&ic);
    if (!t->inner) return -1;

    ic.dir = FAF_DIR_INVERSE;
    t->inner_inv = faf_create_fft(&ic);
    if (!t->inner_inv) return -1;

    size_t elem = (t->precision == FAF_PREC_FP64) ? sizeof(double)
                                                 : sizeof(float);
    t->twiddles[0] = malloc(2 * n * elem);
    t->twiddles[1] = malloc(2 * M * elem);
    if (!t->twiddles[0] || !t->twiddles[1]) {
        faf_set_error("Failed to allocate Bluestein chirp tables");
        return -1;
    }
    t->twiddle_sizes[0] = n;
    t->twiddle_sizes[1] = M;

    /* Scratch: a_re[M], a_im[M], A_re[M], A_im[M], plus b_re/im for init. */
    if (alloc_scratch_bytes(t, 6 * M * elem) != 0)
        return -1;

    if (t->precision == FAF_PREC_FP64) {
        double *phi_re = (double *)t->twiddles[0];
        double *phi_im = phi_re + n;
        double *B_re = (double *)t->twiddles[1];
        double *B_im = B_re + M;
        double *b_re = (double *)t->scratch;
        double *b_im = b_re + M;
        double *Btmp_re = b_im + M;
        double *Btmp_im = Btmp_re + M;

        gen_chirp_f64(phi_re, phi_im, n, inverse);

        /* b[l] = exp(−sign · i π l² / n) on 0..n-1 and M-l, else 0. */
        memset(b_re, 0, M * sizeof(double));
        memset(b_im, 0, M * sizeof(double));
        b_re[0] = 1.0;
        b_im[0] = 0.0;
        for (size_t l = 1; l < n; l++) {
            /* kernel is conjugate of φ */
            b_re[l] = phi_re[l];
            b_im[l] = -phi_im[l];
            b_re[M - l] = phi_re[l];
            b_im[M - l] = -phi_im[l];
        }

        faf_buffer inb = faf_buffer_split(b_re, b_im, M);
        faf_buffer outb = faf_buffer_split(Btmp_re, Btmp_im, M);
        if (faf_execute(t->inner, &outb, &inb) != 0)
            return -1;
        memcpy(B_re, Btmp_re, M * sizeof(double));
        memcpy(B_im, Btmp_im, M * sizeof(double));
    } else {
        float *phi_re = (float *)t->twiddles[0];
        float *phi_im = phi_re + n;
        float *B_re = (float *)t->twiddles[1];
        float *B_im = B_re + M;
        float *b_re = (float *)t->scratch;
        float *b_im = b_re + M;
        float *Btmp_re = b_im + M;
        float *Btmp_im = Btmp_re + M;

        gen_chirp_f32(phi_re, phi_im, n, inverse);

        memset(b_re, 0, M * sizeof(float));
        memset(b_im, 0, M * sizeof(float));
        b_re[0] = 1.0f;
        b_im[0] = 0.0f;
        for (size_t l = 1; l < n; l++) {
            b_re[l] = phi_re[l];
            b_im[l] = -phi_im[l];
            b_re[M - l] = phi_re[l];
            b_im[M - l] = -phi_im[l];
        }

        faf_buffer inb = faf_buffer_split(b_re, b_im, M);
        faf_buffer outb = faf_buffer_split(Btmp_re, Btmp_im, M);
        if (faf_execute(t->inner, &outb, &inb) != 0)
            return -1;
        memcpy(B_re, Btmp_re, M * sizeof(float));
        memcpy(B_im, Btmp_im, M * sizeof(float));
    }

    memset(t->scratch, 0, t->scratch_size);
    return 0;
}

static int bluestein_run_f32(const faf_transform *t, int interleaved,
                             float *out_re, float *out_im,
                             const float *in_re, const float *in_im) {
    size_t n = t->n;
    size_t M = t->inner->n;
    float *a_re = (float *)t->scratch;
    float *a_im = a_re + M;
    float *A_re = a_im + M;
    float *A_im = A_re + M;
    const float *phi_re = (const float *)t->twiddles[0];
    const float *phi_im = phi_re + n;
    const float *B_re = (const float *)t->twiddles[1];
    const float *B_im = B_re + M;

    for (size_t k = 0; k < n; k++) {
        float xr, xi;
        if (interleaved) {
            xr = in_re[2 * k];
            xi = in_re[2 * k + 1];
        } else {
            xr = in_re[k];
            xi = in_im[k];
        }
        a_re[k] = xr * phi_re[k] - xi * phi_im[k];
        a_im[k] = xr * phi_im[k] + xi * phi_re[k];
    }
    for (size_t k = n; k < M; k++) {
        a_re[k] = 0.0f;
        a_im[k] = 0.0f;
    }

    faf_buffer ain = faf_buffer_split(a_re, a_im, M);
    faf_buffer Aout = faf_buffer_split(A_re, A_im, M);
    if (faf_execute(t->inner, &Aout, &ain) != 0)
        return -1;

    for (size_t k = 0; k < M; k++) {
        float ar = A_re[k], ai = A_im[k];
        A_re[k] = ar * B_re[k] - ai * B_im[k];
        A_im[k] = ar * B_im[k] + ai * B_re[k];
    }

    if (faf_execute(t->inner_inv, &ain, &Aout) != 0)
        return -1;

    for (size_t k = 0; k < n; k++) {
        float ar = a_re[k], ai = a_im[k];
        float yr = ar * phi_re[k] - ai * phi_im[k];
        float yi = ar * phi_im[k] + ai * phi_re[k];
        if (interleaved) {
            out_re[2 * k]     = yr;
            out_re[2 * k + 1] = yi;
        } else {
            out_re[k] = yr;
            out_im[k] = yi;
        }
    }
    return 0;
}

static int bluestein_run_f64(const faf_transform *t, int interleaved,
                             double *out_re, double *out_im,
                             const double *in_re, const double *in_im) {
    size_t n = t->n;
    size_t M = t->inner->n;
    double *a_re = (double *)t->scratch;
    double *a_im = a_re + M;
    double *A_re = a_im + M;
    double *A_im = A_re + M;
    const double *phi_re = (const double *)t->twiddles[0];
    const double *phi_im = phi_re + n;
    const double *B_re = (const double *)t->twiddles[1];
    const double *B_im = B_re + M;

    for (size_t k = 0; k < n; k++) {
        double xr, xi;
        if (interleaved) {
            xr = in_re[2 * k];
            xi = in_re[2 * k + 1];
        } else {
            xr = in_re[k];
            xi = in_im[k];
        }
        a_re[k] = xr * phi_re[k] - xi * phi_im[k];
        a_im[k] = xr * phi_im[k] + xi * phi_re[k];
    }
    for (size_t k = n; k < M; k++) {
        a_re[k] = 0.0;
        a_im[k] = 0.0;
    }

    faf_buffer ain = faf_buffer_split(a_re, a_im, M);
    faf_buffer Aout = faf_buffer_split(A_re, A_im, M);
    if (faf_execute(t->inner, &Aout, &ain) != 0)
        return -1;

    for (size_t k = 0; k < M; k++) {
        double ar = A_re[k], ai = A_im[k];
        A_re[k] = ar * B_re[k] - ai * B_im[k];
        A_im[k] = ar * B_im[k] + ai * B_re[k];
    }

    if (faf_execute(t->inner_inv, &ain, &Aout) != 0)
        return -1;

    for (size_t k = 0; k < n; k++) {
        double ar = a_re[k], ai = a_im[k];
        double yr = ar * phi_re[k] - ai * phi_im[k];
        double yi = ar * phi_im[k] + ai * phi_re[k];
        if (interleaved) {
            out_re[2 * k]     = yr;
            out_re[2 * k + 1] = yi;
        } else {
            out_re[k] = yr;
            out_im[k] = yi;
        }
    }
    return 0;
}

int faf_bluestein_execute(const faf_transform *t, faf_buffer *out,
                          const faf_buffer *in) {
    if (!t || !out || !in || !t->inner || !t->inner_inv || !t->scratch) {
        faf_set_error("bluestein execute: missing transform or scratch");
        return -1;
    }
    if (in->layout != t->cfg.layout || out->layout != t->cfg.layout) {
        faf_set_error("buffer layout (%s/%s) does not match transform (%s)",
                      faf_layout_name(in->layout), faf_layout_name(out->layout),
                      faf_layout_name(t->cfg.layout));
        return -1;
    }
    if (!in->re || !out->re) {
        faf_set_error("bluestein: buffers must provide re");
        return -1;
    }
    if (in->n != t->n || out->n != t->n) {
        faf_set_error("bluestein: buffer length must be %zu", t->n);
        return -1;
    }

    int interleaved = (t->cfg.layout == FAF_LAYOUT_INTERLEAVED);
    if (!interleaved) {
        if (t->cfg.layout != FAF_LAYOUT_SPLIT) {
            faf_set_error("bluestein: layout must be split or interleaved");
            return -1;
        }
        if (!in->im || !out->im) {
            faf_set_error("split layout requires im planes");
            return -1;
        }
    }

    if (t->precision == FAF_PREC_FP64)
        return bluestein_run_f64(t, interleaved,
                                 (double *)out->re, (double *)out->im,
                                 (const double *)in->re, (const double *)in->im);
    return bluestein_run_f32(t, interleaved,
                             (float *)out->re, (float *)out->im,
                             (const float *)in->re, (const float *)in->im);
}
