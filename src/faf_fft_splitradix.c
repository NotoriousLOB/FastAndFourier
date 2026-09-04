/**
 * @file faf_fft_splitradix.c
 * @brief Recursive split-radix DIF on split planes (minfft / Notorious layout)
 *
 * Power-of-two N >= 16. Leaves are the N=2/4/8 codelets. Output is in
 * natural order — no bit-reversal. Twiddles live in t->twiddles[0];
 * a 2n-element work pair lives in t->twiddles[1] so interleaved
 * execute scratch cannot alias the recursion buffer.
 */

#include "faf.h"
#include <math.h>
#include <string.h>

#ifdef FAF_ARCH_AARCH64
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

size_t faf_sr_twiddle_count(size_t n) {
    size_t count = 0;
    for (size_t m = n; m >= 16; m >>= 1)
        count += m;
    return count;
}

void faf_gen_sr_twiddles_f32(float *tw, size_t n, int inverse) {
    double sign = inverse ? 1.0 : -1.0;
    size_t off = 0;
    for (size_t m = n; m >= 16; m >>= 1) {
        for (size_t k = 0; k < m / 4; k++) {
            double a1 = sign * 2.0 * M_PI * (double)k / (double)m;
            double a3 = 3.0 * a1;
            tw[off++] = (float)cos(a1);
            tw[off++] = (float)sin(a1);
            tw[off++] = (float)cos(a3);
            tw[off++] = (float)sin(a3);
        }
    }
}

void faf_gen_sr_twiddles_f64(double *tw, size_t n, int inverse) {
    double sign = inverse ? 1.0 : -1.0;
    size_t off = 0;
    for (size_t m = n; m >= 16; m >>= 1) {
        for (size_t k = 0; k < m / 4; k++) {
            double a1 = sign * 2.0 * M_PI * (double)k / (double)m;
            double a3 = 3.0 * a1;
            tw[off++] = cos(a1);
            tw[off++] = sin(a1);
            tw[off++] = cos(a3);
            tw[off++] = sin(a3);
        }
    }
}

/*
 * Self-sorting split-radix DIF (Mukhin / minfft, Notorious sr_dif_cx).
 *
 *   t0 = x[k] + x[k+n/2]
 *   t1 = x[k+n/4] + x[k+3n/4]
 *   t2 = x[k] - x[k+n/2]
 *   t3 = ±j * (x[k+n/4] - x[k+3n/4])   +j forward, -j inverse
 *
 *   t[k]       = t0
 *   t[k+n/4]   = t1
 *   t[k+n/2]   = (t2 - t3) * W_n^k
 *   t[k+3n/4]  = (t2 + t3) * W_n^{3k}
 *
 * Then DFT_{n/2}(t[0..])  → y[0, 2, 4, ...]
 *      DFT_{n/4}(t[n/2..]) → y[1, 5, 9, ...]
 *      DFT_{n/4}(t[3n/4..]) → y[3, 7, 11, ...]
 */

#ifdef FAF_ARCH_AARCH64
static void sr_stage_neon_f32(size_t n, size_t n4, size_t n2,
                              const float *xre, const float *xim,
                              float *tre, float *tim,
                              const float *tw, int inverse) {
    size_t k = 0;
    size_t n3 = 3 * n4;
    for (; k + 3 < n4; k += 4) {
        float32x4_t x0r = vld1q_f32(xre + k);
        float32x4_t x0i = vld1q_f32(xim + k);
        float32x4_t x1r = vld1q_f32(xre + k + n2);
        float32x4_t x1i = vld1q_f32(xim + k + n2);
        float32x4_t x2r = vld1q_f32(xre + k + n4);
        float32x4_t x2i = vld1q_f32(xim + k + n4);
        float32x4_t x3r = vld1q_f32(xre + k + n3);
        float32x4_t x3i = vld1q_f32(xim + k + n3);

        float32x4_t t0r = vaddq_f32(x0r, x1r), t0i = vaddq_f32(x0i, x1i);
        float32x4_t t1r = vaddq_f32(x2r, x3r), t1i = vaddq_f32(x2i, x3i);
        float32x4_t t2r = vsubq_f32(x0r, x1r), t2i = vsubq_f32(x0i, x1i);
        float32x4_t t3r, t3i;
        if (!inverse) {
            t3r = vsubq_f32(x3i, x2i);
            t3i = vsubq_f32(x2r, x3r);
        } else {
            t3r = vsubq_f32(x2i, x3i);
            t3i = vsubq_f32(x3r, x2r);
        }

        vst1q_f32(tre + k, t0r);
        vst1q_f32(tim + k, t0i);
        vst1q_f32(tre + k + n4, t1r);
        vst1q_f32(tim + k + n4, t1i);

        float32x4x4_t w = vld4q_f32(tw + 4 * k);
        float32x4_t w1r = w.val[0], w1i = w.val[1];
        float32x4_t w3r = w.val[2], w3i = w.val[3];

        float32x4_t u0r = vsubq_f32(t2r, t3r), u0i = vsubq_f32(t2i, t3i);
        float32x4_t pr = vfmsq_f32(vmulq_f32(u0r, w1r), u0i, w1i);
        float32x4_t pi = vfmaq_f32(vmulq_f32(u0r, w1i), u0i, w1r);
        vst1q_f32(tre + k + n2, pr);
        vst1q_f32(tim + k + n2, pi);

        float32x4_t u1r = vaddq_f32(t2r, t3r), u1i = vaddq_f32(t2i, t3i);
        float32x4_t qr = vfmsq_f32(vmulq_f32(u1r, w3r), u1i, w3i);
        float32x4_t qi = vfmaq_f32(vmulq_f32(u1r, w3i), u1i, w3r);
        vst1q_f32(tre + k + n3, qr);
        vst1q_f32(tim + k + n3, qi);
    }
    for (; k < n4; k++) {
        float x0r = xre[k], x0i = xim[k];
        float x1r = xre[k + n2], x1i = xim[k + n2];
        float x2r = xre[k + n4], x2i = xim[k + n4];
        float x3r = xre[k + n3], x3i = xim[k + n3];
        float t0r = x0r + x1r, t0i = x0i + x1i;
        float t1r = x2r + x3r, t1i = x2i + x3i;
        float t2r = x0r - x1r, t2i = x0i - x1i;
        float t3r, t3i;
        if (!inverse) { t3r = - (x2i - x3i); t3i = x2r - x3r; }
        else          { t3r =   (x2i - x3i); t3i = -(x2r - x3r); }
        tre[k] = t0r; tim[k] = t0i;
        tre[k + n4] = t1r; tim[k + n4] = t1i;
        float w1r = tw[4 * k], w1i = tw[4 * k + 1];
        float w3r = tw[4 * k + 2], w3i = tw[4 * k + 3];
        float u0r = t2r - t3r, u0i = t2i - t3i;
        tre[k + n2] = u0r * w1r - u0i * w1i;
        tim[k + n2] = u0r * w1i + u0i * w1r;
        float u1r = t2r + t3r, u1i = t2i + t3i;
        tre[k + n3] = u1r * w3r - u1i * w3i;
        tim[k + n3] = u1r * w3i + u1i * w3r;
    }
    (void)n;
}

static void sr_stage_neon_f64(size_t n, size_t n4, size_t n2,
                              const double *xre, const double *xim,
                              double *tre, double *tim,
                              const double *tw, int inverse) {
    size_t k = 0;
    size_t n3 = 3 * n4;
    for (; k + 1 < n4; k += 2) {
        __builtin_prefetch(xre + k + 8, 0, 3);
        __builtin_prefetch(xim + k + 8, 0, 3);
        __builtin_prefetch(xre + k + n2 + 8, 0, 3);
        __builtin_prefetch(xim + k + n2 + 8, 0, 3);
        float64x2_t x0r = vld1q_f64(xre + k);
        float64x2_t x0i = vld1q_f64(xim + k);
        float64x2_t x1r = vld1q_f64(xre + k + n2);
        float64x2_t x1i = vld1q_f64(xim + k + n2);
        float64x2_t x2r = vld1q_f64(xre + k + n4);
        float64x2_t x2i = vld1q_f64(xim + k + n4);
        float64x2_t x3r = vld1q_f64(xre + k + n3);
        float64x2_t x3i = vld1q_f64(xim + k + n3);

        float64x2_t t0r = vaddq_f64(x0r, x1r), t0i = vaddq_f64(x0i, x1i);
        float64x2_t t1r = vaddq_f64(x2r, x3r), t1i = vaddq_f64(x2i, x3i);
        float64x2_t t2r = vsubq_f64(x0r, x1r), t2i = vsubq_f64(x0i, x1i);
        float64x2_t t3r, t3i;
        if (!inverse) {
            t3r = vsubq_f64(x3i, x2i);
            t3i = vsubq_f64(x2r, x3r);
        } else {
            t3r = vsubq_f64(x2i, x3i);
            t3i = vsubq_f64(x3r, x2r);
        }

        vst1q_f64(tre + k, t0r);
        vst1q_f64(tim + k, t0i);
        vst1q_f64(tre + k + n4, t1r);
        vst1q_f64(tim + k + n4, t1i);

        float64x2_t a = vld1q_f64(tw + 4 * k);
        float64x2_t b = vld1q_f64(tw + 4 * k + 2);
        float64x2_t c = vld1q_f64(tw + 4 * (k + 1));
        float64x2_t d = vld1q_f64(tw + 4 * (k + 1) + 2);
        float64x2_t w1r = vuzp1q_f64(a, c);
        float64x2_t w1i = vuzp2q_f64(a, c);
        float64x2_t w3r = vuzp1q_f64(b, d);
        float64x2_t w3i = vuzp2q_f64(b, d);

        float64x2_t u0r = vsubq_f64(t2r, t3r), u0i = vsubq_f64(t2i, t3i);
        float64x2_t pr = vfmsq_f64(vmulq_f64(u0r, w1r), u0i, w1i);
        float64x2_t pi = vfmaq_f64(vmulq_f64(u0r, w1i), u0i, w1r);
        vst1q_f64(tre + k + n2, pr);
        vst1q_f64(tim + k + n2, pi);

        float64x2_t u1r = vaddq_f64(t2r, t3r), u1i = vaddq_f64(t2i, t3i);
        float64x2_t qr = vfmsq_f64(vmulq_f64(u1r, w3r), u1i, w3i);
        float64x2_t qi = vfmaq_f64(vmulq_f64(u1r, w3i), u1i, w3r);
        vst1q_f64(tre + k + n3, qr);
        vst1q_f64(tim + k + n3, qi);
    }
    for (; k < n4; k++) {
        double x0r = xre[k], x0i = xim[k];
        double x1r = xre[k + n2], x1i = xim[k + n2];
        double x2r = xre[k + n4], x2i = xim[k + n4];
        double x3r = xre[k + n3], x3i = xim[k + n3];
        double t0r = x0r + x1r, t0i = x0i + x1i;
        double t1r = x2r + x3r, t1i = x2i + x3i;
        double t2r = x0r - x1r, t2i = x0i - x1i;
        double t3r, t3i;
        if (!inverse) { t3r = -(x2i - x3i); t3i = x2r - x3r; }
        else          { t3r =  (x2i - x3i); t3i = -(x2r - x3r); }
        tre[k] = t0r; tim[k] = t0i;
        tre[k + n4] = t1r; tim[k + n4] = t1i;
        double w1r = tw[4 * k], w1i = tw[4 * k + 1];
        double w3r = tw[4 * k + 2], w3i = tw[4 * k + 3];
        double u0r = t2r - t3r, u0i = t2i - t3i;
        tre[k + n2] = u0r * w1r - u0i * w1i;
        tim[k + n2] = u0r * w1i + u0i * w1r;
        double u1r = t2r + t3r, u1i = t2i + t3i;
        tre[k + n3] = u1r * w3r - u1i * w3i;
        tim[k + n3] = u1r * w3i + u1i * w3r;
    }
    (void)n;
}
#endif /* FAF_ARCH_AARCH64 */

static void sr_dif_f32(size_t n,
                       const float *xre, const float *xim,
                       float *tre, float *tim,
                       float *yre, float *yim, size_t sy,
                       const float *tw, int inverse) {
    if (n == 1) {
        yre[0] = xre[0];
        yim[0] = xim[0];
        return;
    }
    if (n == 2) {
        float r0 = xre[0], i0 = xim[0];
        float r1 = xre[1], i1 = xim[1];
        yre[0] = r0 + r1; yim[0] = i0 + i1;
        yre[sy] = r0 - r1; yim[sy] = i0 - i1;
        return;
    }
    if (n == 4) {
        float re[4], im[4];
        re[0] = xre[0]; im[0] = xim[0];
        re[1] = xre[1]; im[1] = xim[1];
        re[2] = xre[2]; im[2] = xim[2];
        re[3] = xre[3]; im[3] = xim[3];
        faf_fft_kernel_4_f32(re, im, inverse);
        yre[0] = re[0];           yim[0] = im[0];
        yre[sy] = re[1];          yim[sy] = im[1];
        yre[2 * sy] = re[2];      yim[2 * sy] = im[2];
        yre[3 * sy] = re[3];      yim[3 * sy] = im[3];
        return;
    }
    if (n == 8) {
        float re[8], im[8];
        for (size_t i = 0; i < 8; i++) {
            re[i] = xre[i];
            im[i] = xim[i];
        }
        faf_fft_kernel_8_f32(re, im, inverse);
        for (size_t i = 0; i < 8; i++) {
            yre[i * sy] = re[i];
            yim[i * sy] = im[i];
        }
        return;
    }
#ifdef FAF_ARCH_AARCH64
    if (n == 16) {
        float t_re[16], t_im[16];
        sr_stage_neon_f32(16, 4, 8, xre, xim, t_re, t_im, tw, inverse);
        {
            float re[8], im[8];
            memcpy(re, t_re, 8 * sizeof(float));
            memcpy(im, t_im, 8 * sizeof(float));
            faf_fft_kernel_8_f32(re, im, inverse);
            for (size_t i = 0; i < 8; i++) {
                yre[i * 2 * sy] = re[i];
                yim[i * 2 * sy] = im[i];
            }
        }
        {
            float re[4], im[4];
            memcpy(re, t_re + 8, 4 * sizeof(float));
            memcpy(im, t_im + 8, 4 * sizeof(float));
            faf_fft_kernel_4_f32(re, im, inverse);
            yre[sy] = re[0];            yim[sy] = im[0];
            yre[sy + 4 * sy] = re[1];   yim[sy + 4 * sy] = im[1];
            yre[sy + 8 * sy] = re[2];   yim[sy + 8 * sy] = im[2];
            yre[sy + 12 * sy] = re[3];  yim[sy + 12 * sy] = im[3];
        }
        {
            float re[4], im[4];
            memcpy(re, t_re + 12, 4 * sizeof(float));
            memcpy(im, t_im + 12, 4 * sizeof(float));
            faf_fft_kernel_4_f32(re, im, inverse);
            yre[3 * sy] = re[0];            yim[3 * sy] = im[0];
            yre[3 * sy + 4 * sy] = re[1];   yim[3 * sy + 4 * sy] = im[1];
            yre[3 * sy + 8 * sy] = re[2];   yim[3 * sy + 8 * sy] = im[2];
            yre[3 * sy + 12 * sy] = re[3];  yim[3 * sy + 12 * sy] = im[3];
        }
        return;
    }
#endif

    size_t n4 = n / 4;
    size_t n2 = n / 2;
#ifdef FAF_ARCH_AARCH64
    sr_stage_neon_f32(n, n4, n2, xre, xim, tre, tim, tw, inverse);
#else
    for (size_t k = 0; k < n4; k++) {
        float x0r = xre[k],           x0i = xim[k];
        float x1r = xre[k + n2],      x1i = xim[k + n2];
        float x2r = xre[k + n4],      x2i = xim[k + n4];
        float x3r = xre[k + 3 * n4],  x3i = xim[k + 3 * n4];

        float t0r = x0r + x1r, t0i = x0i + x1i;
        float t1r = x2r + x3r, t1i = x2i + x3i;
        float t2r = x0r - x1r, t2i = x0i - x1i;
        float diffr = x2r - x3r, diffi = x2i - x3i;

        /* ±j * diff */
        float t3r, t3i;
        if (!inverse) {
            t3r = -diffi;
            t3i =  diffr;
        } else {
            t3r =  diffi;
            t3i = -diffr;
        }

        tre[k] = t0r;
        tim[k] = t0i;
        tre[k + n4] = t1r;
        tim[k + n4] = t1i;

        float w1r = tw[4 * k],     w1i = tw[4 * k + 1];
        float w3r = tw[4 * k + 2], w3i = tw[4 * k + 3];

        float u0r = t2r - t3r, u0i = t2i - t3i;
        tre[k + n2] = u0r * w1r - u0i * w1i;
        tim[k + n2] = u0r * w1i + u0i * w1r;

        float u1r = t2r + t3r, u1i = t2i + t3i;
        tre[k + 3 * n4] = u1r * w3r - u1i * w3i;
        tim[k + 3 * n4] = u1r * w3i + u1i * w3r;
    }
#endif

    const float *tw_next = tw + n;
    sr_dif_f32(n2, tre, tim, tre, tim,
               yre, yim, 2 * sy, tw_next, inverse);
    sr_dif_f32(n4, tre + n2, tim + n2, tre + n2, tim + n2,
               yre + sy, yim + sy, 4 * sy, tw_next + n2, inverse);
    sr_dif_f32(n4, tre + 3 * n4, tim + 3 * n4, tre + 3 * n4, tim + 3 * n4,
               yre + 3 * sy, yim + 3 * sy, 4 * sy, tw_next + n2, inverse);
}

static void sr_dif_f64(size_t n,
                       const double *xre, const double *xim,
                       double *tre, double *tim,
                       double *yre, double *yim, size_t sy,
                       const double *tw, int inverse) {
    if (n == 1) {
        yre[0] = xre[0];
        yim[0] = xim[0];
        return;
    }
    if (n == 2) {
        double r0 = xre[0], i0 = xim[0];
        double r1 = xre[1], i1 = xim[1];
        yre[0] = r0 + r1; yim[0] = i0 + i1;
        yre[sy] = r0 - r1; yim[sy] = i0 - i1;
        return;
    }
    if (n == 4) {
        double re[4], im[4];
        re[0] = xre[0]; im[0] = xim[0];
        re[1] = xre[1]; im[1] = xim[1];
        re[2] = xre[2]; im[2] = xim[2];
        re[3] = xre[3]; im[3] = xim[3];
        faf_fft_kernel_4_f64(re, im, inverse);
        yre[0] = re[0];           yim[0] = im[0];
        yre[sy] = re[1];          yim[sy] = im[1];
        yre[2 * sy] = re[2];      yim[2 * sy] = im[2];
        yre[3 * sy] = re[3];      yim[3 * sy] = im[3];
        return;
    }
    if (n == 8) {
        double re[8], im[8];
        for (size_t i = 0; i < 8; i++) {
            re[i] = xre[i];
            im[i] = xim[i];
        }
        faf_fft_kernel_8_f64(re, im, inverse);
        for (size_t i = 0; i < 8; i++) {
            yre[i * sy] = re[i];
            yim[i * sy] = im[i];
        }
        return;
    }
#ifdef FAF_ARCH_AARCH64
    if (n == 16) {
        double t_re[16], t_im[16];
        sr_stage_neon_f64(16, 4, 8, xre, xim, t_re, t_im, tw, inverse);
        {
            double re[8], im[8];
            memcpy(re, t_re, 8 * sizeof(double));
            memcpy(im, t_im, 8 * sizeof(double));
            faf_fft_kernel_8_f64(re, im, inverse);
            for (size_t i = 0; i < 8; i++) {
                yre[i * 2 * sy] = re[i];
                yim[i * 2 * sy] = im[i];
            }
        }
        {
            double re[4], im[4];
            memcpy(re, t_re + 8, 4 * sizeof(double));
            memcpy(im, t_im + 8, 4 * sizeof(double));
            faf_fft_kernel_4_f64(re, im, inverse);
            yre[sy] = re[0];            yim[sy] = im[0];
            yre[sy + 4 * sy] = re[1];   yim[sy + 4 * sy] = im[1];
            yre[sy + 8 * sy] = re[2];   yim[sy + 8 * sy] = im[2];
            yre[sy + 12 * sy] = re[3];  yim[sy + 12 * sy] = im[3];
        }
        {
            double re[4], im[4];
            memcpy(re, t_re + 12, 4 * sizeof(double));
            memcpy(im, t_im + 12, 4 * sizeof(double));
            faf_fft_kernel_4_f64(re, im, inverse);
            yre[3 * sy] = re[0];            yim[3 * sy] = im[0];
            yre[3 * sy + 4 * sy] = re[1];   yim[3 * sy + 4 * sy] = im[1];
            yre[3 * sy + 8 * sy] = re[2];   yim[3 * sy + 8 * sy] = im[2];
            yre[3 * sy + 12 * sy] = re[3];  yim[3 * sy + 12 * sy] = im[3];
        }
        return;
    }
#endif

    size_t n4 = n / 4;
    size_t n2 = n / 2;
#ifdef FAF_ARCH_AARCH64
    sr_stage_neon_f64(n, n4, n2, xre, xim, tre, tim, tw, inverse);
#else
    for (size_t k = 0; k < n4; k++) {
        double x0r = xre[k],           x0i = xim[k];
        double x1r = xre[k + n2],      x1i = xim[k + n2];
        double x2r = xre[k + n4],      x2i = xim[k + n4];
        double x3r = xre[k + 3 * n4],  x3i = xim[k + 3 * n4];

        double t0r = x0r + x1r, t0i = x0i + x1i;
        double t1r = x2r + x3r, t1i = x2i + x3i;
        double t2r = x0r - x1r, t2i = x0i - x1i;
        double diffr = x2r - x3r, diffi = x2i - x3i;

        double t3r, t3i;
        if (!inverse) {
            t3r = -diffi;
            t3i =  diffr;
        } else {
            t3r =  diffi;
            t3i = -diffr;
        }

        tre[k] = t0r;
        tim[k] = t0i;
        tre[k + n4] = t1r;
        tim[k + n4] = t1i;

        double w1r = tw[4 * k],     w1i = tw[4 * k + 1];
        double w3r = tw[4 * k + 2], w3i = tw[4 * k + 3];

        double u0r = t2r - t3r, u0i = t2i - t3i;
        tre[k + n2] = u0r * w1r - u0i * w1i;
        tim[k + n2] = u0r * w1i + u0i * w1r;

        double u1r = t2r + t3r, u1i = t2i + t3i;
        tre[k + 3 * n4] = u1r * w3r - u1i * w3i;
        tim[k + 3 * n4] = u1r * w3i + u1i * w3r;
    }
#endif

    const double *tw_next = tw + n;
    sr_dif_f64(n2, tre, tim, tre, tim,
               yre, yim, 2 * sy, tw_next, inverse);
    sr_dif_f64(n4, tre + n2, tim + n2, tre + n2, tim + n2,
               yre + sy, yim + sy, 4 * sy, tw_next + n2, inverse);
    sr_dif_f64(n4, tre + 3 * n4, tim + 3 * n4, tre + 3 * n4, tim + 3 * n4,
               yre + 3 * sy, yim + 3 * sy, 4 * sy, tw_next + n2, inverse);
}

int faf_fft_sr_dif_execute(const faf_transform *t,
                           void *out_re, void *out_im,
                           const void *in_re, const void *in_im) {
    if (!t || !t->twiddles[0] || !t->twiddles[1]) return -1;
    int inverse = (t->flags & FAF_FLAG_INVERSE) != 0;
    size_t n = t->n;

    if (t->precision == FAF_PREC_FP64) {
        sr_dif_f64(n,
                   (const double *)in_re, (const double *)in_im,
                   (double *)t->twiddles[1],
                   (double *)t->twiddles[1] + n,
                   (double *)out_re, (double *)out_im, 1,
                   (const double *)t->twiddles[0], inverse);
        return 0;
    }

    sr_dif_f32(n,
               (const float *)in_re, (const float *)in_im,
               (float *)t->twiddles[1],
               (float *)t->twiddles[1] + n,
               (float *)out_re, (float *)out_im, 1,
               (const float *)t->twiddles[0], inverse);
    return 0;
}

static void bitrev_split_f32(float *re, float *im, size_t n) {
    size_t bits = 0;
    for (size_t tmp = n; tmp > 1; tmp >>= 1) bits++;
    for (size_t i = 0; i < n; i++) {
        size_t j = 0, x = i;
        for (size_t k = 0; k < bits; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            float tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr;    im[j] = ti;
        }
    }
}

static void bitrev_split_f64(double *re, double *im, size_t n) {
    size_t bits = 0;
    for (size_t tmp = n; tmp > 1; tmp >>= 1) bits++;
    for (size_t i = 0; i < n; i++) {
        size_t j = 0, x = i;
        for (size_t k = 0; k < bits; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            double tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr;    im[j] = ti;
        }
    }
}

int faf_fft_dit_execute(const faf_transform *t,
                        void *out_re, void *out_im,
                        const void *in_re, const void *in_im) {
    if (!t || !t->twiddles[2]) return -1;
    int inverse = (t->flags & FAF_FLAG_INVERSE) != 0;
    size_t n = t->n;

    if (t->precision == FAF_PREC_FP64) {
        double *ore = (double *)out_re;
        double *oim = (double *)out_im;
        if (ore != in_re) memcpy(ore, in_re, n * sizeof(double));
        if (oim != in_im) memcpy(oim, in_im, n * sizeof(double));
        bitrev_split_f64(ore, oim, n);
        const double *tw = (const double *)t->twiddles[2];
        for (size_t group = 2; group <= n; group <<= 1) {
            size_t tw_step = n / group;
            faf_fft_stage_split_f64(ore, oim, n, (uint32_t)group, 1u,
                                    (uint32_t)tw_step, tw, n, inverse);
        }
        return 0;
    }

    float *ore = (float *)out_re;
    float *oim = (float *)out_im;
    if (ore != in_re) memcpy(ore, in_re, n * sizeof(float));
    if (oim != in_im) memcpy(oim, in_im, n * sizeof(float));
    bitrev_split_f32(ore, oim, n);
    const float *tw = (const float *)t->twiddles[2];
    for (size_t group = 2; group <= n; group <<= 1) {
        size_t tw_step = n / group;
        faf_fft_stage_split_f32(ore, oim, n, (uint32_t)group, 1u,
                                (uint32_t)tw_step, tw, n, inverse);
    }
    return 0;
}
