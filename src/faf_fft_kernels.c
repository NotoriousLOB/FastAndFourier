/**
 * @file faf_fft_kernels.c
 * @brief Fully unrolled split-plane FFT codelets for N = 2, 4, 8
 *
 * Natural-order in, natural-order out. Norms are applied outside, same
 * as the VM path. Inverse is a sign flip on the W_4 / W_8 twiddles.
 */

#include "faf.h"
#include <string.h>
#include <math.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

/* N=2 DFT: y[0] = x[0]+x[1], y[1] = x[0]-x[1]. Same fwd/inv. */

void faf_fft_kernel_2_f32(float *re, float *im, int inverse) {
    (void)inverse;
    float t0r = re[0] + re[1], t1r = re[0] - re[1];
    float t0i = im[0] + im[1], t1i = im[0] - im[1];
    re[0] = t0r; re[1] = t1r;
    im[0] = t0i; im[1] = t1i;
}

void faf_fft_kernel_2_f64(double *re, double *im, int inverse) {
    (void)inverse;
    double t0r = re[0] + re[1], t1r = re[0] - re[1];
    double t0i = im[0] + im[1], t1i = im[0] - im[1];
    re[0] = t0r; re[1] = t1r;
    im[0] = t0i; im[1] = t1i;
}

/* N=4 DFT, radix-4.  W_4^1 = -i (fwd) or +i (inv). */

void faf_fft_kernel_4_f32(float *re, float *im, int inverse) {
    float a0r = re[0], a0i = im[0];
    float a1r = re[1], a1i = im[1];
    float a2r = re[2], a2i = im[2];
    float a3r = re[3], a3i = im[3];

    float t0r = a0r + a2r, t0i = a0i + a2i;
    float t1r = a0r - a2r, t1i = a0i - a2i;
    float t2r = a1r + a3r, t2i = a1i + a3i;
    float t3r = a1r - a3r, t3i = a1i - a3i;

    re[0] = t0r + t2r; im[0] = t0i + t2i;
    re[2] = t0r - t2r; im[2] = t0i - t2i;
    if (!inverse) {
        re[1] = t1r + t3i; im[1] = t1i - t3r;
        re[3] = t1r - t3i; im[3] = t1i + t3r;
    } else {
        re[1] = t1r - t3i; im[1] = t1i + t3r;
        re[3] = t1r + t3i; im[3] = t1i - t3r;
    }
}

void faf_fft_kernel_4_f64(double *re, double *im, int inverse) {
    double a0r = re[0], a0i = im[0];
    double a1r = re[1], a1i = im[1];
    double a2r = re[2], a2i = im[2];
    double a3r = re[3], a3i = im[3];

    double t0r = a0r + a2r, t0i = a0i + a2i;
    double t1r = a0r - a2r, t1i = a0i - a2i;
    double t2r = a1r + a3r, t2i = a1i + a3i;
    double t3r = a1r - a3r, t3i = a1i - a3i;

    re[0] = t0r + t2r; im[0] = t0i + t2i;
    re[2] = t0r - t2r; im[2] = t0i - t2i;
    if (!inverse) {
        re[1] = t1r + t3i; im[1] = t1i - t3r;
        re[3] = t1r - t3i; im[3] = t1i + t3r;
    } else {
        re[1] = t1r - t3i; im[1] = t1i + t3r;
        re[3] = t1r + t3i; im[3] = t1i - t3r;
    }
}

/*
 * N=8 DFT, fully unrolled Cooley-Tukey DIT.
 * Stage 1: 4 radix-2 on pairs (0,4),(1,5),(2,6),(3,7) — no twiddles.
 * Stage 2: 2 radix-2 on quads (0,2),(1,3) and (4,6),(5,7) with W_4 twiddles.
 * Stage 3: 1 radix-2 on octets with W_8 twiddles.
 *
 * Twiddle constants (forward):
 *   W_4^0 = (1, 0)          W_8^0 = (1,  0)
 *   W_4^1 = (0, -1)         W_8^1 = (√2/2, -√2/2)
 *                            W_8^2 = (0, -1)
 *                            W_8^3 = (-√2/2, -√2/2)
 * Inverse: negate the imaginary part of each twiddle.
 *
 * Input in natural order, bit-reversal baked into the indexing.
 */

void faf_fft_kernel_8_f32(float *re, float *im, int inverse) {
    const float s = inverse ? -1.0f : 1.0f;
    const float c1 = (float)M_SQRT1_2;

    /* Bit-reversal permutation: {0,4,2,6,1,5,3,7} */
    float r0 = re[0], i0 = im[0];
    float r1 = re[4], i1 = im[4];
    float r2 = re[2], i2 = im[2];
    float r3 = re[6], i3 = im[6];
    float r4 = re[1], i4 = im[1];
    float r5 = re[5], i5 = im[5];
    float r6 = re[3], i6 = im[3];
    float r7 = re[7], i7 = im[7];

    /* Stage 1: radix-2 butterflies, no twiddles */
    { float tr = r1, ti = i1; r1 = r0 - tr; i1 = i0 - ti; r0 += tr; i0 += ti; }
    { float tr = r3, ti = i3; r3 = r2 - tr; i3 = i2 - ti; r2 += tr; i2 += ti; }
    { float tr = r5, ti = i5; r5 = r4 - tr; i5 = i4 - ti; r4 += tr; i4 += ti; }
    { float tr = r7, ti = i7; r7 = r6 - tr; i7 = i6 - ti; r6 += tr; i6 += ti; }

    /* Stage 2: radix-2 with W_4 twiddles */
    /* Pair (r0,r2): W_4^0 = (1,0) */
    { float tr = r2, ti = i2; r2 = r0 - tr; i2 = i0 - ti; r0 += tr; i0 += ti; }
    /* Pair (r1,r3): W_4^1 = (0, -s) → twiddle(r3,i3) = (s*i3, -s*r3) */
    { float tr = s * i3, ti = -s * r3; r3 = r1 - tr; i3 = i1 - ti; r1 += tr; i1 += ti; }
    /* Pair (r4,r6): W_4^0 = (1,0) */
    { float tr = r6, ti = i6; r6 = r4 - tr; i6 = i4 - ti; r4 += tr; i4 += ti; }
    /* Pair (r5,r7): W_4^1 = (0, -s) */
    { float tr = s * i7, ti = -s * r7; r7 = r5 - tr; i7 = i5 - ti; r5 += tr; i5 += ti; }

    /* Stage 3: radix-2 with W_8 twiddles */
    /* k=0: W_8^0 = (1,0) on r4 */
    { float tr = r4, ti = i4; r4 = r0 - tr; i4 = i0 - ti; r0 += tr; i0 += ti; }
    /* k=1: W_8^1 = (c1, -s*c1) on r5 */
    { float tr = c1*r5 + s*c1*i5, ti = c1*i5 - s*c1*r5;
      r5 = r1 - tr; i5 = i1 - ti; r1 += tr; i1 += ti; }
    /* k=2: W_8^2 = (0, -s) on r6 */
    { float tr = s*i6, ti = -s*r6;
      r6 = r2 - tr; i6 = i2 - ti; r2 += tr; i2 += ti; }
    /* k=3: W_8^3 = (-c1, -s*c1) on r7 */
    { float tr = -c1*r7 + s*c1*i7, ti = -c1*i7 - s*c1*r7;
      r7 = r3 - tr; i7 = i3 - ti; r3 += tr; i3 += ti; }

    re[0] = r0; im[0] = i0;
    re[1] = r1; im[1] = i1;
    re[2] = r2; im[2] = i2;
    re[3] = r3; im[3] = i3;
    re[4] = r4; im[4] = i4;
    re[5] = r5; im[5] = i5;
    re[6] = r6; im[6] = i6;
    re[7] = r7; im[7] = i7;
}

void faf_fft_kernel_8_f64(double *re, double *im, int inverse) {
    const double s = inverse ? -1.0 : 1.0;
    const double c1 = M_SQRT1_2;

    double r0 = re[0], i0 = im[0];
    double r1 = re[4], i1 = im[4];
    double r2 = re[2], i2 = im[2];
    double r3 = re[6], i3 = im[6];
    double r4 = re[1], i4 = im[1];
    double r5 = re[5], i5 = im[5];
    double r6 = re[3], i6 = im[3];
    double r7 = re[7], i7 = im[7];

    { double tr = r1, ti = i1; r1 = r0 - tr; i1 = i0 - ti; r0 += tr; i0 += ti; }
    { double tr = r3, ti = i3; r3 = r2 - tr; i3 = i2 - ti; r2 += tr; i2 += ti; }
    { double tr = r5, ti = i5; r5 = r4 - tr; i5 = i4 - ti; r4 += tr; i4 += ti; }
    { double tr = r7, ti = i7; r7 = r6 - tr; i7 = i6 - ti; r6 += tr; i6 += ti; }

    { double tr = r2, ti = i2; r2 = r0 - tr; i2 = i0 - ti; r0 += tr; i0 += ti; }
    { double tr = s * i3, ti = -s * r3; r3 = r1 - tr; i3 = i1 - ti; r1 += tr; i1 += ti; }
    { double tr = r6, ti = i6; r6 = r4 - tr; i6 = i4 - ti; r4 += tr; i4 += ti; }
    { double tr = s * i7, ti = -s * r7; r7 = r5 - tr; i7 = i5 - ti; r5 += tr; i5 += ti; }

    { double tr = r4, ti = i4; r4 = r0 - tr; i4 = i0 - ti; r0 += tr; i0 += ti; }
    { double tr = c1*r5 + s*c1*i5, ti = c1*i5 - s*c1*r5;
      r5 = r1 - tr; i5 = i1 - ti; r1 += tr; i1 += ti; }
    { double tr = s*i6, ti = -s*r6;
      r6 = r2 - tr; i6 = i2 - ti; r2 += tr; i2 += ti; }
    { double tr = -c1*r7 + s*c1*i7, ti = -c1*i7 - s*c1*r7;
      r7 = r3 - tr; i7 = i3 - ti; r3 += tr; i3 += ti; }

    re[0] = r0; im[0] = i0;
    re[1] = r1; im[1] = i1;
    re[2] = r2; im[2] = i2;
    re[3] = r3; im[3] = i3;
    re[4] = r4; im[4] = i4;
    re[5] = r5; im[5] = i5;
    re[6] = r6; im[6] = i6;
    re[7] = r7; im[7] = i7;
}

/*
 * Direct DFT codelets for the other FFTW-style leaves (3,5,6,7,9,10,12).
 * Twiddles are a length-n table; the n² FMAs beat the VM for these sizes
 * and are the inner FFTs Rader actually runs (10,12,18-via-9, …).
 */
int faf_fft_is_codelet_size(size_t n) {
    switch (n) {
        case 2: case 3: case 4: case 5: case 6: case 7:
        case 8: case 9: case 10: case 12:
            return 1;
        default:
            return 0;
    }
}

static void dft_direct_f32(float *re, float *im, size_t n, int inverse) {
    float xr[12], xi[12], wr[12], wi[12];
    double sign = inverse ? 1.0 : -1.0;
    for (size_t t = 0; t < n; t++) {
        xr[t] = re[t];
        xi[t] = im[t];
        double ang = sign * 2.0 * 3.14159265358979323846 * (double)t / (double)n;
        wr[t] = (float)cos(ang);
        wi[t] = (float)sin(ang);
    }
    for (size_t k = 0; k < n; k++) {
        float sr = 0.0f, si = 0.0f;
        for (size_t t = 0; t < n; t++) {
            size_t e = (k * t) % n;
            sr += xr[t] * wr[e] - xi[t] * wi[e];
            si += xr[t] * wi[e] + xi[t] * wr[e];
        }
        re[k] = sr;
        im[k] = si;
    }
}

static void dft_direct_f64(double *re, double *im, size_t n, int inverse) {
    double xr[12], xi[12], wr[12], wi[12];
    double sign = inverse ? 1.0 : -1.0;
    for (size_t t = 0; t < n; t++) {
        xr[t] = re[t];
        xi[t] = im[t];
        double ang = sign * 2.0 * 3.14159265358979323846 * (double)t / (double)n;
        wr[t] = cos(ang);
        wi[t] = sin(ang);
    }
    for (size_t k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (size_t t = 0; t < n; t++) {
            size_t e = (k * t) % n;
            sr += xr[t] * wr[e] - xi[t] * wi[e];
            si += xr[t] * wi[e] + xi[t] * wr[e];
        }
        re[k] = sr;
        im[k] = si;
    }
}

void faf_fft_kernel_3_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 3, inverse);
}
void faf_fft_kernel_5_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 5, inverse);
}
void faf_fft_kernel_6_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 6, inverse);
}
void faf_fft_kernel_7_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 7, inverse);
}
void faf_fft_kernel_9_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 9, inverse);
}
void faf_fft_kernel_10_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 10, inverse);
}
void faf_fft_kernel_12_f32(float *re, float *im, int inverse) {
    dft_direct_f32(re, im, 12, inverse);
}

void faf_fft_kernel_3_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 3, inverse);
}
void faf_fft_kernel_5_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 5, inverse);
}
void faf_fft_kernel_6_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 6, inverse);
}
void faf_fft_kernel_7_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 7, inverse);
}
void faf_fft_kernel_9_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 9, inverse);
}
void faf_fft_kernel_10_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 10, inverse);
}
void faf_fft_kernel_12_f64(double *re, double *im, int inverse) {
    dft_direct_f64(re, im, 12, inverse);
}

/* Dispatcher matching execute_func signature */

int faf_fft_kernel_execute(const faf_transform *t,
                           void *out_re, void *out_im,
                           const void *in_re, const void *in_im) {
    const size_t n = t->n;
    int inverse = (t->flags & FAF_FLAG_INVERSE) != 0;

    if (t->precision == FAF_PREC_FP64) {
        double *ore = (double *)out_re;
        double *oim = (double *)out_im;
        if (ore != in_re) memcpy(ore, in_re, n * sizeof(double));
        if (oim != in_im) memcpy(oim, in_im, n * sizeof(double));
        switch (n) {
            case 2: faf_fft_kernel_2_f64(ore, oim, inverse); return 0;
            case 3: faf_fft_kernel_3_f64(ore, oim, inverse); return 0;
            case 4: faf_fft_kernel_4_f64(ore, oim, inverse); return 0;
            case 5: faf_fft_kernel_5_f64(ore, oim, inverse); return 0;
            case 6: faf_fft_kernel_6_f64(ore, oim, inverse); return 0;
            case 7: faf_fft_kernel_7_f64(ore, oim, inverse); return 0;
            case 8: faf_fft_kernel_8_f64(ore, oim, inverse); return 0;
            case 9: faf_fft_kernel_9_f64(ore, oim, inverse); return 0;
            case 10: faf_fft_kernel_10_f64(ore, oim, inverse); return 0;
            case 12: faf_fft_kernel_12_f64(ore, oim, inverse); return 0;
            default: return -1;
        }
    }

    float *ore = (float *)out_re;
    float *oim = (float *)out_im;
    if (ore != in_re) memcpy(ore, in_re, n * sizeof(float));
    if (oim != in_im) memcpy(oim, in_im, n * sizeof(float));
    switch (n) {
        case 2: faf_fft_kernel_2_f32(ore, oim, inverse); return 0;
        case 3: faf_fft_kernel_3_f32(ore, oim, inverse); return 0;
        case 4: faf_fft_kernel_4_f32(ore, oim, inverse); return 0;
        case 5: faf_fft_kernel_5_f32(ore, oim, inverse); return 0;
        case 6: faf_fft_kernel_6_f32(ore, oim, inverse); return 0;
        case 7: faf_fft_kernel_7_f32(ore, oim, inverse); return 0;
        case 8: faf_fft_kernel_8_f32(ore, oim, inverse); return 0;
        case 9: faf_fft_kernel_9_f32(ore, oim, inverse); return 0;
        case 10: faf_fft_kernel_10_f32(ore, oim, inverse); return 0;
        case 12: faf_fft_kernel_12_f32(ore, oim, inverse); return 0;
        default: return -1;
    }
}
