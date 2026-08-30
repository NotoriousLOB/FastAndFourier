/**
 * @file faf_cwt_neon.c
 * @brief NEON-vectorized CWT Hermitian multiply / accumulate kernels (AArch64)
 */

#include "faf.h"

#ifdef FAF_ARCH_AARCH64

#include <arm_neon.h>

void faf_cwt_mul_hermitian_f32_neon(float *FAF_RESTRICT y_re, float *FAF_RESTRICT y_im,
                                    const float *FAF_RESTRICT x_re, const float *FAF_RESTRICT x_im,
                                    const float *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n4 = (n_bins / 4) * 4;
    for (; k < n4; k += 4) {
        float32x4_t p = vld1q_f32(psi + k);
        float32x4_t xr = vld1q_f32(x_re + k);
        float32x4_t xi = vld1q_f32(x_im + k);
        vst1q_f32(y_re + k, vmulq_f32(xr, p));
        vst1q_f32(y_im + k, vmulq_f32(xi, p));
    }
    for (; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0f;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0f;
}

void faf_cwt_mul_hermitian_f64_neon(double *FAF_RESTRICT y_re, double *FAF_RESTRICT y_im,
                                    const double *FAF_RESTRICT x_re, const double *FAF_RESTRICT x_im,
                                    const double *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n2 = (n_bins / 2) * 2;
    for (; k < n2; k += 2) {
        float64x2_t p = vld1q_f64(psi + k);
        float64x2_t xr = vld1q_f64(x_re + k);
        float64x2_t xi = vld1q_f64(x_im + k);
        vst1q_f64(y_re + k, vmulq_f64(xr, p));
        vst1q_f64(y_im + k, vmulq_f64(xi, p));
    }
    for (; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0;
}

void faf_cwt_mulacc_hermitian_f32_neon(float *FAF_RESTRICT acc_re, float *FAF_RESTRICT acc_im,
                                       const float *FAF_RESTRICT y_re, const float *FAF_RESTRICT y_im,
                                       const float *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n4 = (n_bins / 4) * 4;
    for (; k < n4; k += 4) {
        float32x4_t p = vld1q_f32(psi + k);
        float32x4_t ar = vld1q_f32(acc_re + k);
        float32x4_t ai = vld1q_f32(acc_im + k);
        float32x4_t yr = vld1q_f32(y_re + k);
        float32x4_t yi = vld1q_f32(y_im + k);
        vst1q_f32(acc_re + k, vfmaq_f32(ar, yr, p));
        vst1q_f32(acc_im + k, vfmaq_f32(ai, yi, p));
    }
    for (; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

void faf_cwt_mulacc_hermitian_f64_neon(double *FAF_RESTRICT acc_re, double *FAF_RESTRICT acc_im,
                                       const double *FAF_RESTRICT y_re, const double *FAF_RESTRICT y_im,
                                       const double *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n2 = (n_bins / 2) * 2;
    for (; k < n2; k += 2) {
        float64x2_t p = vld1q_f64(psi + k);
        float64x2_t ar = vld1q_f64(acc_re + k);
        float64x2_t ai = vld1q_f64(acc_im + k);
        float64x2_t yr = vld1q_f64(y_re + k);
        float64x2_t yi = vld1q_f64(y_im + k);
        vst1q_f64(acc_re + k, vfmaq_f64(ar, yr, p));
        vst1q_f64(acc_im + k, vfmaq_f64(ai, yi, p));
    }
    for (; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

#endif /* FAF_ARCH_AARCH64 */
