/**
 * @file faf_cwt_kernels.c
 * @brief CWT Hermitian multiply / accumulate kernels with SIMD dispatch
 */

#include "faf_cwt.h"

#ifdef FAF_ARCH_X86_64
void faf_cwt_mul_hermitian_f32_avx2(float *, float *, const float *, const float *, const float *, size_t);
void faf_cwt_mul_hermitian_f64_avx2(double *, double *, const double *, const double *, const double *, size_t);
void faf_cwt_mulacc_hermitian_f32_avx2(float *, float *, const float *, const float *, const float *, size_t);
void faf_cwt_mulacc_hermitian_f64_avx2(double *, double *, const double *, const double *, const double *, size_t);
#endif

#ifdef FAF_ARCH_AARCH64
void faf_cwt_mul_hermitian_f32_neon(float *, float *, const float *, const float *, const float *, size_t);
void faf_cwt_mul_hermitian_f64_neon(double *, double *, const double *, const double *, const double *, size_t);
void faf_cwt_mulacc_hermitian_f32_neon(float *, float *, const float *, const float *, const float *, size_t);
void faf_cwt_mulacc_hermitian_f64_neon(double *, double *, const double *, const double *, const double *, size_t);
#endif

/* ---- scalar fallback ---- */

static void cwt_mul_hermitian_f32_scalar(float *FAF_RESTRICT y_re, float *FAF_RESTRICT y_im,
                                         const float *FAF_RESTRICT x_re, const float *FAF_RESTRICT x_im,
                                         const float *FAF_RESTRICT psi, size_t n_bins)
{
    for (size_t k = 0; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0f;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0f;
}

static void cwt_mul_hermitian_f64_scalar(double *FAF_RESTRICT y_re, double *FAF_RESTRICT y_im,
                                         const double *FAF_RESTRICT x_re, const double *FAF_RESTRICT x_im,
                                         const double *FAF_RESTRICT psi, size_t n_bins)
{
    for (size_t k = 0; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0;
}

static void cwt_mulacc_hermitian_f32_scalar(float *FAF_RESTRICT acc_re, float *FAF_RESTRICT acc_im,
                                            const float *FAF_RESTRICT y_re, const float *FAF_RESTRICT y_im,
                                            const float *FAF_RESTRICT psi, size_t n_bins)
{
    for (size_t k = 0; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

static void cwt_mulacc_hermitian_f64_scalar(double *FAF_RESTRICT acc_re, double *FAF_RESTRICT acc_im,
                                            const double *FAF_RESTRICT y_re, const double *FAF_RESTRICT y_im,
                                            const double *FAF_RESTRICT psi, size_t n_bins)
{
    for (size_t k = 0; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

/* ---- dispatch ---- */

void faf_cwt_mul_hermitian_f32(float *FAF_RESTRICT y_re, float *FAF_RESTRICT y_im,
                               const float *FAF_RESTRICT x_re, const float *FAF_RESTRICT x_im,
                               const float *FAF_RESTRICT psi, size_t n_bins)
{
#if defined(FAF_ARCH_X86_64) && defined(FAF_HAVE_AVX2)
    faf_cwt_mul_hermitian_f32_avx2(y_re, y_im, x_re, x_im, psi, n_bins);
#elif defined(FAF_ARCH_AARCH64)
    faf_cwt_mul_hermitian_f32_neon(y_re, y_im, x_re, x_im, psi, n_bins);
#else
    cwt_mul_hermitian_f32_scalar(y_re, y_im, x_re, x_im, psi, n_bins);
#endif
}

void faf_cwt_mul_hermitian_f64(double *FAF_RESTRICT y_re, double *FAF_RESTRICT y_im,
                               const double *FAF_RESTRICT x_re, const double *FAF_RESTRICT x_im,
                               const double *FAF_RESTRICT psi, size_t n_bins)
{
#if defined(FAF_ARCH_X86_64) && defined(FAF_HAVE_AVX2)
    faf_cwt_mul_hermitian_f64_avx2(y_re, y_im, x_re, x_im, psi, n_bins);
#elif defined(FAF_ARCH_AARCH64)
    faf_cwt_mul_hermitian_f64_neon(y_re, y_im, x_re, x_im, psi, n_bins);
#else
    cwt_mul_hermitian_f64_scalar(y_re, y_im, x_re, x_im, psi, n_bins);
#endif
}

void faf_cwt_mulacc_hermitian_f32(float *FAF_RESTRICT acc_re, float *FAF_RESTRICT acc_im,
                                  const float *FAF_RESTRICT y_re, const float *FAF_RESTRICT y_im,
                                  const float *FAF_RESTRICT psi, size_t n_bins)
{
#if defined(FAF_ARCH_X86_64) && defined(FAF_HAVE_AVX2)
    faf_cwt_mulacc_hermitian_f32_avx2(acc_re, acc_im, y_re, y_im, psi, n_bins);
#elif defined(FAF_ARCH_AARCH64)
    faf_cwt_mulacc_hermitian_f32_neon(acc_re, acc_im, y_re, y_im, psi, n_bins);
#else
    cwt_mulacc_hermitian_f32_scalar(acc_re, acc_im, y_re, y_im, psi, n_bins);
#endif
}

void faf_cwt_mulacc_hermitian_f64(double *FAF_RESTRICT acc_re, double *FAF_RESTRICT acc_im,
                                  const double *FAF_RESTRICT y_re, const double *FAF_RESTRICT y_im,
                                  const double *FAF_RESTRICT psi, size_t n_bins)
{
#if defined(FAF_ARCH_X86_64) && defined(FAF_HAVE_AVX2)
    faf_cwt_mulacc_hermitian_f64_avx2(acc_re, acc_im, y_re, y_im, psi, n_bins);
#elif defined(FAF_ARCH_AARCH64)
    faf_cwt_mulacc_hermitian_f64_neon(acc_re, acc_im, y_re, y_im, psi, n_bins);
#else
    cwt_mulacc_hermitian_f64_scalar(acc_re, acc_im, y_re, y_im, psi, n_bins);
#endif
}
