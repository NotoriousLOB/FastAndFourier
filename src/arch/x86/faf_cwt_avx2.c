/**
 * @file faf_cwt_avx2.c
 * @brief AVX2-vectorized CWT Hermitian multiply / accumulate kernels (x86_64)
 */

#include "faf.h"

#ifdef FAF_ARCH_X86_64

#include <immintrin.h>

void faf_cwt_mul_hermitian_f32_avx2(float *FAF_RESTRICT y_re, float *FAF_RESTRICT y_im,
                                    const float *FAF_RESTRICT x_re, const float *FAF_RESTRICT x_im,
                                    const float *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n8 = (n_bins / 8) * 8;
    for (; k < n8; k += 8) {
        __m256 p = _mm256_load_ps(psi + k);
        __m256 xr = _mm256_load_ps(x_re + k);
        __m256 xi = _mm256_load_ps(x_im + k);
        _mm256_store_ps(y_re + k, _mm256_mul_ps(xr, p));
        _mm256_store_ps(y_im + k, _mm256_mul_ps(xi, p));
    }
    for (; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0f;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0f;
}

void faf_cwt_mul_hermitian_f64_avx2(double *FAF_RESTRICT y_re, double *FAF_RESTRICT y_im,
                                    const double *FAF_RESTRICT x_re, const double *FAF_RESTRICT x_im,
                                    const double *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n4 = (n_bins / 4) * 4;
    for (; k < n4; k += 4) {
        __m256d p = _mm256_load_pd(psi + k);
        __m256d xr = _mm256_load_pd(x_re + k);
        __m256d xi = _mm256_load_pd(x_im + k);
        _mm256_store_pd(y_re + k, _mm256_mul_pd(xr, p));
        _mm256_store_pd(y_im + k, _mm256_mul_pd(xi, p));
    }
    for (; k < n_bins; k++) {
        y_re[k] = x_re[k] * psi[k];
        y_im[k] = x_im[k] * psi[k];
    }
    y_im[0] = 0.0;
    if (n_bins > 1) y_im[n_bins - 1] = 0.0;
}

void faf_cwt_mulacc_hermitian_f32_avx2(float *FAF_RESTRICT acc_re, float *FAF_RESTRICT acc_im,
                                       const float *FAF_RESTRICT y_re, const float *FAF_RESTRICT y_im,
                                       const float *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n8 = (n_bins / 8) * 8;
    for (; k < n8; k += 8) {
        __m256 p = _mm256_load_ps(psi + k);
        __m256 ar = _mm256_load_ps(acc_re + k);
        __m256 ai = _mm256_load_ps(acc_im + k);
        __m256 yr = _mm256_load_ps(y_re + k);
        __m256 yi = _mm256_load_ps(y_im + k);
#ifdef __FMA__
        _mm256_store_ps(acc_re + k, _mm256_fmadd_ps(yr, p, ar));
        _mm256_store_ps(acc_im + k, _mm256_fmadd_ps(yi, p, ai));
#else
        _mm256_store_ps(acc_re + k, _mm256_add_ps(_mm256_mul_ps(yr, p), ar));
        _mm256_store_ps(acc_im + k, _mm256_add_ps(_mm256_mul_ps(yi, p), ai));
#endif
    }
    for (; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

void faf_cwt_mulacc_hermitian_f64_avx2(double *FAF_RESTRICT acc_re, double *FAF_RESTRICT acc_im,
                                       const double *FAF_RESTRICT y_re, const double *FAF_RESTRICT y_im,
                                       const double *FAF_RESTRICT psi, size_t n_bins)
{
    size_t k = 0;
    size_t n4 = (n_bins / 4) * 4;
    for (; k < n4; k += 4) {
        __m256d p = _mm256_load_pd(psi + k);
        __m256d ar = _mm256_load_pd(acc_re + k);
        __m256d ai = _mm256_load_pd(acc_im + k);
        __m256d yr = _mm256_load_pd(y_re + k);
        __m256d yi = _mm256_load_pd(y_im + k);
#ifdef __FMA__
        _mm256_store_pd(acc_re + k, _mm256_fmadd_pd(yr, p, ar));
        _mm256_store_pd(acc_im + k, _mm256_fmadd_pd(yi, p, ai));
#else
        _mm256_store_pd(acc_re + k, _mm256_add_pd(_mm256_mul_pd(yr, p), ar));
        _mm256_store_pd(acc_im + k, _mm256_add_pd(_mm256_mul_pd(yi, p), ai));
#endif
    }
    for (; k < n_bins; k++) {
        acc_re[k] += y_re[k] * psi[k];
        acc_im[k] += y_im[k] * psi[k];
    }
}

#endif /* FAF_ARCH_X86_64 */
