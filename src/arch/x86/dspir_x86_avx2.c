/**
 * @file dspir_x86_avx2.c
 * @brief AVX2 vectorized kernels for x86_64
 * 
 * AVX2 provides 256-bit registers (8 floats) and FMA support
 */

#include "dspir.h"

#ifdef DSPIR_ARCH_X86_64

#include <immintrin.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* AVX2 register width: 8 floats */
#define AVX2_WIDTH 8

/**
 * @brief Complex multiplication using AVX2 with FMA
 */
static inline __m256 avx2_complex_mul_re(__m256 a_re, __m256 a_im,
                                          __m256 b_re, __m256 b_im) {
    /* (a+ib)*(c+id) real part = ac - bd */
    /* Using FMA: fmadd(a, c, -b*d) */
    return _mm256_fmsub_ps(a_re, b_re, _mm256_mul_ps(a_im, b_im));
}

static inline __m256 avx2_complex_mul_im(__m256 a_re, __m256 a_im,
                                          __m256 b_re, __m256 b_im) {
    /* (a+ib)*(c+id) imag part = ad + bc */
    /* Using FMA: fmadd(a, d, b*c) */
    return _mm256_fmadd_ps(a_re, b_im, _mm256_mul_ps(a_im, b_re));
}

/**
 * @brief Radix-2 butterfly using AVX2
 */
static inline void avx2_bfly2(__m256 *a_re, __m256 *a_im,
                               __m256 *b_re, __m256 *b_im,
                               __m256 w_re, __m256 w_im) {
    /* twiddle b */
    __m256 t_re = avx2_complex_mul_re(*b_re, *b_im, w_re, w_im);
    __m256 t_im = avx2_complex_mul_im(*b_re, *b_im, w_re, w_im);
    
    /* butterfly: a' = a + t, b' = a - t */
    __m256 new_a_re = _mm256_add_ps(*a_re, t_re);
    __m256 new_a_im = _mm256_add_ps(*a_im, t_im);
    __m256 new_b_re = _mm256_sub_ps(*a_re, t_re);
    __m256 new_b_im = _mm256_sub_ps(*a_im, t_im);
    
    *a_re = new_a_re;
    *a_im = new_a_im;
    *b_re = new_b_re;
    *b_im = new_b_im;
}

/**
 * @brief Radix-4 butterfly using AVX2
 * 
 * More efficient: 4 outputs from 4 inputs with fewer operations
 */
static inline void avx2_bfly4(__m256 *r0, __m256 *i0,
                               __m256 *r1, __m256 *i1,
                               __m256 *r2, __m256 *i2,
                               __m256 *r3, __m256 *i3,
                               __m256 w1_re, __m256 w1_im,
                               __m256 w2_re, __m256 w2_im,
                               __m256 w3_re, __m256 w3_im) {
    /* Stage 1: Two radix-2 butterflies */
    avx2_bfly2(r0, i0, r2, i2, w2_re, w2_im);
    avx2_bfly2(r1, i1, r3, i3, w2_re, w2_im);
    
    /* Stage 2: Final butterfly with j multiplication for r3/i3 */
    /* Multiply r3/i3 by j = (0, 1): (a+ib)*j = -b + ia */
    __m256 t3_re = _mm256_sub_ps(_mm256_setzero_ps(), *i3);  /* -b */
    __m256 t3_im = *r3;  /* a */
    
    __m256 new_r0 = _mm256_add_ps(*r0, *r1);
    __m256 new_i0 = _mm256_add_ps(*i0, *i1);
    __m256 new_r1 = _mm256_add_ps(*r0, t3_re);
    __m256 new_i1 = _mm256_add_ps(*i0, t3_im);
    __m256 new_r2 = _mm256_sub_ps(*r0, *r1);
    __m256 new_i2 = _mm256_sub_ps(*i0, *i1);
    __m256 new_r3 = _mm256_sub_ps(*r0, t3_re);
    __m256 new_i3 = _mm256_sub_ps(*i0, t3_im);
    
    *r0 = new_r0; *i0 = new_i0;
    *r1 = new_r1; *i1 = new_i1;
    *r2 = new_r2; *i2 = new_i2;
    *r3 = new_r3; *i3 = new_i3;
}

/**
 * @brief AVX2-optimized FFT execution
 */
void dspir_x86_avx2_execute_f32(const dspir_transform *t,
                                 float *restrict out,
                                 const float *restrict in) {
    const size_t n = t->n;
    const float *tw = (const float *)t->twiddles[0];
    
    /* Allocate aligned buffer */
    float *work = NULL;
    if (posix_memalign((void**)&work, 32, n * 2 * sizeof(float)) != 0) {
        work = malloc(n * 2 * sizeof(float));
    }
    
    /* Copy input to working buffer (interleaved complex) */
    for (size_t i = 0; i < n; i++) {
        work[2*i] = in[i];
        work[2*i + 1] = 0.0f;
    }
    
    /* Bit-reversal permutation */
    size_t bits = 0;
    size_t temp = n;
    while (temp > 1) { temp >>= 1; bits++; }
    
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        size_t x = i;
        for (size_t k = 0; k < bits; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            float tr = work[2*i], ti = work[2*i + 1];
            work[2*i] = work[2*j];
            work[2*i + 1] = work[2*j + 1];
            work[2*j] = tr;
            work[2*j + 1] = ti;
        }
    }
    
    /* FFT stages */
    for (size_t stage = 0; stage < bits; stage++) {
        size_t stride = 1 << stage;
        size_t twiddle_step = n / (2 * stride);
        
        for (size_t group = 0; group < n / (2 * stride); group++) {
            size_t base = group * 2 * stride;
            
            /* Process 8 butterflies at a time with AVX2 */
            size_t i = 0;
            for (; i + 7 < stride; i += 8) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                /* Load 8 complex values for each input */
                __m256 a_re = _mm256_loadu_ps(&work[2*idx0]);
                __m256 a_im = _mm256_loadu_ps(&work[2*idx0 + 8]);
                __m256 b_re = _mm256_loadu_ps(&work[2*idx1]);
                __m256 b_im = _mm256_loadu_ps(&work[2*idx1 + 8]);
                
                /* Load twiddles */
                size_t tw_idx = i * twiddle_step;
                __m256 w_re = _mm256_set_ps(
                    tw[2*(tw_idx+7*twiddle_step)], tw[2*(tw_idx+6*twiddle_step)],
                    tw[2*(tw_idx+5*twiddle_step)], tw[2*(tw_idx+4*twiddle_step)],
                    tw[2*(tw_idx+3*twiddle_step)], tw[2*(tw_idx+2*twiddle_step)],
                    tw[2*(tw_idx+twiddle_step)], tw[2*tw_idx]
                );
                __m256 w_im = _mm256_set_ps(
                    tw[2*(tw_idx+7*twiddle_step)+1], tw[2*(tw_idx+6*twiddle_step)+1],
                    tw[2*(tw_idx+5*twiddle_step)+1], tw[2*(tw_idx+4*twiddle_step)+1],
                    tw[2*(tw_idx+3*twiddle_step)+1], tw[2*(tw_idx+2*twiddle_step)+1],
                    tw[2*(tw_idx+twiddle_step)+1], tw[2*tw_idx+1]
                );
                
                /* Butterfly with FMA */
                avx2_bfly2(&a_re, &a_im, &b_re, &b_im, w_re, w_im);
                
                /* Store back */
                _mm256_storeu_ps(&work[2*idx0], a_re);
                _mm256_storeu_ps(&work[2*idx0 + 8], a_im);
                _mm256_storeu_ps(&work[2*idx1], b_re);
                _mm256_storeu_ps(&work[2*idx1 + 8], b_im);
            }
            
            /* SSE remainder (4 elements) */
            for (; i + 3 < stride; i += 4) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                __m128 a_re = _mm_loadu_ps(&work[2*idx0]);
                __m128 a_im = _mm_loadu_ps(&work[2*idx0 + 4]);
                __m128 b_re = _mm_loadu_ps(&work[2*idx1]);
                __m128 b_im = _mm_loadu_ps(&work[2*idx1 + 4]);
                
                size_t tw_idx = i * twiddle_step;
                __m128 w_re = _mm_set_ps(
                    tw[2*(tw_idx+3*twiddle_step)], tw[2*(tw_idx+2*twiddle_step)],
                    tw[2*(tw_idx+twiddle_step)], tw[2*tw_idx]
                );
                __m128 w_im = _mm_set_ps(
                    tw[2*(tw_idx+3*twiddle_step)+1], tw[2*(tw_idx+2*twiddle_step)+1],
                    tw[2*(tw_idx+twiddle_step)+1], tw[2*tw_idx+1]
                );
                
                /* Complex multiply b by twiddle */
                __m128 t_re = _mm_sub_ps(_mm_mul_ps(b_re, w_re), _mm_mul_ps(b_im, w_im));
                __m128 t_im = _mm_add_ps(_mm_mul_ps(b_re, w_im), _mm_mul_ps(b_im, w_re));
                
                __m128 new_a_re = _mm_add_ps(a_re, t_re);
                __m128 new_a_im = _mm_add_ps(a_im, t_im);
                __m128 new_b_re = _mm_sub_ps(a_re, t_re);
                __m128 new_b_im = _mm_sub_ps(a_im, t_im);
                
                _mm_storeu_ps(&work[2*idx0], new_a_re);
                _mm_storeu_ps(&work[2*idx0 + 4], new_a_im);
                _mm_storeu_ps(&work[2*idx1], new_b_re);
                _mm_storeu_ps(&work[2*idx1 + 4], new_b_im);
            }
            
            /* Scalar remainder */
            for (; i < stride; i++) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                float a_re = work[2*idx0], a_im = work[2*idx0+1];
                float b_re = work[2*idx1], b_im = work[2*idx1+1];
                
                size_t tw_idx = i * twiddle_step;
                float w_re = tw[2*tw_idx], w_im = tw[2*tw_idx+1];
                
                float t_re = b_re * w_re - b_im * w_im;
                float t_im = b_re * w_im + b_im * w_re;
                
                work[2*idx0] = a_re + t_re;
                work[2*idx0+1] = a_im + t_im;
                work[2*idx1] = a_re - t_re;
                work[2*idx1+1] = a_im - t_im;
            }
        }
    }
    
    /* Copy output */
    for (size_t i = 0; i < n; i++) {
        out[i] = work[2*i];
    }
    
    free(work);
}

/**
 * @brief AVX2-optimized polyphase FIR
 */
void dspir_x86_avx2_polyfir_f32(const dspir_transform *t,
                                 float *restrict out,
                                 const float *restrict in) {
    const size_t nch = 8;  /* 8 channels */
    const size_t taps = 32;  /* 32 taps */
    const float *coeffs = (const float *)t->twiddles[0];
    
    /* Process each channel */
    for (size_t ch = 0; ch < nch; ch++) {
        const float *x = &in[ch * taps];
        
        __m256 acc = _mm256_setzero_ps();
        
        /* Process 8 taps at a time with FMA */
        size_t k = 0;
        for (; k + 7 < taps; k += 8) {
            __m256 xv = _mm256_loadu_ps(&x[k]);
            __m256 cv = _mm256_loadu_ps(&coeffs[k]);
            acc = _mm256_fmadd_ps(xv, cv, acc);
        }
        
        /* Horizontal sum */
        __m128 low = _mm256_castps256_ps128(acc);
        __m128 high = _mm256_extractf128_ps(acc, 1);
        low = _mm_add_ps(low, high);
        low = _mm_hadd_ps(low, low);
        low = _mm_hadd_ps(low, low);
        
        /* Scalar remainder */
        float sum = _mm_cvtss_f32(low);
        for (; k < taps; k++) {
            sum += x[k] * coeffs[k];
        }
        
        out[ch] = sum;
    }
}

#endif /* DSPIR_ARCH_X86_64 */
