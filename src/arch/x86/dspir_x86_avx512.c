/**
 * @file dspir_x86_avx512.c
 * @brief AVX-512 vectorized kernels for x86_64
 * 
 * AVX-512 provides 512-bit registers (16 floats), FMA, and mask registers
 */

#include "dspir.h"

#if defined(DSPIR_ARCH_X86_64) && defined(DSPIR_HAVE_AVX512)

#include <immintrin.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* AVX-512 register width: 16 floats */
#define AVX512_WIDTH 16

/**
 * @brief Complex multiplication using AVX-512 with FMA
 */
static inline __m512 avx512_complex_mul_re(__m512 a_re, __m512 a_im,
                                            __m512 b_re, __m512 b_im) {
    /* (a+ib)*(c+id) real part = ac - bd */
    return _mm512_fmsub_ps(a_re, b_re, _mm512_mul_ps(a_im, b_im));
}

static inline __m512 avx512_complex_mul_im(__m512 a_re, __m512 a_im,
                                            __m512 b_re, __m512 b_im) {
    /* (a+ib)*(c+id) imag part = ad + bc */
    return _mm512_fmadd_ps(a_re, b_im, _mm512_mul_ps(a_im, b_re));
}

/**
 * @brief Radix-2 butterfly using AVX-512
 */
static inline void avx512_bfly2(__m512 *a_re, __m512 *a_im,
                                 __m512 *b_re, __m512 *b_im,
                                 __m512 w_re, __m512 w_im) {
    /* twiddle b */
    __m512 t_re = avx512_complex_mul_re(*b_re, *b_im, w_re, w_im);
    __m512 t_im = avx512_complex_mul_im(*b_re, *b_im, w_re, w_im);
    
    /* butterfly */
    __m512 new_a_re = _mm512_add_ps(*a_re, t_re);
    __m512 new_a_im = _mm512_add_ps(*a_im, t_im);
    __m512 new_b_re = _mm512_sub_ps(*a_re, t_re);
    __m512 new_b_im = _mm512_sub_ps(*a_im, t_im);
    
    *a_re = new_a_re;
    *a_im = new_a_im;
    *b_re = new_b_re;
    *b_im = new_b_im;
}

/**
 * @brief AVX-512-optimized FFT execution
 * 
 * Uses 16-wide vectorization for maximum throughput
 */
void dspir_x86_avx512_execute_f32(const dspir_transform *t,
                                   float *restrict out,
                                   const float *restrict in) {
    const size_t n = t->n;
    const float *tw = (const float *)t->twiddles[0];
    
    /* Allocate aligned buffer */
    float *work = NULL;
    if (posix_memalign((void**)&work, 64, n * 2 * sizeof(float)) != 0) {
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
            
            /* Process 16 butterflies at a time with AVX-512 */
            size_t i = 0;
            for (; i + 15 < stride; i += 16) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                /* Load 16 complex values for each input */
                __m512 a_re = _mm512_loadu_ps(&work[2*idx0]);
                __m512 a_im = _mm512_loadu_ps(&work[2*idx0 + 16]);
                __m512 b_re = _mm512_loadu_ps(&work[2*idx1]);
                __m512 b_im = _mm512_loadu_ps(&work[2*idx1 + 16]);
                
                /* Load twiddles - broadcast each twiddle to all lanes */
                /* For AVX-512, we use gather or broadcast depending on pattern */
                size_t tw_idx = i * twiddle_step;
                __m512 w_re, w_im;
                
                if (twiddle_step == 1) {
                    /* Contiguous twiddles - use load */
                    w_re = _mm512_loadu_ps(&tw[2*tw_idx]);
                    w_im = _mm512_loadu_ps(&tw[2*tw_idx + 16]);
                } else {
                    /* Strided twiddles - use set */
                    float w_re_arr[16], w_im_arr[16];
                    for (int k = 0; k < 16; k++) {
                        w_re_arr[k] = tw[2*(tw_idx + k*twiddle_step)];
                        w_im_arr[k] = tw[2*(tw_idx + k*twiddle_step)+1];
                    }
                    w_re = _mm512_loadu_ps(w_re_arr);
                    w_im = _mm512_loadu_ps(w_im_arr);
                }
                
                /* Butterfly with FMA */
                avx512_bfly2(&a_re, &a_im, &b_re, &b_im, w_re, w_im);
                
                /* Store back */
                _mm512_storeu_ps(&work[2*idx0], a_re);
                _mm512_storeu_ps(&work[2*idx0 + 16], a_im);
                _mm512_storeu_ps(&work[2*idx1], b_re);
                _mm512_storeu_ps(&work[2*idx1 + 16], b_im);
            }
            
            /* AVX2 remainder (8 elements) */
            for (; i + 7 < stride; i += 8) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                __m256 a_re = _mm256_loadu_ps(&work[2*idx0]);
                __m256 a_im = _mm256_loadu_ps(&work[2*idx0 + 8]);
                __m256 b_re = _mm256_loadu_ps(&work[2*idx1]);
                __m256 b_im = _mm256_loadu_ps(&work[2*idx1 + 8]);
                
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
                
                __m256 t_re = _mm256_fmsub_ps(b_re, w_re, _mm256_mul_ps(b_im, w_im));
                __m256 t_im = _mm256_fmadd_ps(b_re, w_im, _mm256_mul_ps(b_im, w_re));
                
                __m256 new_a_re = _mm256_add_ps(a_re, t_re);
                __m256 new_a_im = _mm256_add_ps(a_im, t_im);
                __m256 new_b_re = _mm256_sub_ps(a_re, t_re);
                __m256 new_b_im = _mm256_sub_ps(a_im, t_im);
                
                _mm256_storeu_ps(&work[2*idx0], new_a_re);
                _mm256_storeu_ps(&work[2*idx0 + 8], new_a_im);
                _mm256_storeu_ps(&work[2*idx1], new_b_re);
                _mm256_storeu_ps(&work[2*idx1 + 8], new_b_im);
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
                
                __m128 t_re = _mm_sub_ps(_mm_mul_ps(b_re, w_re), _mm_mul_ps(b_im, w_im));
                __m128 t_im = _mm_add_ps(_mm_mul_ps(b_re, w_im), _mm_mul_ps(b_im, w_re));
                
                _mm_storeu_ps(&work[2*idx0], _mm_add_ps(a_re, t_re));
                _mm_storeu_ps(&work[2*idx0 + 4], _mm_add_ps(a_im, t_im));
                _mm_storeu_ps(&work[2*idx1], _mm_sub_ps(a_re, t_re));
                _mm_storeu_ps(&work[2*idx1 + 4], _mm_sub_ps(a_im, t_im));
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
 * @brief AVX-512-optimized polyphase FIR with 16-wide unrolling
 */
void dspir_x86_avx512_polyfir_f32(const dspir_transform *t,
                                   float *restrict out,
                                   const float *restrict in) {
    const size_t nch = 8;
    const size_t taps = 64;  /* 64 taps for AVX-512 */
    const float *coeffs = (const float *)t->twiddles[0];
    
    for (size_t ch = 0; ch < nch; ch++) {
        const float *x = &in[ch * taps];
        
        __m512 acc = _mm512_setzero_ps();
        
        /* Process 16 taps at a time with FMA */
        size_t k = 0;
        for (; k + 15 < taps; k += 16) {
            __m512 xv = _mm512_loadu_ps(&x[k]);
            __m512 cv = _mm512_loadu_ps(&coeffs[k]);
            acc = _mm512_fmadd_ps(xv, cv, acc);
        }
        
        /* Horizontal sum using reduce_add */
        float sum = _mm512_reduce_add_ps(acc);
        
        /* Scalar remainder */
        for (; k < taps; k++) {
            sum += x[k] * coeffs[k];
        }
        
        out[ch] = sum;
    }
}

/**
 * @brief AVX-512-optimized MDCT using 16-wide transforms
 */
void dspir_x86_avx512_mdct_f32(const dspir_transform *t,
                                float *restrict out,
                                const float *restrict in) {
    const size_t n = t->n;
    const size_t n2 = n / 2;
    const size_t n4 = n / 4;
    const float *tw = (const float *)t->twiddles[0];
    
    float *work = NULL;
    if (posix_memalign((void**)&work, 64, n * sizeof(float)) != 0) {
        work = malloc(n * sizeof(float));
    }
    
    /* Pre-rotation using AVX-512 */
    size_t i = 0;
    for (; i + 15 < n4; i += 16) {
        __m512 re = _mm512_loadu_ps(&in[i]);
        __m512 im = _mm512_loadu_ps(&in[n - 1 - i - 15]);
        im = _mm512_permutexvar_ps(_mm512_setr_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        ), im);
        
        __m512 w_re = _mm512_loadu_ps(&tw[2*i]);
        __m512 w_im = _mm512_loadu_ps(&tw[2*i + 16]);
        
        __m256 low_re = _mm512_castps512_ps256(re);
        __m256 high_re = _mm512_extractf32x8_ps(re, 1);
        __m256 low_im = _mm512_castps512_ps256(im);
        __m256 high_im = _mm512_extractf32x8_ps(im, 1);
        
        /* Complex multiply */
        __m256 t_re_low = _mm256_fmsub_ps(low_re, 
            _mm512_castps512_ps256(w_re), 
            _mm256_mul_ps(low_im, _mm512_castps512_ps256(w_im)));
        __m256 t_im_low = _mm256_fmadd_ps(low_re,
            _mm512_castps512_ps256(w_im),
            _mm256_mul_ps(low_im, _mm512_castps512_ps256(w_re)));
        
        _mm256_storeu_ps(&work[2*i], t_re_low);
        _mm256_storeu_ps(&work[2*i + 8], t_im_low);
    }
    
    /* Scalar remainder */
    for (; i < n4; i++) {
        float re = in[i];
        float im = in[n - 1 - i];
        float w_re = tw[2*i], w_im = tw[2*i + 1];
        work[2*i] = re * w_re - im * w_im;
        work[2*i + 1] = re * w_im + im * w_re;
    }
    
    /* n/2 point FFT would go here */
    /* For now, copy to output */
    memcpy(out, work, n2 * sizeof(float));
    
    free(work);
}

#endif /* DSPIR_ARCH_X86_64 && DSPIR_HAVE_AVX512 */
