/**
 * @file dspir_x86_sse.c
 * @brief SSE4.2 vectorized kernels for x86_64
 */

#include "dspir.h"

#ifdef DSPIR_ARCH_X86_64

#include <immintrin.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* SSE register width: 4 floats */
#define SSE_WIDTH 4

/**
 * @brief Complex multiplication using SSE
 * 
 * (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
 */
static inline __m128 sse_complex_mul_re(__m128 a_re, __m128 a_im, 
                                         __m128 b_re, __m128 b_im) {
    /* ac - bd */
    return _mm_sub_ps(_mm_mul_ps(a_re, b_re), _mm_mul_ps(a_im, b_im));
}

static inline __m128 sse_complex_mul_im(__m128 a_re, __m128 a_im,
                                         __m128 b_re, __m128 b_im) {
    /* ad + bc */
    return _mm_add_ps(_mm_mul_ps(a_re, b_im), _mm_mul_ps(a_im, b_re));
}

/**
 * @brief Radix-2 butterfly using SSE
 */
static inline void sse_bfly2(__m128 *a_re, __m128 *a_im,
                              __m128 *b_re, __m128 *b_im,
                              __m128 w_re, __m128 w_im) {
    /* twiddle b */
    __m128 t_re = sse_complex_mul_re(*b_re, *b_im, w_re, w_im);
    __m128 t_im = sse_complex_mul_im(*b_re, *b_im, w_re, w_im);
    
    /* butterfly */
    __m128 new_a_re = _mm_add_ps(*a_re, t_re);
    __m128 new_a_im = _mm_add_ps(*a_im, t_im);
    __m128 new_b_re = _mm_sub_ps(*a_re, t_re);
    __m128 new_b_im = _mm_sub_ps(*a_im, t_im);
    
    *a_re = new_a_re;
    *a_im = new_a_im;
    *b_re = new_b_re;
    *b_im = new_b_im;
}

/**
 * @brief SSE-optimized FFT execution
 */
void dspir_x86_sse_execute_f32(const dspir_transform *t,
                                float *restrict out,
                                const float *restrict in) {
    const size_t n = t->n;
    const float *tw = (const float *)t->twiddles[0];
    
    /* Allocate aligned buffer */
    float *work = NULL;
    if (posix_memalign((void**)&work, 16, n * 2 * sizeof(float)) != 0) {
        /* Fall back to non-aligned */
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
            
            /* Process 4 butterflies at a time with SSE */
            size_t i = 0;
            for (; i + 3 < stride; i += 4) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                /* Load 4 complex values for each input */
                __m128 a_re = _mm_set_ps(work[2*(idx0+3)], work[2*(idx0+2)],
                                          work[2*(idx0+1)], work[2*idx0]);
                __m128 a_im = _mm_set_ps(work[2*(idx0+3)+1], work[2*(idx0+2)+1],
                                          work[2*(idx0+1)+1], work[2*idx0+1]);
                __m128 b_re = _mm_set_ps(work[2*(idx1+3)], work[2*(idx1+2)],
                                          work[2*(idx1+1)], work[2*idx1]);
                __m128 b_im = _mm_set_ps(work[2*(idx1+3)+1], work[2*(idx1+2)+1],
                                          work[2*(idx1+1)+1], work[2*idx1+1]);
                
                /* Load twiddles */
                size_t tw_idx = i * twiddle_step;
                __m128 w_re = _mm_set_ps(tw[2*(tw_idx+3*twiddle_step)],
                                          tw[2*(tw_idx+2*twiddle_step)],
                                          tw[2*(tw_idx+twiddle_step)],
                                          tw[2*tw_idx]);
                __m128 w_im = _mm_set_ps(tw[2*(tw_idx+3*twiddle_step)+1],
                                          tw[2*(tw_idx+2*twiddle_step)+1],
                                          tw[2*(tw_idx+twiddle_step)+1],
                                          tw[2*tw_idx+1]);
                
                /* Butterfly */
                sse_bfly2(&a_re, &a_im, &b_re, &b_im, w_re, w_im);
                
                /* Store back */
                float a_re_arr[4], a_im_arr[4], b_re_arr[4], b_im_arr[4];
                _mm_storeu_ps(a_re_arr, a_re);
                _mm_storeu_ps(a_im_arr, a_im);
                _mm_storeu_ps(b_re_arr, b_re);
                _mm_storeu_ps(b_im_arr, b_im);
                
                for (int k = 0; k < 4; k++) {
                    work[2*(idx0+k)] = a_re_arr[k];
                    work[2*(idx0+k)+1] = a_im_arr[k];
                    work[2*(idx1+k)] = b_re_arr[k];
                    work[2*(idx1+k)+1] = b_im_arr[k];
                }
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
    
    /* Copy output (magnitude or complex) */
    for (size_t i = 0; i < n; i++) {
        out[i] = work[2*i];  /* Real part */
    }
    
    free(work);
}

/**
 * @brief SSE-optimized DCT-II
 */
void dspir_x86_sse_dct_ii_f32(const dspir_transform *t,
                               float *restrict out,
                               const float *restrict in) {
    const size_t n = t->n;
    
    /* DCT-II via FFT with pre/post processing */
    float *work = NULL;
    if (posix_memalign((void**)&work, 16, n * 2 * sizeof(float)) != 0) {
        work = malloc(n * 2 * sizeof(float));
    }
    
    /* Reorder input: even indices to real, odd to imag */
    for (size_t i = 0; i < n / 2; i++) {
        work[2*i] = in[2*i];
        work[2*i + 1] = in[2*i + 1];
    }
    
    /* Apply pre-twiddles using SSE */
    const float *tw = (const float *)t->twiddles[0];
    size_t i = 0;
    for (; i + 3 < n / 2; i += 4) {
        __m128 re = _mm_loadu_ps(&work[2*i]);
        __m128 im = _mm_loadu_ps(&work[2*i + 4]);
        __m128 w_re = _mm_loadu_ps(&tw[2*i]);
        __m128 w_im = _mm_loadu_ps(&tw[2*i + 4]);
        
        __m128 new_re = _mm_sub_ps(_mm_mul_ps(re, w_re), _mm_mul_ps(im, w_im));
        __m128 new_im = _mm_add_ps(_mm_mul_ps(re, w_im), _mm_mul_ps(im, w_re));
        
        _mm_storeu_ps(&work[2*i], new_re);
        _mm_storeu_ps(&work[2*i + 4], new_im);
    }
    
    /* Scalar remainder */
    for (; i < n / 2; i++) {
        float re = work[2*i], im = work[2*i + 1];
        float w_re = tw[2*i], w_im = tw[2*i + 1];
        work[2*i] = re * w_re - im * w_im;
        work[2*i + 1] = re * w_im + im * w_re;
    }
    
    /* Execute n/2 point complex FFT (would call FFT routine) */
    /* For now, copy to output */
    memcpy(out, work, n * sizeof(float));
    
    free(work);
}

/**
 * @brief SSE-optimized Haar wavelet
 */
void dspir_x86_sse_haar_f32(const dspir_transform *t,
                             float *restrict out,
                             const float *restrict in) {
    const size_t n = t->n;
    
    float *work = NULL;
    if (posix_memalign((void**)&work, 16, n * sizeof(float)) != 0) {
        work = malloc(n * sizeof(float));
    }
    memcpy(work, in, n * sizeof(float));
    
    const float scale = 0.7071067811865476f; /* 1/sqrt(2) */
    const __m128 vscale = _mm_set1_ps(scale);
    
    size_t current_n = n;
    while (current_n >= 8) {
        /* Process 4 pairs at a time */
        size_t i = 0;
        for (; i + 7 < current_n; i += 8) {
            __m128 even = _mm_loadu_ps(&work[i]);
            __m128 odd = _mm_loadu_ps(&work[i + 4]);
            
            /* Deinterleave */
            __m128 x0 = _mm_shuffle_ps(even, odd, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 x1 = _mm_shuffle_ps(even, odd, _MM_SHUFFLE(3, 1, 3, 1));
            
            /* Predict and update */
            __m128 diff = _mm_sub_ps(x1, x0);
            __m128 sum = _mm_add_ps(x0, _mm_mul_ps(diff, _mm_set1_ps(0.5f)));
            
            /* Scale */
            sum = _mm_mul_ps(sum, vscale);
            diff = _mm_mul_ps(diff, vscale);
            
            /* Store */
            _mm_storeu_ps(&work[i/2], sum);
            _mm_storeu_ps(&work[current_n/2 + i/2], diff);
        }
        
        /* Scalar remainder */
        for (; i < current_n; i += 2) {
            float even = work[i];
            float odd = work[i + 1];
            float diff = (odd - even) * scale;
            float sum = (even + odd) * scale;
            work[i/2] = sum;
            work[current_n/2 + i/2] = diff;
        }
        
        current_n /= 2;
    }
    
    memcpy(out, work, n * sizeof(float));
    free(work);
}

#endif /* DSPIR_ARCH_X86_64 */
