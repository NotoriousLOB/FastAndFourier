/**
 * @file faf_arm_neon.c
 * @brief ARM NEON vectorized kernels for AArch64
 * 
 * NEON provides 128-bit registers (4 floats) on AArch64
 */

#include "faf.h"

#ifdef FAF_ARCH_AARCH64

#include <arm_neon.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* NEON register width: 4 floats */
#define NEON_WIDTH 4

/**
 * @brief Complex multiplication using NEON
 * 
 * (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
 */
static inline float32x4_t neon_complex_mul_re(float32x4_t a_re, float32x4_t a_im,
                                               float32x4_t b_re, float32x4_t b_im) {
    /* ac - bd */
    float32x4_t ac = vmulq_f32(a_re, b_re);
    float32x4_t bd = vmulq_f32(a_im, b_im);
    return vsubq_f32(ac, bd);
}

static inline float32x4_t neon_complex_mul_im(float32x4_t a_re, float32x4_t a_im,
                                               float32x4_t b_re, float32x4_t b_im) {
    /* ad + bc */
    float32x4_t ad = vmulq_f32(a_re, b_im);
    float32x4_t bc = vmulq_f32(a_im, b_re);
    return vaddq_f32(ad, bc);
}

/**
 * @brief Radix-2 butterfly using NEON
 */
static inline void neon_bfly2(float32x4_t *a_re, float32x4_t *a_im,
                               float32x4_t *b_re, float32x4_t *b_im,
                               float32x4_t w_re, float32x4_t w_im) {
    /* twiddle b */
    float32x4_t t_re = neon_complex_mul_re(*b_re, *b_im, w_re, w_im);
    float32x4_t t_im = neon_complex_mul_im(*b_re, *b_im, w_re, w_im);
    
    /* butterfly */
    float32x4_t new_a_re = vaddq_f32(*a_re, t_re);
    float32x4_t new_a_im = vaddq_f32(*a_im, t_im);
    float32x4_t new_b_re = vsubq_f32(*a_re, t_re);
    float32x4_t new_b_im = vsubq_f32(*a_im, t_im);
    
    *a_re = new_a_re;
    *a_im = new_a_im;
    *b_re = new_b_re;
    *b_im = new_b_im;
}

/**
 * @brief NEON-optimized DCT-II
 */
void faf_arm_neon_dct_ii_f32(const faf_transform *t,
                                float *restrict out,
                                const float *restrict in) {
    const size_t n = t->n;
    const float *tw = (const float *)t->twiddles[0];
    
    float *work = NULL;
    if (posix_memalign((void**)&work, 16, n * 2 * sizeof(float)) != 0) {
        work = malloc(n * 2 * sizeof(float));
    }
    
    /* Reorder input */
    for (size_t i = 0; i < n / 2; i++) {
        work[2*i] = in[2*i];
        work[2*i + 1] = in[2*i + 1];
    }
    
    /* Apply pre-twiddles using NEON */
    size_t i = 0;
    for (; i + 3 < n / 2; i += 4) {
        float32x4_t re = vld1q_f32(&work[2*i]);
        float32x4_t im = vld1q_f32(&work[2*i + 4]);
        float32x4_t w_re = vld1q_f32(&tw[2*i]);
        float32x4_t w_im = vld1q_f32(&tw[2*i + 4]);
        
        float32x4_t new_re = vsubq_f32(vmulq_f32(re, w_re), vmulq_f32(im, w_im));
        float32x4_t new_im = vaddq_f32(vmulq_f32(re, w_im), vmulq_f32(im, w_re));
        
        vst1q_f32(&work[2*i], new_re);
        vst1q_f32(&work[2*i + 4], new_im);
    }
    
    /* Scalar remainder */
    for (; i < n / 2; i++) {
        float re = work[2*i], im = work[2*i + 1];
        float w_re = tw[2*i], w_im = tw[2*i + 1];
        work[2*i] = re * w_re - im * w_im;
        work[2*i + 1] = re * w_im + im * w_re;
    }
    
    memcpy(out, work, n * sizeof(float));
    free(work);
}

/**
 * @brief NEON-optimized Haar wavelet
 */
void faf_arm_neon_haar_f32(const faf_transform *t,
                              float *restrict out,
                              const float *restrict in) {
    const size_t n = t->n;
    
    float *work = NULL;
    if (posix_memalign((void**)&work, 16, n * sizeof(float)) != 0) {
        work = malloc(n * sizeof(float));
    }
    memcpy(work, in, n * sizeof(float));
    
    const float scale = 0.7071067811865476f;
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    
    size_t current_n = n;
    while (current_n >= 8) {
        /* Process 4 pairs at a time with NEON */
        size_t i = 0;
        for (; i + 7 < current_n; i += 8) {
            float32x4_t x0 = vld1q_f32(&work[i]);
            float32x4_t x1 = vld1q_f32(&work[i + 4]);
            
            /* Predict: diff = x1 - x0 */
            float32x4_t diff = vsubq_f32(x1, x0);
            
            /* Update: sum = x0 + diff/2 */
            float32x4_t sum = vaddq_f32(x0, vmulq_f32(diff, vhalf));
            
            /* Scale */
            sum = vmulq_f32(sum, vscale);
            diff = vmulq_f32(diff, vscale);
            
            /* Store */
            vst1q_f32(&work[i/2], sum);
            vst1q_f32(&work[current_n/2 + i/2], diff);
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

/**
 * @brief NEON-optimized polyphase FIR
 */
void faf_arm_neon_polyfir_f32(const faf_transform *t,
                                 float *restrict out,
                                 const float *restrict in) {
    const size_t nch = 8;
    const size_t taps = 32;
    const float *coeffs = (const float *)t->twiddles[0];
    
    for (size_t ch = 0; ch < nch; ch++) {
        const float *x = &in[ch * taps];
        
        float32x4_t acc_low = vdupq_n_f32(0.0f);
        float32x4_t acc_high = vdupq_n_f32(0.0f);
        
        /* Process 8 taps at a time */
        size_t k = 0;
        for (; k + 7 < taps; k += 8) {
            float32x4_t xv_low = vld1q_f32(&x[k]);
            float32x4_t xv_high = vld1q_f32(&x[k + 4]);
            float32x4_t cv_low = vld1q_f32(&coeffs[k]);
            float32x4_t cv_high = vld1q_f32(&coeffs[k + 4]);
            
            acc_low = vmlaq_f32(acc_low, xv_low, cv_low);
            acc_high = vmlaq_f32(acc_high, xv_high, cv_high);
        }
        
        /* Horizontal sum */
        float32x4_t acc = vaddq_f32(acc_low, acc_high);
        float sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                    vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
        
        /* Scalar remainder */
        for (; k < taps; k++) {
            sum += x[k] * coeffs[k];
        }
        
        out[ch] = sum;
    }
}

#endif /* FAF_ARCH_AARCH64 */
