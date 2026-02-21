/**
 * @file dspir_arm_sve.c
 * @brief ARM SVE (Scalable Vector Extension) kernels for AArch64
 * 
 * SVE provides variable-width vector registers (128-2048 bits)
 * The vector length is determined at runtime
 */

#include "dspir.h"

#ifdef DSPIR_ARCH_AARCH64

#ifdef DSPIR_HAVE_SVE

#include <arm_sve.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Get the SVE vector length in floats
 */
static inline uint64_t sve_vector_length_f32(void) {
    return svcntw();  /* Count of 32-bit elements */
}

/**
 * @brief Complex multiplication using SVE
 * 
 * (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
 */
static inline svfloat32_t sve_complex_mul_re(svfloat32_t a_re, svfloat32_t a_im,
                                              svfloat32_t b_re, svfloat32_t b_im,
                                              svbool_t pg) {
    /* ac - bd */
    svfloat32_t ac = svmul_f32_z(pg, a_re, b_re);
    svfloat32_t bd = svmul_f32_z(pg, a_im, b_im);
    return svsub_f32_z(pg, ac, bd);
}

static inline svfloat32_t sve_complex_mul_im(svfloat32_t a_re, svfloat32_t a_im,
                                              svfloat32_t b_re, svfloat32_t b_im,
                                              svbool_t pg) {
    /* ad + bc */
    svfloat32_t ad = svmul_f32_z(pg, a_re, b_im);
    svfloat32_t bc = svmul_f32_z(pg, a_im, b_re);
    return svadd_f32_z(pg, ad, bc);
}

/**
 * @brief SVE-optimized FFT execution
 * 
 * Uses variable-width vectors for optimal performance across
 * different SVE implementations (128-bit to 2048-bit)
 */
void dspir_arm_sve_execute_f32(const dspir_transform *t,
                                float *restrict out,
                                const float *restrict in) {
    const size_t n = t->n;
    const float *tw = (const float *)t->twiddles[0];
    
    /* Get SVE vector length */
    const uint64_t vl = sve_vector_length_f32();
    const svbool_t pg = svptrue_b32();
    
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
            
            /* Process vl butterflies at a time with SVE */
            size_t i = 0;
            for (; i + vl <= stride; i += vl) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                /* Create indices for gather */
                svuint32_t indices = svindex_u32(0, 1);
                
                /* Load a values */
                svfloat32_t a_re = svld1_gather_index_f32(pg, &work[2*idx0], indices);
                svfloat32_t a_im = svld1_gather_index_f32(pg, &work[2*idx0 + vl], indices);
                
                /* Load b values */
                svfloat32_t b_re = svld1_gather_index_f32(pg, &work[2*idx1], indices);
                svfloat32_t b_im = svld1_gather_index_f32(pg, &work[2*idx1 + vl], indices);
                
                /* Load twiddles */
                size_t tw_idx = i * twiddle_step;
                svfloat32_t w_re = svld1_f32(pg, &tw[2*tw_idx]);
                svfloat32_t w_im = svld1_f32(pg, &tw[2*tw_idx + vl]);
                
                /* Complex multiply b by twiddle */
                svfloat32_t t_re = sve_complex_mul_re(b_re, b_im, w_re, w_im, pg);
                svfloat32_t t_im = sve_complex_mul_im(b_re, b_im, w_re, w_im, pg);
                
                /* Butterfly */
                svfloat32_t new_a_re = svadd_f32_z(pg, a_re, t_re);
                svfloat32_t new_a_im = svadd_f32_z(pg, a_im, t_im);
                svfloat32_t new_b_re = svsub_f32_z(pg, a_re, t_re);
                svfloat32_t new_b_im = svsub_f32_z(pg, a_im, t_im);
                
                /* Store back */
                svst1_scatter_index_f32(pg, &work[2*idx0], indices, new_a_re);
                svst1_scatter_index_f32(pg, &work[2*idx0 + vl], indices, new_a_im);
                svst1_scatter_index_f32(pg, &work[2*idx1], indices, new_b_re);
                svst1_scatter_index_f32(pg, &work[2*idx1 + vl], indices, new_b_im);
            }
            
            /* NEON remainder (4 elements) */
            for (; i + 3 < stride; i += 4) {
                size_t idx0 = base + i;
                size_t idx1 = base + i + stride;
                
                float32x4_t a_re = vld1q_f32(&work[2*idx0]);
                float32x4_t a_im = vld1q_f32(&work[2*idx0 + 4]);
                float32x4_t b_re = vld1q_f32(&work[2*idx1]);
                float32x4_t b_im = vld1q_f32(&work[2*idx1 + 4]);
                
                size_t tw_idx = i * twiddle_step;
                float32x4_t w_re = vld1q_f32(&tw[2*tw_idx]);
                float32x4_t w_im = vld1q_f32(&tw[2*tw_idx + 4]);
                
                float32x4_t t_re = vsubq_f32(vmulq_f32(b_re, w_re), vmulq_f32(b_im, w_im));
                float32x4_t t_im = vaddq_f32(vmulq_f32(b_re, w_im), vmulq_f32(b_im, w_re));
                
                vst1q_f32(&work[2*idx0], vaddq_f32(a_re, t_re));
                vst1q_f32(&work[2*idx0 + 4], vaddq_f32(a_im, t_im));
                vst1q_f32(&work[2*idx1], vsubq_f32(a_re, t_re));
                vst1q_f32(&work[2*idx1 + 4], vsubq_f32(a_im, t_im));
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
 * @brief SVE-optimized polyphase FIR
 * 
 * Uses SVE's horizontal add reduction for efficient accumulation
 */
void dspir_arm_sve_polyfir_f32(const dspir_transform *t,
                                float *restrict out,
                                const float *restrict in) {
    const size_t nch = 8;
    const size_t taps = 64;
    const float *coeffs = (const float *)t->twiddles[0];
    
    const uint64_t vl = sve_vector_length_f32();
    const svbool_t pg = svptrue_b32();
    
    for (size_t ch = 0; ch < nch; ch++) {
        const float *x = &in[ch * taps];
        
        svfloat32_t acc = svdup_f32(0.0f);
        
        /* Process vl taps at a time */
        size_t k = 0;
        for (; k + vl <= taps; k += vl) {
            svfloat32_t xv = svld1_f32(pg, &x[k]);
            svfloat32_t cv = svld1_f32(pg, &coeffs[k]);
            acc = svmla_f32_z(pg, acc, xv, cv);
        }
        
        /* Horizontal sum using SVE addv */
        float sum = svaddv_f32(pg, acc);
        
        /* NEON remainder (4 elements) */
        float32x4_t acc_neon = vdupq_n_f32(0.0f);
        for (; k + 3 < taps; k += 4) {
            float32x4_t xv = vld1q_f32(&x[k]);
            float32x4_t cv = vld1q_f32(&coeffs[k]);
            acc_neon = vmlaq_f32(acc_neon, xv, cv);
        }
        sum += vgetq_lane_f32(acc_neon, 0) + vgetq_lane_f32(acc_neon, 1) +
               vgetq_lane_f32(acc_neon, 2) + vgetq_lane_f32(acc_neon, 3);
        
        /* Scalar remainder */
        for (; k < taps; k++) {
            sum += x[k] * coeffs[k];
        }
        
        out[ch] = sum;
    }
}

/**
 * @brief SVE-optimized Haar wavelet
 * 
 * Variable-width vectorization adapts to SVE vector length
 */
void dspir_arm_sve_haar_f32(const dspir_transform *t,
                             float *restrict out,
                             const float *restrict in) {
    const size_t n = t->n;
    
    float *work = NULL;
    if (posix_memalign((void**)&work, 64, n * sizeof(float)) != 0) {
        work = malloc(n * sizeof(float));
    }
    memcpy(work, in, n * sizeof(float));
    
    const uint64_t vl = sve_vector_length_f32();
    const svbool_t pg = svptrue_b32();
    
    const float scale = 0.7071067811865476f;
    const svfloat32_t vscale = svdup_f32(scale);
    const svfloat32_t vhalf = svdup_f32(0.5f);
    
    size_t current_n = n;
    while (current_n >= 2 * vl) {
        /* Process vl pairs at a time with SVE */
        size_t i = 0;
        for (; i + 2*vl <= current_n; i += 2*vl) {
            /* Load even and odd elements */
            svfloat32_t even = svld1_f32(pg, &work[i]);
            svfloat32_t odd = svld1_f32(pg, &work[i + vl]);
            
            /* Predict: diff = odd - even */
            svfloat32_t diff = svsub_f32_z(pg, odd, even);
            
            /* Update: sum = even + diff/2 */
            svfloat32_t sum = svadd_f32_z(pg, even, svmul_f32_z(pg, diff, vhalf));
            
            /* Scale */
            sum = svmul_f32_z(pg, sum, vscale);
            diff = svmul_f32_z(pg, diff, vscale);
            
            /* Store */
            svst1_f32(pg, &work[i/2], sum);
            svst1_f32(pg, &work[current_n/2 + i/2], diff);
        }
        
        /* NEON remainder (4 pairs) */
        float32x4_t vscale_neon = vdupq_n_f32(scale);
        float32x4_t vhalf_neon = vdupq_n_f32(0.5f);
        for (; i + 7 < current_n; i += 8) {
            float32x4_t x0 = vld1q_f32(&work[i]);
            float32x4_t x1 = vld1q_f32(&work[i + 4]);
            
            float32x4_t diff = vsubq_f32(x1, x0);
            float32x4_t sum = vaddq_f32(x0, vmulq_f32(diff, vhalf_neon));
            
            sum = vmulq_f32(sum, vscale_neon);
            diff = vmulq_f32(diff, vscale_neon);
            
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

#endif /* DSPIR_HAVE_SVE */
#endif /* DSPIR_ARCH_AARCH64 */
