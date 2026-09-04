/**
 * @file faf_fft_stage_neon.c
 * @brief NEON radix-2 butterflies on split re[]/im[] planes
 *
 * Twiddles are already conjugated at create. Vector path requires
 * stride == 1 (contiguous planes) and tw_step == 1. Interleaved
 * pointers never enter here.
 */

#include "faf.h"

#ifdef FAF_ARCH_AARCH64
#include <arm_neon.h>

void faf_radix2_split_neon_f32(float *re, float *im, size_t n,
                                size_t group, size_t stride, size_t tw_step,
                                const float *tw, size_t ntw) {
    size_t half = group / 2;
    if (half == 0 || group == 0) return;
    size_t ngroups = n / (group * stride);

    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group * stride;
        size_t r = 0;

        if (stride == 1 && tw_step == 1 && tw && half >= 4) {
            for (; r + 3 < half; r += 4) {
                size_t i1 = base + r * stride;
                size_t i2 = i1 + half * stride;

                float32x4_t ar = vld1q_f32(&re[i1]);
                float32x4_t ai = vld1q_f32(&im[i1]);
                float32x4_t br = vld1q_f32(&re[i2]);
                float32x4_t bi = vld1q_f32(&im[i2]);

                float32x4x2_t tw_pairs = vld2q_f32(&tw[2 * r]);
                float32x4_t wr = tw_pairs.val[0];
                float32x4_t wi = tw_pairs.val[1];

                float32x4_t tr = vmlsq_f32(vmulq_f32(br, wr), bi, wi);
                float32x4_t ti = vmlaq_f32(vmulq_f32(bi, wr), br, wi);

                vst1q_f32(&re[i1], vaddq_f32(ar, tr));
                vst1q_f32(&im[i1], vaddq_f32(ai, ti));
                vst1q_f32(&re[i2], vsubq_f32(ar, tr));
                vst1q_f32(&im[i2], vsubq_f32(ai, ti));
            }
        }

        for (; r < half; r++) {
            size_t i1 = base + r * stride;
            size_t i2 = i1 + half * stride;
            if (i1 >= n || i2 >= n) continue;
            float ar = re[i1], ai = im[i1];
            float br = re[i2], bi = im[i2];
            if (tw) {
                size_t idx = r * tw_step;
                if (idx < ntw) {
                    float wr = tw[2 * idx], wi = tw[2 * idx + 1];
                    float tr = br * wr - bi * wi;
                    float ti = br * wi + bi * wr;
                    br = tr; bi = ti;
                }
            }
            re[i1] = ar + br; im[i1] = ai + bi;
            re[i2] = ar - br; im[i2] = ai - bi;
        }
    }
}

void faf_radix2_split_neon_f64(double *re, double *im, size_t n,
                                size_t group, size_t stride, size_t tw_step,
                                const double *tw, size_t ntw) {
    size_t half = group / 2;
    if (half == 0 || group == 0) return;
    size_t ngroups = n / (group * stride);

    for (size_t g = 0; g < ngroups; g++) {
        size_t base = g * group * stride;
        size_t r = 0;

        if (stride == 1 && tw_step == 1 && tw && half >= 2) {
            for (; r + 1 < half; r += 2) {
                size_t i1 = base + r * stride;
                size_t i2 = i1 + half * stride;

                float64x2_t ar = vld1q_f64(&re[i1]);
                float64x2_t ai = vld1q_f64(&im[i1]);
                float64x2_t br = vld1q_f64(&re[i2]);
                float64x2_t bi = vld1q_f64(&im[i2]);

                float64x2_t wr = {tw[2 * r], tw[2 * (r + 1)]};
                float64x2_t wi = {tw[2 * r + 1], tw[2 * (r + 1) + 1]};

                float64x2_t tr = vmlsq_f64(vmulq_f64(br, wr), bi, wi);
                float64x2_t ti = vmlaq_f64(vmulq_f64(bi, wr), br, wi);

                vst1q_f64(&re[i1], vaddq_f64(ar, tr));
                vst1q_f64(&im[i1], vaddq_f64(ai, ti));
                vst1q_f64(&re[i2], vsubq_f64(ar, tr));
                vst1q_f64(&im[i2], vsubq_f64(ai, ti));
            }
        }

        for (; r < half; r++) {
            size_t i1 = base + r * stride;
            size_t i2 = i1 + half * stride;
            if (i1 >= n || i2 >= n) continue;
            double ar = re[i1], ai = im[i1];
            double br = re[i2], bi = im[i2];
            if (tw) {
                size_t idx = r * tw_step;
                if (idx < ntw) {
                    double wr = tw[2 * idx], wi = tw[2 * idx + 1];
                    double tr = br * wr - bi * wi;
                    double ti = br * wi + bi * wr;
                    br = tr; bi = ti;
                }
            }
            re[i1] = ar + br; im[i1] = ai + bi;
            re[i2] = ar - br; im[i2] = ai - bi;
        }
    }
}

#endif /* FAF_ARCH_AARCH64 */
