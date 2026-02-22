/**
 * @file test_arm.cpp
 * @brief ARM-specific tests
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"

#ifdef FAF_ARCH_AARCH64

#include <arm_neon.h>

/* Test NEON availability */
TEST(ARMTest, NEONAvailable) {
    /* NEON is always available on AArch64 */
    float32x4_t a = vdupq_n_f32(1.0f);
    float32x4_t b = vdupq_n_f32(2.0f);
    float32x4_t c = vaddq_f32(a, b);
    
    float result[4];
    vst1q_f32(result, c);
    
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result[i], 3.0f);
    }
}

/* Test NEON FMA */
TEST(ARMTest, NEONFMA) {
    float32x4_t a = vdupq_n_f32(2.0f);
    float32x4_t b = vdupq_n_f32(3.0f);
    float32x4_t c = vdupq_n_f32(1.0f);
    
    /* FMA: a*b + c */
    float32x4_t result = vmlaq_f32(c, a, b);
    
    float vals[4];
    vst1q_f32(vals, result);
    
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(vals[i], 7.0f);
    }
}

/* Test NEON multiply */
TEST(ARMTest, NEONMultiply) {
    float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    float32x4_t a = vld1q_f32(a_vals);
    float32x4_t b = vld1q_f32(b_vals);
    float32x4_t c = vmulq_f32(a, b);
    
    float result[4];
    vst1q_f32(result, c);
    
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 6.0f);
    EXPECT_FLOAT_EQ(result[2], 12.0f);
    EXPECT_FLOAT_EQ(result[3], 20.0f);
}

/* Test SVE availability if compiled with SVE support */
TEST(ARMTest, SVEAvailable) {
    #ifdef FAF_HAVE_SVE
        /* Get vector length */
        uint64_t vl = svcntw();
        EXPECT_GE(vl, 4);   /* At least 128 bits */
        EXPECT_LE(vl, 64);  /* At most 2048 bits */
        
        /* Try SVE operations */
        svfloat32_t a = svdup_f32(1.0f);
        svfloat32_t b = svdup_f32(2.0f);
        svbool_t pg = svptrue_b32();
        svfloat32_t c = svadd_f32_z(pg, a, b);
        
        /* Sum all elements */
        float sum = svaddv_f32(pg, c);
        EXPECT_FLOAT_EQ(sum, 3.0f * vl);
    #else
        GTEST_SKIP() << "SVE not available";
    #endif
}

/* Test SVE FMA */
TEST(ARMTest, SVEFMA) {
    #ifdef FAF_HAVE_SVE
        svfloat32_t a = svdup_f32(2.0f);
        svfloat32_t b = svdup_f32(3.0f);
        svfloat32_t c = svdup_f32(1.0f);
        svbool_t pg = svptrue_b32();
        
        /* FMA: a*b + c */
        svfloat32_t result = svmla_f32_z(pg, c, a, b);
        
        float sum = svaddv_f32(pg, result);
        uint64_t vl = svcntw();
        EXPECT_FLOAT_EQ(sum, 7.0f * vl);
    #else
        GTEST_SKIP() << "SVE not available";
    #endif
}

#endif /* FAF_ARCH_AARCH64 */
