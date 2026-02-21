/**
 * @file test_x86.cpp
 * @brief x86-specific tests
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"

#ifdef DSPIR_ARCH_X86_64

#include <immintrin.h>

/* Test SSE availability */
TEST(X86Test, SSEAvailable) {
    /* SSE is always available on x86_64 */
    EXPECT_TRUE(true);
}

/* Test AVX2 availability */
TEST(X86Test, AVX2Available) {
    #ifdef DSPIR_HAVE_AVX2
        /* Try to execute an AVX2 instruction */
        __m256i a = _mm256_set1_epi32(1);
        __m256i b = _mm256_set1_epi32(2);
        __m256i c = _mm256_add_epi32(a, b);
        
        int result[8];
        _mm256_storeu_si256((__m256i*)result, c);
        
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(result[i], 3);
        }
    #else
        GTEST_SKIP() << "AVX2 not available";
    #endif
}

/* Test AVX-512 availability */
TEST(X86Test, AVX512Available) {
    #ifdef DSPIR_HAVE_AVX512
        /* Try to execute an AVX-512 instruction */
        __m512 a = _mm512_set1_ps(1.0f);
        __m512 b = _mm512_set1_ps(2.0f);
        __m512 c = _mm512_add_ps(a, b);
        
        float result[16];
        _mm512_storeu_ps(result, c);
        
        for (int i = 0; i < 16; i++) {
            EXPECT_FLOAT_EQ(result[i], 3.0f);
        }
    #else
        GTEST_SKIP() << "AVX-512 not available";
    #endif
}

/* Test FMA availability */
TEST(X86Test, FMAAvailable) {
    #ifdef DSPIR_HAVE_AVX2
        __m256 a = _mm256_set1_ps(2.0f);
        __m256 b = _mm256_set1_ps(3.0f);
        __m256 c = _mm256_set1_ps(1.0f);
        __m256 result = _mm256_fmadd_ps(a, b, c);  /* 2*3 + 1 = 7 */
        
        float vals[8];
        _mm256_storeu_ps(vals, result);
        
        for (int i = 0; i < 8; i++) {
            EXPECT_FLOAT_EQ(vals[i], 7.0f);
        }
    #else
        GTEST_SKIP() << "FMA not available";
    #endif
}

#endif /* DSPIR_ARCH_X86_64 */
