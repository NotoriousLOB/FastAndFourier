/**
 * @file test_chirp_compositions.cpp
 * @brief Execution tests for Chirp DSL builtin compositions
 *
 * These tests verify that Chirp-compiled transforms actually run and produce
 * numerically sensible results. They focus on the CALL_BUILTIN opcode and
 * simple compositions that do not rely on a full FFT expansion.
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ALIGNED_ALLOC64(n) aligned_alloc(64, (((size_t)(n)) + 63u) & ~(size_t)63u)

class ChirpCompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        chirp_register_standard_builtins();
    }
    void TearDown() override {
        chirp_cleanup();
    }
};

/* Fill an interleaved complex f32 buffer. real_fn receives the sample index and
 * returns the real value; imaginary part is set to zero. */
static void fill_interleaved_f32(float *buf, size_t n,
                                  float (*real_fn)(size_t)) {
    for (size_t i = 0; i < n; i++) {
        buf[2 * i] = real_fn(i);
        buf[2 * i + 1] = 0.0f;
    }
}

static void fill_interleaved_f64(double *buf, size_t n,
                                  double (*real_fn)(size_t)) {
    for (size_t i = 0; i < n; i++) {
        buf[2 * i] = real_fn(i);
        buf[2 * i + 1] = 0.0;
    }
}

/* ============================================================================
 * Standalone builtin execution
 * ============================================================================ */

TEST_F(ChirpCompositionTest, ExecuteSin) {
    faf_transform *t = chirp_compile("(sin)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i * 0.1f; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float expected = std::sin((float)i * 0.1f);
        EXPECT_NEAR(out[2 * i], expected, 1e-5f)
            << "sin mismatch at sample " << i;
        EXPECT_NEAR(out[2 * i + 1], 0.0f, 1e-7f)
            << "imaginary part should remain untouched at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(ChirpCompositionTest, ExecuteExp) {
    faf_transform *t = chirp_compile("(exp)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i * 0.05f; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float expected = std::exp((float)i * 0.05f);
        EXPECT_NEAR(out[2 * i], expected, 1e-5f)
            << "exp mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(ChirpCompositionTest, ExecuteRelu) {
    faf_transform *t = chirp_compile("(relu)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) {
        return (i % 2 == 0) ? -(float)i : (float)i;
    });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float x = (i % 2 == 0) ? -(float)i : (float)i;
        float expected = x > 0.0f ? x : 0.0f;
        EXPECT_NEAR(out[2 * i], expected, 1e-7f)
            << "relu mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(ChirpCompositionTest, ExecuteSqrt) {
    faf_transform *t = chirp_compile("(sqrt)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)(i + 1); });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float expected = std::sqrt((float)(i + 1));
        EXPECT_NEAR(out[2 * i], expected, 1e-5f)
            << "sqrt mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

/* ============================================================================
 * Composition pipelines
 * ============================================================================ */

TEST_F(ChirpCompositionTest, PipelineSinThenExp) {
    faf_transform *t = chirp_compile("(pipeline sin exp)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i * 0.05f; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float expected = std::exp(std::sin((float)i * 0.05f));
        EXPECT_NEAR(out[2 * i], expected, 1e-5f)
            << "pipeline sin->exp mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(ChirpCompositionTest, PipelineSqrtThenSigmoid) {
    faf_transform *t = chirp_compile("(pipeline sqrt sigmoid)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)(i + 1); });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        float x = std::sqrt((float)(i + 1));
        float expected = 1.0f / (1.0f + std::exp(-x));
        EXPECT_NEAR(out[2 * i], expected, 1e-5f)
            << "pipeline sqrt->sigmoid mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

TEST_F(ChirpCompositionTest, BuiltinAfterFFTStageRuns) {
    /* (fft :size N) emits a real staged FFT. Appending a unary builtin must
     * still execute and produce finite output. */
    faf_transform *t = chirp_compile("(pipeline (fft :size 64) sin)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 64u);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i * 0.1f; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < 2 * t->n; i++) {
        EXPECT_TRUE(std::isfinite(out[i]))
            << "non-finite output at index " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

/* ============================================================================
 * Precision variants
 * ============================================================================ */

TEST_F(ChirpCompositionTest, ExecuteSinFP64) {
    faf_transform *t = chirp_compile("(sin)");
    ASSERT_NE(t, nullptr);
    t->precision = FAF_PREC_FP64;

    double *in = (double*)ALIGNED_ALLOC64(2 * t->n * sizeof(double));
    double *out = (double*)ALIGNED_ALLOC64(2 * t->n * sizeof(double));
    fill_interleaved_f64(in, t->n, [](size_t i) { return (double)i * 0.1; });

    EXPECT_EQ(faf_execute_f64(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        double expected = std::sin((double)i * 0.1);
        EXPECT_NEAR(out[2 * i], expected, 1e-12)
            << "sin f64 mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

/* ============================================================================
 * Custom builtins
 * ============================================================================ */

static float double_fn_f32(float x) { return x * 2.0f; }
static double double_fn_f64(double x) { return x * 2.0; }

TEST_F(ChirpCompositionTest, ExecuteCustomBuiltin) {
    chirp_register("double_f32", (void(*)(void))double_fn_f32);
    chirp_register("double_f64", (void(*)(void))double_fn_f64);
    chirp_register("double", (void(*)(void))double_fn_f32);

    faf_transform *t = chirp_compile("(double)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    for (size_t i = 0; i < t->n; i++) {
        EXPECT_NEAR(out[2 * i], 2.0f * (float)i, 1e-7f)
            << "custom builtin mismatch at sample " << i;
    }

    free(in); free(out);
    faf_destroy_transform(t);
}

/* ============================================================================
 * JIT fallback for CALL_BUILTIN
 * ============================================================================ */

TEST_F(ChirpCompositionTest, JITRejectsBuiltinTransformAndFallsBack) {
    faf_transform *t = chirp_compile("(sin)");
    ASSERT_NE(t, nullptr);

    faf_jit_ctx *ctx = faf_jit_create();
    ASSERT_NE(ctx, nullptr);

    /* The JIT currently does not generate code for FAF_CALL_BUILTIN, so
     * compilation should fail and the executor should fall back to the VM. */
    EXPECT_NE(faf_jit_compile(ctx, t), 0);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    fill_interleaved_f32(in, t->n, [](size_t i) { return (float)i * 0.1f; });

    EXPECT_EQ(faf_execute_f32(t, out, in), 0);

    faf_jit_destroy(ctx);
    faf_destroy_transform(t);
    free(in); free(out);
}
