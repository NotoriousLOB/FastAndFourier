/**
 * @file test_chirp.cpp
 * @brief Comprehensive tests for the Chirp DSL compiler and standard library
 *
 * These tests exercise Chirp parsing, IR generation, builtin registration,
 * and execution of Chirp-compiled transforms. They are designed to cover as
 * much of the Chirp language surface and standard library as possible.
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

/* aligned_alloc requires size to be a multiple of alignment */
#define ALIGNED_ALLOC64(n) aligned_alloc(64, (((size_t)(n)) + 63u) & ~(size_t)63u)

/* ============================================================================
 * Test fixtures and helpers
 * ============================================================================ */

/**
 * @brief RAII guard that registers standard builtins on construction and
 *        cleans up the registry on destruction.
 */
class ChirpRegistryGuard {
public:
    ChirpRegistryGuard() {
        count_ = chirp_register_standard_builtins();
    }
    ~ChirpRegistryGuard() {
        chirp_cleanup();
    }
    int count() const { return count_; }
private:
    int count_;
};

/* Custom builtin used in registration tests */
static void test_custom_fn(void) {
    /* no-op stub */
}

/* Count how many instructions in t have the given opcode */
static size_t count_opcodes(const faf_transform *t, uint8_t op) {
    size_t count = 0;
    for (size_t i = 0; i < t->n_inst; i++) {
        if (FAF_GET_OP(t->code[i].packed) == op) {
            count++;
        }
    }
    return count;
}

/* Return true if t contains at least one instruction with opcode op */
static bool has_opcode(const faf_transform *t, uint8_t op) {
    return count_opcodes(t, op) > 0;
}

/* ============================================================================
 * Builtin registration
 * ============================================================================ */

TEST(ChirpTest, StandardBuiltinsRegistered) {
    ChirpRegistryGuard guard;
    EXPECT_GT(guard.count(), 0);
}

TEST(ChirpTest, CustomRegistration) {
    /* Start from a clean slate */
    chirp_cleanup();
    EXPECT_EQ(chirp_count(), 0);

    int id1 = chirp_register("myfn", (void (*)(void))test_custom_fn);
    EXPECT_GE(id1, 0);
    EXPECT_EQ(chirp_count(), 1);

    int id2 = chirp_register("another", (void (*)(void))test_custom_fn);
    EXPECT_GT(id2, id1);
    EXPECT_EQ(chirp_count(), 2);

    /* Duplicate name updates the existing slot */
    int id3 = chirp_register("myfn", (void (*)(void))test_custom_fn);
    EXPECT_EQ(id3, id1);
    EXPECT_EQ(chirp_count(), 2);

    chirp_cleanup();
}

TEST(ChirpTest, CleanupResetsRegistry) {
    chirp_register("myfn", (void (*)(void))test_custom_fn);
    EXPECT_GT(chirp_count(), 0);
    chirp_cleanup();
    EXPECT_EQ(chirp_count(), 0);
}

/* ============================================================================
 * Core language constructs
 * ============================================================================ */

TEST(ChirpTest, CompileSimpleFFT) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile("(fft :size 64)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, FAF_TRANSFORM_FFT);
    EXPECT_EQ(t->n, 64u);
    EXPECT_EQ(t->precision, FAF_PREC_FP32);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));

    faf_destroy_transform(t);
}

TEST(ChirpTest, FFTDefaultSize) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile("(fft)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 64u);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));

    faf_destroy_transform(t);
}

TEST(ChirpTest, FFTViaPositionalArg) {
    ChirpRegistryGuard guard;

    /* The parser also accepts a bare number as the size argument. */
    faf_transform *t = chirp_compile("(fft 128)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 128u);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));

    faf_destroy_transform(t);
}

TEST(ChirpTest, CompileTwiddle) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile("(pipeline (fft :size 64) twiddle)");
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));
    EXPECT_TRUE(has_opcode(t, FAF_TWIDDLE_MUL));

    faf_destroy_transform(t);
}

TEST(ChirpTest, CompileButterflyRadices) {
    ChirpRegistryGuard guard;

    faf_transform *t2 = chirp_compile("(pipeline (fft :size 64) (bfly 2))");
    ASSERT_NE(t2, nullptr);
    EXPECT_TRUE(has_opcode(t2, FAF_BFLY2));
    faf_destroy_transform(t2);

    faf_transform *t4 = chirp_compile("(pipeline (fft :size 64) (bfly 4))");
    ASSERT_NE(t4, nullptr);
    EXPECT_TRUE(has_opcode(t4, FAF_BFLY4));
    faf_destroy_transform(t4);

    faf_transform *t8 = chirp_compile("(pipeline (fft :size 64) (bfly 8))");
    ASSERT_NE(t8, nullptr);
    EXPECT_TRUE(has_opcode(t8, FAF_BFLY8));
    faf_destroy_transform(t8);

    /* Invalid radix falls back to BFLY2 */
    faf_transform *tbad = chirp_compile("(pipeline (fft :size 64) (bfly 5))");
    ASSERT_NE(tbad, nullptr);
    EXPECT_TRUE(has_opcode(tbad, FAF_BFLY2));
    faf_destroy_transform(tbad);
}

TEST(ChirpTest, CompileReduce) {
    ChirpRegistryGuard guard;

    faf_transform *ts = chirp_compile("(pipeline (fft :size 64) reduce-sum)");
    ASSERT_NE(ts, nullptr);
    EXPECT_TRUE(has_opcode(ts, FAF_REDUCE_SUM));
    faf_destroy_transform(ts);

    faf_transform *tm = chirp_compile("(pipeline (fft :size 64) reduce-max)");
    ASSERT_NE(tm, nullptr);
    EXPECT_TRUE(has_opcode(tm, FAF_REDUCE_MAX));
    faf_destroy_transform(tm);

    faf_transform *tn = chirp_compile("(pipeline (fft :size 64) reduce-min)");
    ASSERT_NE(tn, nullptr);
    EXPECT_TRUE(has_opcode(tn, FAF_REDUCE_MIN));
    faf_destroy_transform(tn);
}

TEST(ChirpTest, CompileLift) {
    ChirpRegistryGuard guard;

    /* Use standard builtins that are registered as predict/update targets. */
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (fft :size 64)"
        "  (lift :predict sigmoid :update relu))"
    );
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(has_opcode(t, FAF_LIFT_PRED));
    EXPECT_TRUE(has_opcode(t, FAF_LIFT_UPD));

    faf_destroy_transform(t);
}

TEST(ChirpTest, CustomVsFirstClassSyntax) {
    ChirpRegistryGuard guard;

    /* Old explicit wrapper */
    faf_transform *t1 = chirp_compile("(pipeline (fft :size 64) (custom sigmoid))");
    ASSERT_NE(t1, nullptr);
    EXPECT_TRUE(has_opcode(t1, FAF_CALL_BUILTIN));
    faf_destroy_transform(t1);

    /* New first-class syntax */
    faf_transform *t2 = chirp_compile("(pipeline (fft :size 64) sigmoid)");
    ASSERT_NE(t2, nullptr);
    EXPECT_TRUE(has_opcode(t2, FAF_CALL_BUILTIN));
    faf_destroy_transform(t2);
}

TEST(ChirpTest, PipelineSequencing) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile(
        "(pipeline "
        "  (fft :size 64)"
        "  twiddle"
        "  (bfly 4)"
        "  (bfly 2)"
        "  reduce-sum)"
    );
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(count_opcodes(t, FAF_FFT_STAGE), 1u);
    EXPECT_EQ(count_opcodes(t, FAF_TWIDDLE_MUL), 1u);
    EXPECT_EQ(count_opcodes(t, FAF_BFLY4), 1u);
    EXPECT_EQ(count_opcodes(t, FAF_BFLY2), 1u);
    EXPECT_EQ(count_opcodes(t, FAF_REDUCE_SUM), 1u);

    faf_destroy_transform(t);
}

/* ============================================================================
 * Standard library coverage (compile-time)
 * ============================================================================ */

TEST(ChirpTest, TrigBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) sin)",
        "(pipeline (fft :size 64) cos)",
        "(pipeline (fft :size 64) tan)",
        "(pipeline (fft :size 64) asin)",
        "(pipeline (fft :size 64) acos)",
        "(pipeline (fft :size 64) atan)",
        "(pipeline (fft :size 64) sinh)",
        "(pipeline (fft :size 64) cosh)",
        "(pipeline (fft :size 64) tanh)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, ExpLogBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) exp)",
        "(pipeline (fft :size 64) exp2)",
        "(pipeline (fft :size 64) log)",
        "(pipeline (fft :size 64) log2)",
        "(pipeline (fft :size 64) log10)",
        "(pipeline (fft :size 64) sqrt)",
        "(pipeline (fft :size 64) cbrt)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, SpecialFunctionBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) erf)",
        "(pipeline (fft :size 64) erfc)",
        "(pipeline (fft :size 64) tgamma)",
        "(pipeline (fft :size 64) lgamma)",
        "(pipeline (fft :size 64) j0)",
        "(pipeline (fft :size 64) j1)",
        "(pipeline (fft :size 64) y0)",
        "(pipeline (fft :size 64) y1)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, DistributionBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) gaussian_pdf)",
        "(pipeline (fft :size 64) cauchy_pdf)",
        "(pipeline (fft :size 64) exponential_pdf)",
        "(pipeline (fft :size 64) lognormal_pdf)",
        "(pipeline (fft :size 64) laplace_pdf)",
        "(pipeline (fft :size 64) beta_pdf)",
        "(pipeline (fft :size 64) gamma_pdf)",
        "(pipeline (fft :size 64) weibull_pdf)",
        "(pipeline (fft :size 64) uniform_pdf)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, ActivationBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) sigmoid)",
        "(pipeline (fft :size 64) relu)",
        "(pipeline (fft :size 64) gelu)",
        "(pipeline (fft :size 64) swish)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, VectorBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) sum)",
        "(pipeline (fft :size 64) dot)",
        "(pipeline (fft :size 64) norm)",
        "(pipeline (fft :size 64) norm1)",
        "(pipeline (fft :size 64) norm_inf)",
        "(pipeline (fft :size 64) mean)",
        "(pipeline (fft :size 64) variance)",
        "(pipeline (fft :size 64) std)",
        "(pipeline (fft :size 64) cumsum)",
        "(pipeline (fft :size 64) mul)",
        "(pipeline (fft :size 64) add)",
        "(pipeline (fft :size 64) sub)",
        "(pipeline (fft :size 64) div)",
        "(pipeline (fft :size 64) scale)",
        "(pipeline (fft :size 64) saxpy)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, WindowBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) hann_window)",
        "(pipeline (fft :size 64) hamming_window)",
        "(pipeline (fft :size 64) blackman_window)",
        "(pipeline (fft :size 64) flattop_window)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, WaveletBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) haar)",
        "(pipeline (fft :size 64) haar_inverse)",
        "(pipeline (fft :size 64) daubechies4)",
        "(pipeline (fft :size 64) morlet)",
        "(pipeline (fft :size 64) mexican_hat)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, UtilityBuiltinsCompile) {
    ChirpRegistryGuard guard;

    const char *programs[] = {
        "(pipeline (fft :size 64) clamp)",
        "(pipeline (fft :size 64) lerp)",
        "(pipeline (fft :size 64) smoothstep)",
        "(pipeline (fft :size 64) smootherstep)",
        "(pipeline (fft :size 64) step)",
        "(pipeline (fft :size 64) sign)",
        "(pipeline (fft :size 64) deg2rad)",
        "(pipeline (fft :size 64) rad2deg)",
    };

    for (const char *src : programs) {
        faf_transform *t = chirp_compile(src);
        ASSERT_NE(t, nullptr) << "Failed to compile: " << src;
        EXPECT_TRUE(has_opcode(t, FAF_CALL_BUILTIN)) << src;
        faf_destroy_transform(t);
    }
}

TEST(ChirpTest, ComplexPipelineCompiles) {
    ChirpRegistryGuard guard;

    /* A kitchen-sink pipeline touching many language features and builtins. */
    faf_transform *t = chirp_compile(
        "(pipeline "
        "  hann_window"
        "  (fft :size 256)"
        "  twiddle"
        "  (bfly 4)"
        "  (bfly 2)"
        "  sigmoid"
        "  gelu"
        "  (lift :predict gaussian_pdf :update laplace_pdf)"
        "  reduce-sum)"
    );
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 256u);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));
    EXPECT_TRUE(has_opcode(t, FAF_TWIDDLE_MUL));
    EXPECT_TRUE(has_opcode(t, FAF_BFLY4));
    EXPECT_TRUE(has_opcode(t, FAF_BFLY2));
    EXPECT_TRUE(has_opcode(t, FAF_LIFT_PRED));
    EXPECT_TRUE(has_opcode(t, FAF_LIFT_UPD));
    EXPECT_TRUE(has_opcode(t, FAF_REDUCE_SUM));
    EXPECT_GE(count_opcodes(t, FAF_CALL_BUILTIN), 3u);

    faf_destroy_transform(t);
}

/* ============================================================================
 * Execution of Chirp-compiled transforms (non-builtin constructs)
 * ============================================================================ */

TEST(ChirpTest, ExecuteFFTPipelineDoesNotCrash) {
    ChirpRegistryGuard guard;

    /* The Chirp (fft :size N) form emits a single FAF_FFT_STAGE instruction.
     * It is a high-level construct and does not expand to a complete FFT
     * bytecode; we only verify here that executing it does not crash. */
    faf_transform *t = chirp_compile("(fft :size 64)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));

    for (size_t i = 0; i < t->n; i++) {
        in[2*i] = (i == 0) ? 1.0f : 0.0f;
        in[2*i + 1] = 0.0f;
    }

    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);

    free(in);
    free(out);
    faf_destroy_transform(t);
}

TEST(ChirpTest, ChirpFFTInstructionMatchesRequestedSize) {
    ChirpRegistryGuard guard;

    /* Verify that the high-level FFT form encodes the requested size in the
     * emitted FAF_FFT_STAGE instruction. */
    faf_transform *t = chirp_compile("(fft :size 128)");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n, 128u);

    bool found = false;
    for (size_t i = 0; i < t->n_inst; i++) {
        if (FAF_GET_OP(t->code[i].packed) == FAF_FFT_STAGE) {
            EXPECT_EQ(t->code[i].a0, 128u);
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    faf_destroy_transform(t);
}

TEST(ChirpTest, ExecuteReduceSum) {
    ChirpRegistryGuard guard;

    /* reduce-sum is implemented in the VM, so this should actually run. */
    faf_transform *t = chirp_compile("(pipeline (fft :size 64) reduce-sum)");
    ASSERT_NE(t, nullptr);

    float *in = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));
    float *out = (float*)ALIGNED_ALLOC64(2 * t->n * sizeof(float));

    for (size_t i = 0; i < t->n; i++) {
        in[2*i] = 1.0f;
        in[2*i + 1] = 0.0f;
    }

    int result = faf_execute_f32(t, out, in);
    EXPECT_EQ(result, 0);

    free(in);
    free(out);
    faf_destroy_transform(t);
}

/* ============================================================================
 * Error handling
 * ============================================================================ */

TEST(ChirpTest, UnknownBuiltin) {
    ChirpRegistryGuard guard;

    /* Unknown bare symbols compile as literals, not builtins. */
    faf_transform *t = chirp_compile("(pipeline (fft :size 64) definitely_not_a_builtin)");
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(has_opcode(t, FAF_CALL_BUILTIN));
    faf_destroy_transform(t);
}

TEST(ChirpTest, UnbalancedParens) {
    ChirpRegistryGuard guard;

    /* The current parser emits a warning to stderr but returns a partial
     * transform for unbalanced input. We verify it does not crash and that
     * the returned transform has at least the instructions that were parsed. */
    faf_transform *t = chirp_compile("(pipeline (fft :size 64)");
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));
    faf_destroy_transform(t);
}

TEST(ChirpTest, EmptyProgram) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile("");
    EXPECT_EQ(t, nullptr);
}

TEST(ChirpTest, CommentsAreIgnored) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile(
        "; this is a comment\n"
        "(pipeline\n"
        "  (fft :size 64) ; inline comment\n"
        "  twiddle\n"
        ")"
    );
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(has_opcode(t, FAF_FFT_STAGE));
    EXPECT_TRUE(has_opcode(t, FAF_TWIDDLE_MUL));
    faf_destroy_transform(t);
}

/* ============================================================================
 * IR shape verification
 * ============================================================================ */

TEST(ChirpTest, TransformEndsWithEND) {
    ChirpRegistryGuard guard;

    faf_transform *t = chirp_compile("(fft :size 64)");
    ASSERT_NE(t, nullptr);
    ASSERT_GT(t->n_inst, 0u);
    EXPECT_EQ(FAF_GET_OP(t->code[t->n_inst - 1].packed), FAF_END);

    faf_destroy_transform(t);
}
