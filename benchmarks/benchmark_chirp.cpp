/**
 * @file benchmark_chirp.cpp
 * @brief Benchmarks for the Chirp DSL compiler and Chirp-originated transforms
 *
 * These benchmarks measure Chirp compilation time for programs of varying
 * complexity. Execution benchmarks are limited to constructs that are fully
 * implemented in the VM/JIT; the high-level (fft :size N) form emitted by
 * Chirp is a single FAF_FFT_STAGE instruction and is not expanded to a
 * complete transform bytecode here.
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <cmath>
#include <cstring>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Ensure standard builtins are registered exactly once for this TU. */
static void ensure_chirp_builtins(void) {
    static bool initialized = []() {
        chirp_register_standard_builtins();
        return true;
    }();
    (void)initialized;
}

/* ============================================================================
 * Compilation benchmarks
 * ============================================================================ */

static void BM_Chirp_Compile_SimpleFFT(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);
    const std::string src = "(fft :size " + std::to_string(n) + ")";

    for (auto _ : state) {
        faf_transform* t = chirp_compile(src.c_str());
        benchmark::DoNotOptimize(t);
        faf_destroy_transform(t);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("size=" + std::to_string(n));
}
BENCHMARK(BM_Chirp_Compile_SimpleFFT)
    ->Range(16, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Compile_FFTPipeline(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);
    const std::string src =
        "(pipeline "
        "  (fft :size " + std::to_string(n) + ")"
        "  twiddle"
        "  (bfly 4)"
        "  (bfly 2)"
        "  reduce-sum)";

    for (auto _ : state) {
        faf_transform* t = chirp_compile(src.c_str());
        benchmark::DoNotOptimize(t);
        faf_destroy_transform(t);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("size=" + std::to_string(n));
}
BENCHMARK(BM_Chirp_Compile_FFTPipeline)
    ->Range(16, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Compile_KitchenSink(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);
    const std::string src =
        "(pipeline "
        "  hann_window"
        "  (fft :size " + std::to_string(n) + ")"
        "  twiddle"
        "  (bfly 4)"
        "  sigmoid"
        "  gelu"
        "  (lift :predict gaussian_pdf :update laplace_pdf)"
        "  exp"
        "  log"
        "  sqrt"
        "  sin"
        "  cos"
        "  tanh"
        "  relu"
        "  swish"
        "  sum"
        "  norm"
        "  mean"
        "  clamp"
        "  lerp"
        "  smoothstep"
        "  reduce-sum)";

    for (auto _ : state) {
        faf_transform* t = chirp_compile(src.c_str());
        benchmark::DoNotOptimize(t);
        faf_destroy_transform(t);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("size=" + std::to_string(n));
}
BENCHMARK(BM_Chirp_Compile_KitchenSink)
    ->Range(16, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Compile_StandardLibraryTour(benchmark::State& state) {
    ensure_chirp_builtins();
    (void)state.range(0);

    /* A program that references one builtin from every major category.
     * Short names are used to exercise the alias registration path. */
    const char* src =
        "(pipeline "
        "  (fft :size 256)"
        "  sin cos tan"
        "  exp log sqrt cbrt"
        "  erf tgamma j0 y0"
        "  gaussian_pdf cauchy_pdf exponential_pdf laplace_pdf"
        "  sigmoid relu gelu swish"
        "  sum dot norm norm1 norm_inf mean variance std"
        "  cumsum mul add sub div scale saxpy"
        "  hann_window hamming_window blackman_window flattop_window"
        "  haar daubechies4 morlet mexican_hat"
        "  clamp lerp smoothstep smootherstep step sign deg2rad rad2deg"
        "  reduce-sum)";

    for (auto _ : state) {
        faf_transform* t = chirp_compile(src);
        benchmark::DoNotOptimize(t);
        faf_destroy_transform(t);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Chirp_Compile_StandardLibraryTour)
    ->Range(16, 16)
    ->Unit(benchmark::kMicrosecond);

/* ============================================================================
 * Execution benchmarks for fully-implemented Chirp constructs
 * ============================================================================ */

static void BM_Chirp_Execute_ReduceSum(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);

    /* The high-level (fft :size N) form does not expand to a complete FFT
     * bytecode, so this execution benchmark uses a standalone reduce-sum
     * which is fully implemented in the VM. */
    faf_transform* t = chirp_compile("reduce-sum");
    if (!t) {
        state.SkipWithError("Failed to compile reduce-sum");
        return;
    }

    float* in = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));

    for (size_t i = 0; i < 2 * t->n; i++) {
        in[i] = 1.0f;
    }

    for (auto _ : state) {
        int result = faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(state.iterations() * t->n);
    state.SetBytesProcessed(state.iterations() * t->n * sizeof(float) * 2);
    state.SetLabel("size=" + std::to_string(t->n));

    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_ReduceSum)
    ->Range(16, 16)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Execute_Sin(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);

    faf_transform* t = chirp_compile("(sin)");
    if (!t) {
        state.SkipWithError("Failed to compile (sin)");
        return;
    }

    float* in = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));

    for (size_t i = 0; i < t->n; i++) {
        in[2 * i] = (float)i * 0.1f;
        in[2 * i + 1] = 0.0f;
    }

    for (auto _ : state) {
        int result = faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(state.iterations() * t->n);
    state.SetBytesProcessed(state.iterations() * t->n * sizeof(float) * 2);
    state.SetLabel("size=" + std::to_string(t->n));

    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_Sin)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Execute_PipelineSinExp(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);

    faf_transform* t = chirp_compile("(pipeline sin exp)");
    if (!t) {
        state.SkipWithError("Failed to compile (pipeline sin exp)");
        return;
    }

    float* in = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));

    for (size_t i = 0; i < t->n; i++) {
        in[2 * i] = (float)i * 0.05f;
        in[2 * i + 1] = 0.0f;
    }

    for (auto _ : state) {
        int result = faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(state.iterations() * t->n);
    state.SetBytesProcessed(state.iterations() * t->n * sizeof(float) * 2);
    state.SetLabel("size=" + std::to_string(t->n));

    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_PipelineSinExp)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Execute_ScalarBuiltinTour(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = state.range(0);

    /* A pipeline of scalar builtins that are safe for element-wise execution.
     * Vector builtins (sum, norm, etc.) are excluded because they require a
     * size argument and are not supported by the current CALL_BUILTIN handler. */
    faf_transform* t = chirp_compile(
        "(pipeline "
        "  sin cos"
        "  exp log sqrt"
        "  sigmoid relu gelu tanh"
        "  clamp sign"
        ")"
    );
    if (!t) {
        state.SkipWithError("Failed to compile scalar builtin tour");
        return;
    }

    float* in = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * t->n * sizeof(float));

    for (size_t i = 0; i < t->n; i++) {
        in[2 * i] = 0.5f + (float)i * 0.01f;
        in[2 * i + 1] = 0.0f;
    }

    for (auto _ : state) {
        int result = faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(result);
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(state.iterations() * t->n);
    state.SetBytesProcessed(state.iterations() * t->n * sizeof(float) * 2);
    state.SetLabel("size=" + std::to_string(t->n));

    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_ScalarBuiltinTour)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Execute_DWT(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = (size_t)state.range(0);
    const std::string src =
        "(dwt :family cdf97 :size " + std::to_string(n) + " :levels 4)";
    faf_transform* t = chirp_compile(src.c_str());
    if (!t) {
        state.SkipWithError("Failed to compile Chirp DWT");
        return;
    }
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2 * i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2 * i + 1] = 0.0f;
    }
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetLabel("chirp-cdf97");
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_DWT)->Range(64, 4096)->Unit(benchmark::kMicrosecond);

static void BM_Chirp_Execute_Denoise(benchmark::State& state) {
    ensure_chirp_builtins();
    const size_t n = 1024;
    faf_transform* t = chirp_compile(
        "(pipeline "
        "  (dwt :family cdf97 :size 1024 :levels 5)"
        "  (threshold :mode soft :lambda 0.08)"
        "  (idwt :family cdf97 :size 1024 :levels 5))"
    );
    if (!t) {
        state.SkipWithError("Failed to compile denoise pipeline");
        return;
    }
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2 * i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
        in[2 * i + 1] = 0.0f;
    }
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    state.SetItemsProcessed(state.iterations() * n);
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Chirp_Execute_Denoise)->Unit(benchmark::kMicrosecond);
