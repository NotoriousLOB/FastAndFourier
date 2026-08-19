/**
 * @file benchmark_fft.cpp
 * @brief FFT performance benchmarks
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "bench_util.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Benchmark FFT at various sizes */
static void BM_FFT_Vm_Small(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Initialize with complex sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in[2*i + 1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FFT_Vm_Small)
    ->Range(16, 128)  /* Max 128 due to register file limit */
    ->Unit(benchmark::kMicrosecond);

/* Benchmark double precision FFT */
static void BM_FFT_Double(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    double *in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double *out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);
        in[2*i + 1] = 0.0;
    }
    
    for (auto _ : state) {
        faf_execute_f64(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FFT_Double)
    ->Range(16, 128)  /* Max 128 due to register file limit */
    ->Unit(benchmark::kMicrosecond);

/* Benchmark DCT */
static void BM_DCT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_dct(n, 2);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* DCT uses complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (float)i / (float)n;
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_DCT)
    ->Range(16, 128)
    ->Unit(benchmark::kMicrosecond);

/* Benchmark Haar wavelet */
static void BM_Haar(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_haar(n, 3);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Haar uses complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Haar)
    ->Range(16, 128)
    ->Unit(benchmark::kMicrosecond);

/* Benchmark MDCT */
static void BM_MDCT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_mdct(n);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* MDCT uses complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in);
    free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_MDCT)
    ->Range(16, 128)
    ->Unit(benchmark::kMicrosecond);

/* Benchmark JIT-compiled FFT */
static void BM_FFT_JIT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Create JIT context and compile */
    faf_jit_ctx* jit = faf_jit_create();
    if (!jit) {
        state.SkipWithError("Failed to create JIT context");
        faf_destroy_transform(t);
        return;
    }
    
    if (faf_jit_compile(jit, t) != 0) {
        state.SkipWithError("JIT compilation failed");
        faf_jit_destroy(jit);
        faf_destroy_transform(t);
        return;
    }
    
    faf_kernel_fn kernel = faf_jit_get_kernel(jit);
    if (!kernel) {
        state.SkipWithError("Failed to get JIT kernel");
        faf_jit_destroy(jit);
        faf_destroy_transform(t);
        return;
    }
    
    /* Complex data format: 2*n floats */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        kernel(out, in, n, t->twiddles[0]);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in);
    free(out);
    faf_jit_destroy(jit);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FFT_JIT)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Arg(8192)
    ->Arg(16384)
    ->Unit(benchmark::kMicrosecond);
