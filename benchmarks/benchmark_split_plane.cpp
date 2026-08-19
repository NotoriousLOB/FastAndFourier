/**
 * @file benchmark_split_plane.cpp
 * @brief Benchmark split-plane FP32 vs FP64 performance
 */

#include <benchmark/benchmark.h>
#include <fastandfourier.h>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Benchmark split-plane FP32 */
static void BM_SplitPlane_FP32(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate split-plane buffers */
    float* in_re = (float*)faf_aligned_alloc(n * sizeof(float));
    float* in_im = (float*)faf_aligned_alloc(n * sizeof(float));
    float* out_re = (float*)faf_aligned_alloc(n * sizeof(float));
    float* out_im = (float*)faf_aligned_alloc(n * sizeof(float));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in_im[i] = 0.0f;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        faf_execute_split_f32(t, out_re, out_im, in_re, in_im);
    }
    
    for (auto _ : state) {
        faf_execute_split_f32(t, out_re, out_im, in_re, in_im);
        benchmark::DoNotOptimize(out_re);
        benchmark::DoNotOptimize(out_im);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
    
    faf_aligned_free(in_re);
    faf_aligned_free(in_im);
    faf_aligned_free(out_re);
    faf_aligned_free(out_im);
    faf_destroy_transform(t);
}

/* Benchmark standard interleaved FP32 */
static void BM_Standard_FP32(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate interleaved buffers */
    float* in = (float*)faf_aligned_alloc(2 * n * sizeof(float));
    float* out = (float*)faf_aligned_alloc(2 * n * sizeof(float));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in[2*i + 1] = 0.0f;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        faf_execute_f32(t, out, in);
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
    
    faf_aligned_free(in);
    faf_aligned_free(out);
    faf_destroy_transform(t);
}

/* Benchmark split-plane FP64 */
static void BM_SplitPlane_FP64(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate split-plane buffers */
    double* in_re = (double*)faf_aligned_alloc(n * sizeof(double));
    double* in_im = (double*)faf_aligned_alloc(n * sizeof(double));
    double* out_re = (double*)faf_aligned_alloc(n * sizeof(double));
    double* out_im = (double*)faf_aligned_alloc(n * sizeof(double));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);
        in_im[i] = 0.0;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        faf_execute_split_f64(t, out_re, out_im, in_re, in_im);
    }
    
    for (auto _ : state) {
        faf_execute_split_f64(t, out_re, out_im, in_re, in_im);
        benchmark::DoNotOptimize(out_re);
        benchmark::DoNotOptimize(out_im);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(double));
    state.SetItemsProcessed(state.iterations() * n);
    
    faf_aligned_free(in_re);
    faf_aligned_free(in_im);
    faf_aligned_free(out_re);
    faf_aligned_free(out_im);
    faf_destroy_transform(t);
}

/* Benchmark standard interleaved FP64 */
static void BM_Standard_FP64(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate interleaved buffers */
    double* in = (double*)faf_aligned_alloc(2 * n * sizeof(double));
    double* out = (double*)faf_aligned_alloc(2 * n * sizeof(double));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);
        in[2*i + 1] = 0.0;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        faf_execute_f64(t, out, in);
    }
    
    for (auto _ : state) {
        faf_execute_f64(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(double));
    state.SetItemsProcessed(state.iterations() * n);
    
    faf_aligned_free(in);
    faf_aligned_free(out);
    faf_destroy_transform(t);
}

BENCHMARK(BM_SplitPlane_FP32)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_Standard_FP32)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_SplitPlane_FP64)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_Standard_FP64)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
