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
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate split-plane buffers */
    float* in_re = (float*)dspir_aligned_alloc(n * sizeof(float));
    float* in_im = (float*)dspir_aligned_alloc(n * sizeof(float));
    float* out_re = (float*)dspir_aligned_alloc(n * sizeof(float));
    float* out_im = (float*)dspir_aligned_alloc(n * sizeof(float));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in_im[i] = 0.0f;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        dspir_execute_split_f32(t, out_re, out_im, in_re, in_im);
    }
    
    for (auto _ : state) {
        dspir_execute_split_f32(t, out_re, out_im, in_re, in_im);
        benchmark::DoNotOptimize(out_re);
        benchmark::DoNotOptimize(out_im);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
    
    dspir_aligned_free(in_re);
    dspir_aligned_free(in_im);
    dspir_aligned_free(out_re);
    dspir_aligned_free(out_im);
    dspir_destroy_transform(t);
}

/* Benchmark standard interleaved FP32 */
static void BM_Standard_FP32(benchmark::State& state) {
    const size_t n = state.range(0);
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate interleaved buffers */
    float* in = (float*)dspir_aligned_alloc(2 * n * sizeof(float));
    float* out = (float*)dspir_aligned_alloc(2 * n * sizeof(float));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in[2*i + 1] = 0.0f;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        dspir_execute_f32(t, out, in);
    }
    
    for (auto _ : state) {
        dspir_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
    
    dspir_aligned_free(in);
    dspir_aligned_free(out);
    dspir_destroy_transform(t);
}

/* Benchmark split-plane FP64 */
static void BM_SplitPlane_FP64(benchmark::State& state) {
    const size_t n = state.range(0);
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP64, 0);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate split-plane buffers */
    double* in_re = (double*)dspir_aligned_alloc(n * sizeof(double));
    double* in_im = (double*)dspir_aligned_alloc(n * sizeof(double));
    double* out_re = (double*)dspir_aligned_alloc(n * sizeof(double));
    double* out_im = (double*)dspir_aligned_alloc(n * sizeof(double));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);
        in_im[i] = 0.0;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        dspir_execute_split_f64(t, out_re, out_im, in_re, in_im);
    }
    
    for (auto _ : state) {
        dspir_execute_split_f64(t, out_re, out_im, in_re, in_im);
        benchmark::DoNotOptimize(out_re);
        benchmark::DoNotOptimize(out_im);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(double));
    state.SetItemsProcessed(state.iterations() * n);
    
    dspir_aligned_free(in_re);
    dspir_aligned_free(in_im);
    dspir_aligned_free(out_re);
    dspir_aligned_free(out_im);
    dspir_destroy_transform(t);
}

/* Benchmark standard interleaved FP64 */
static void BM_Standard_FP64(benchmark::State& state) {
    const size_t n = state.range(0);
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP64, 0);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Allocate interleaved buffers */
    double* in = (double*)dspir_aligned_alloc(2 * n * sizeof(double));
    double* out = (double*)dspir_aligned_alloc(2 * n * sizeof(double));
    
    /* Initialize with sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * 4.0 * (double)i / (double)n);
        in[2*i + 1] = 0.0;
    }
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        dspir_execute_f64(t, out, in);
    }
    
    for (auto _ : state) {
        dspir_execute_f64(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetBytesProcessed(state.iterations() * 2 * n * sizeof(double));
    state.SetItemsProcessed(state.iterations() * n);
    
    dspir_aligned_free(in);
    dspir_aligned_free(out);
    dspir_destroy_transform(t);
}

BENCHMARK(BM_SplitPlane_FP32)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_Standard_FP32)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_SplitPlane_FP64)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_Standard_FP64)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
