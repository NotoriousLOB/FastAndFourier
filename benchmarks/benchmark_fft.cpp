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

/* Packed R2C vs C2C of a real signal with a zeroed imag plane. */
static void BM_RFFT(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    const size_t nb = n / 2 + 1;
    faf_transform *t = bench_rfft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create rfft");
        return;
    }
    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *re = (float *)aligned_alloc(64, nb * sizeof(float));
    float *im = (float *)aligned_alloc(64, nb * sizeof(float));
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_hermitian(re, im, nb);
    for (auto _ : state) {
        faf_execute(t, &out, &in);
        benchmark::DoNotOptimize(re);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.SetBytesProcessed(state.iterations() * (int64_t)n * sizeof(float));
    free(x); free(re); free(im);
    faf_destroy_transform(t);
}
BENCHMARK(BM_RFFT)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_C2C_ZeroImag(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    faf_transform *t = bench_fft_split(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create fft");
        return;
    }
    float *re_in = (float *)aligned_alloc(64, n * sizeof(float));
    float *im_in = (float *)aligned_alloc(64, n * sizeof(float));
    float *re_out = (float *)aligned_alloc(64, n * sizeof(float));
    float *im_out = (float *)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        re_in[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        im_in[i] = 0.0f;
    }
    faf_buffer in = faf_buffer_split(re_in, im_in, n);
    faf_buffer out = faf_buffer_split(re_out, im_out, n);
    for (auto _ : state) {
        faf_execute(t, &out, &in);
        benchmark::DoNotOptimize(re_out);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.SetBytesProcessed(state.iterations() * (int64_t)n * 2 * sizeof(float));
    free(re_in); free(im_in); free(re_out); free(im_out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_C2C_ZeroImag)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);
