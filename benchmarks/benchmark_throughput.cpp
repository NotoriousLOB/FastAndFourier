/**
 * @file benchmark_throughput.cpp
 * @brief Throughput and scaling benchmarks
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "bench_util.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Throughput benchmark: many small FFTs */
static void BM_Throughput_SmallFFTs(benchmark::State& state) {
    const size_t n = 64;
    const size_t batch = 1000;
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        for (size_t b = 0; b < batch; b++) {
            faf_execute_f32(t, out, in);
            benchmark::DoNotOptimize(out);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * batch * n);
    state.SetBytesProcessed(state.iterations() * batch * n * sizeof(float) * 2);
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Throughput_SmallFFTs)
    ->Unit(benchmark::kMillisecond);

/* Scaling benchmark: how performance scales with size */
static void BM_Scaling_FFT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    /* Set complexity N for auto-complexity calculation */
    state.SetComplexityN(n);
    
    /* Report GFLOPS (5*N*log2(N) operations per FFT) */
    size_t ops = 5 * n * (size_t)log2(n);
    state.counters["GFLOPS"] = benchmark::Counter(
        state.iterations() * ops, benchmark::Counter::kIsRate);
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Scaling_FFT)
    ->RangeMultiplier(2)
    ->Range(16, 8192)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity(benchmark::oNLogN);

/* Memory bandwidth benchmark */
static void BM_MemoryBandwidth(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (float)i;
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    /* Report memory bandwidth */
    size_t bytes = 2 * n * sizeof(float) * 2;  /* read + write */
    state.SetBytesProcessed(state.iterations() * bytes);
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_MemoryBandwidth)
    ->RangeMultiplier(4)
    ->Range(1024, 1048576)
    ->Unit(benchmark::kMillisecond);

/* Latency benchmark: single FFT latency */
static void BM_Latency_FFT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP32);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = (float)i;
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations());
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Latency_FFT)
    ->RangeMultiplier(2)
    ->Range(16, 128)
    ->Unit(benchmark::kNanosecond);
