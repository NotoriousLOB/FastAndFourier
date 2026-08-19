/**
 * @file benchmark_comparison.cpp
 * @brief Comprehensive benchmark comparing FastAndFourier against popular FFT libraries
 * 
 * Libraries compared:
 * - FFTW3 (industry standard)
 * - Kiss FFT (simple, portable)
 * - PocketFFT (header-only, modern C++)
 * - MinFFT (minimalist)
 * - muFFT (single header)
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "bench_util.h"
#include <cmath>
#include <cstring>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============ External Library Includes ============ */

// Kiss FFT
extern "C" {
#include <kiss_fft.h>
}

// PocketFFT - disabled (C implementation has build issues)
// extern "C" {
// #include <pocketfft.h>
// }

// MinFFT - disabled (API mismatch)
// extern "C" {
// #include <minfft.h>
// }

// muFFT - disabled (API mismatch)
// extern "C" {
// #include <fft.h>
// }

// Notorious FFT — enabled (bugs fixed in latest version)
// Uses double precision (default)
#define NOTORIOUS_FFT_IMPLEMENTATION
#include <notorious_fft.h>

// FFTW3 (optional)
#ifdef HAS_FFTW3
#include <fftw3.h>
#endif

/* ============ Helper Functions ============ */

// Generate test signal
static void generate_signal(float* out, size_t n, int num_freqs = 3) {
    memset(out, 0, n * sizeof(float));
    for (int f = 1; f <= num_freqs; f++) {
        for (size_t i = 0; i < n; i++) {
            out[i] += sinf(2.0f * (float)M_PI * f * (float)i / (float)n) / f;
        }
    }
}

static void generate_signal_complex(std::complex<float>* out, size_t n, int num_freqs = 3) {
    for (size_t i = 0; i < n; i++) {
        out[i] = std::complex<float>(0.0f, 0.0f);
    }
    for (int f = 1; f <= num_freqs; f++) {
        for (size_t i = 0; i < n; i++) {
            float val = sinf(2.0f * (float)M_PI * f * (float)i / (float)n) / f;
            out[i] += std::complex<float>(val, 0.0f);
        }
    }
}

/* ============ FastAndFourier Benchmarks ============ */

static void BM_FAF_FFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Use double precision for fair comparison
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    double* in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double* out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    for (auto _ : state) {
        faf_execute_f64(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FAF_FFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FastAndFourier/FFT_Real");

/* ============ Kiss FFT Benchmarks ============ */

static void BM_KissFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Use double precision (kiss_fft_scalar is double by default)
    kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, NULL, NULL);
    if (!cfg) {
        state.SkipWithError("Failed to create Kiss FFT config");
        return;
    }
    
    kiss_fft_cpx* in = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    kiss_fft_cpx* out = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    
    // Initialize complex input (double precision)
    for (size_t i = 0; i < n; i++) {
        in[i].r = sin(2.0 * M_PI * (double)i / (double)n);
        in[i].i = 0.0;
    }
    
    for (auto _ : state) {
        kiss_fft(cfg, in, out);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(kiss_fft_cpx) * 2);
    
    free(in); free(out);
    free(cfg);
}
BENCHMARK(BM_KissFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("KissFFT/FFT_Real");

/* ============ PocketFFT Benchmarks ============ */
// Disabled - C implementation has build issues with current source

/*
static void BM_PocketFFT_Real(benchmark::State& state) {
    state.SkipWithError("PocketFFT disabled");
}
BENCHMARK(BM_PocketFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("PocketFFT/FFT_Real");
*/

/* ============ MinFFT Benchmarks ============ */
// Disabled - API mismatch with current MinFFT version

/* ============ muFFT Benchmarks ============ */
// Disabled - API mismatch with current muFFT version

/* ============ FFTW3 Benchmarks (if available) ============ */

#ifdef HAS_FFTW3

static void BM_FFTW3_Estimate(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Use complex-to-complex (c2c) with double precision
    fftw_complex* in = (fftw_complex*)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex* out = (fftw_complex*)fftw_malloc(n * sizeof(fftw_complex));
    
    for (size_t i = 0; i < n; i++) {
        in[i][0] = sin(2.0 * M_PI * (double)i / (double)n);
        in[i][1] = 0.0;
    }
    
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) {
        state.SkipWithError("Failed to create FFTW3 plan");
        return;
    }
    
    for (auto _ : state) {
        fftw_execute(plan);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(fftw_complex) * 2);
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}
BENCHMARK(BM_FFTW3_Estimate)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FFTW3/Estimate");

static void BM_FFTW3_Measure(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // Use complex-to-complex (c2c) with double precision
    fftw_complex* in = (fftw_complex*)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex* out = (fftw_complex*)fftw_malloc(n * sizeof(fftw_complex));
    
    for (size_t i = 0; i < n; i++) {
        in[i][0] = sin(2.0 * M_PI * (double)i / (double)n);
        in[i][1] = 0.0;
    }
    
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
    if (!plan) {
        state.SkipWithError("Failed to create FFTW3 plan");
        return;
    }
    
    for (auto _ : state) {
        fftw_execute(plan);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(fftw_complex) * 2);
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}
BENCHMARK(BM_FFTW3_Measure)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FFTW3/Measure");

#endif // HAS_FFTW3

/* ============ Notorious FFT Benchmarks ============ */

static void BM_NotoriousFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // NotoriousFFT uses a plan-based API (double precision)
    notorious_fft_plan* plan = notorious_fft_create_plan(n, 0);
    if (!plan) {
        state.SkipWithError("Failed to create NotoriousFFT plan");
        return;
    }
    
    double* in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double* out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    
    // Initialize complex input (interleaved real/imag)
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    for (auto _ : state) {
        notorious_fft_execute_cx(plan, in, out, 0);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2 * 2);
    
    free(in); free(out);
    notorious_fft_destroy_plan(plan);
}
BENCHMARK(BM_NotoriousFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("NotoriousFFT/FFT_Real");

/* ============ Latency Comparison (Small Sizes) ============ */

static void BM_Latency_FAF(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    double* in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double* out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    for (auto _ : state) {
        faf_execute_f64(t, out, in);
        benchmark::ClobberMemory();
    }
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Latency_FAF)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kNanosecond)
    ->Name("Latency/FastAndFourier");

static void BM_Latency_KissFFT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, NULL, NULL);
    kiss_fft_cpx* in = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    kiss_fft_cpx* out = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    
    for (size_t i = 0; i < n; i++) {
        in[i].r = sin(2.0 * M_PI * (double)i / (double)n);
        in[i].i = 0.0;
    }
    
    for (auto _ : state) {
        kiss_fft(cfg, in, out);
        benchmark::ClobberMemory();
    }
    
    free(in); free(out);
    free(cfg);
}
BENCHMARK(BM_Latency_KissFFT)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kNanosecond)
    ->Name("Latency/KissFFT");

static void BM_Latency_NotoriousFFT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    notorious_fft_plan* plan = notorious_fft_create_plan(n, 0);
    double* in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double* out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    for (auto _ : state) {
        notorious_fft_execute_cx(plan, in, out, 0);
        benchmark::ClobberMemory();
    }
    
    free(in); free(out);
    notorious_fft_destroy_plan(plan);
}
BENCHMARK(BM_Latency_NotoriousFFT)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kNanosecond)
    ->Name("Latency/NotoriousFFT");

/* ============ Throughput Comparison (Large Sizes) ============ */

static void BM_Throughput_FAF(benchmark::State& state) {
    const size_t n = state.range(0);
    const int batch = 100;
    
    faf_transform* t = bench_fft(n, FAF_PREC_FP64);
    double* in = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    double* out = (double*)aligned_alloc(64, 2 * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sin(2.0 * M_PI * (double)i / (double)n);
        in[2*i+1] = 0.0;
    }
    
    for (auto _ : state) {
        for (int b = 0; b < batch; b++) {
            faf_execute_f64(t, out, in);
        }
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * n);
    state.SetBytesProcessed(state.iterations() * batch * 2 * n * sizeof(double) * 2);
    
    free(in); free(out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_Throughput_FAF)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Unit(benchmark::kMillisecond)
    ->Name("Throughput/FastAndFourier");

#ifdef HAS_FFTW3
static void BM_Throughput_FFTW3(benchmark::State& state) {
    const size_t n = state.range(0);
    const int batch = 100;
    
    double* in = (double*)fftw_malloc(n * sizeof(double));
    fftw_complex* out = (fftw_complex*)fftw_malloc(n * sizeof(fftw_complex));
    
    for (size_t i = 0; i < n; i++) {
        in[i] = sin(2.0 * M_PI * (double)i / (double)n);
    }
    fftw_plan plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
    
    for (auto _ : state) {
        for (int b = 0; b < batch; b++) {
            fftw_execute(plan);
        }
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * n);
    state.SetBytesProcessed(state.iterations() * batch * n * sizeof(double) * 2);
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}
BENCHMARK(BM_Throughput_FFTW3)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Unit(benchmark::kMillisecond)
    ->Name("Throughput/FFTW3");
#endif

/* Benchmark registration is automatic; main is provided by benchmark::benchmark_main */
