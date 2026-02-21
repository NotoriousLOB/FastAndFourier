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
#include <cmath>
#include <cstring>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============ External Library Includes ============ */

// Kiss FFT
extern "C" {
#include <kissfft/kiss_fft.h>
}

// PocketFFT (header-only)
#define POCKETFFT_NO_MULTITHREADING
#include <pocketfft/pocketfft_hdronly.h>

// MinFFT
extern "C" {
#include <minfft/minfft.h>
}

// muFFT
#define MUFFT_API static inline
#include <mufft/mufft.c>

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
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        dspir_execute_f32(t, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    
    free(in); free(out);
    dspir_destroy_transform(t);
}
BENCHMARK(BM_FAF_FFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FastAndFourier/FFT_Real");

/* ============ Kiss FFT Benchmarks ============ */

static void BM_KissFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, NULL, NULL);
    if (!cfg) {
        state.SkipWithError("Failed to create Kiss FFT config");
        return;
    }
    
    kiss_fft_cpx* in = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    kiss_fft_cpx* out = (kiss_fft_cpx*)aligned_alloc(64, n * sizeof(kiss_fft_cpx));
    
    // Initialize complex input
    for (size_t i = 0; i < n; i++) {
        in[i].r = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[i].i = 0.0f;
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

static void BM_PocketFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    std::vector<std::complex<float>> in(n);
    std::vector<std::complex<float>> out(n);
    generate_signal_complex(in.data(), n);
    
    for (auto _ : state) {
        pocketfft::c2c(n, in, out, pocketfft::FORWARD);
        benchmark::DoNotOptimize(out.data());
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(std::complex<float>) * 2);
}
BENCHMARK(BM_PocketFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("PocketFFT/FFT_Real");

/* ============ MinFFT Benchmarks ============ */

static void BM_MinFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    // MinFFT uses its own complex type
    minfft_aux* aux = minfft_mkaux_dft_1d(n);
    if (!aux) {
        state.SkipWithError("Failed to create MinFFT aux");
        return;
    }
    
    minfft_real* in = (minfft_real*)aligned_alloc(64, 2 * n * sizeof(minfft_real));
    minfft_real* out = (minfft_real*)aligned_alloc(64, 2 * n * sizeof(minfft_real));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        minfft_dft(in, out, aux);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(minfft_real) * 4);
    
    free(in); free(out);
    minfft_free_aux(aux);
}
BENCHMARK(BM_MinFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("MinFFT/FFT_Real");

/* ============ muFFT Benchmarks ============ */

static void BM_muFFT_Real(benchmark::State& state) {
    const size_t n = state.range(0);
    
    mufft_plan_1d* plan = mufft_create_plan_1d_c2c(n, MUFFT_FORWARD, MUFFT_FLAG_CPU_ANY);
    if (!plan) {
        state.SkipWithError("Failed to create muFFT plan");
        return;
    }
    
    cfloat* in = (cfloat*)aligned_alloc(64, n * sizeof(cfloat));
    cfloat* out = (cfloat*)aligned_alloc(64, n * sizeof(cfloat));
    
    for (size_t i = 0; i < n; i++) {
        in[i].real = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[i].imag = 0.0f;
    }
    
    for (auto _ : state) {
        mufft_execute_plan_1d(plan, out, in);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(cfloat) * 2);
    
    free(in); free(out);
    mufft_free_plan_1d(plan);
}
BENCHMARK(BM_muFFT_Real)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("muFFT/FFT_Real");

/* ============ FFTW3 Benchmarks (if available) ============ */

#ifdef HAS_FFTW3

static void BM_FFTW3_Estimate(benchmark::State& state) {
    const size_t n = state.range(0);
    
    float* in = (float*)fftwf_malloc(n * sizeof(float));
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    
    generate_signal(in, n);
    
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    if (!plan) {
        state.SkipWithError("Failed to create FFTW3 plan");
        return;
    }
    
    for (auto _ : state) {
        fftwf_execute(plan);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}
BENCHMARK(BM_FFTW3_Estimate)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FFTW3/Estimate");

static void BM_FFTW3_Measure(benchmark::State& state) {
    const size_t n = state.range(0);
    
    float* in = (float*)fftwf_malloc(n * sizeof(float));
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    
    generate_signal(in, n);
    
    // Use wisdom if available, otherwise measure
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
    if (!plan) {
        state.SkipWithError("Failed to create FFTW3 plan");
        return;
    }
    
    for (auto _ : state) {
        fftwf_execute(plan);
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}
BENCHMARK(BM_FFTW3_Measure)
    ->RangeMultiplier(4)
    ->Range(16, 65536)
    ->Unit(benchmark::kMicrosecond)
    ->Name("FFTW3/Measure");

#endif // HAS_FFTW3

/* ============ Latency Comparison (Small Sizes) ============ */

static void BM_Latency_FAF(benchmark::State& state) {
    const size_t n = state.range(0);
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        dspir_execute_f32(t, out, in);
        benchmark::ClobberMemory();
    }
    
    free(in); free(out);
    dspir_destroy_transform(t);
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
        in[i].r = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[i].i = 0.0f;
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

/* ============ Throughput Comparison (Large Sizes) ============ */

static void BM_Throughput_FAF(benchmark::State& state) {
    const size_t n = state.range(0);
    const int batch = 100;
    
    dspir_transform* t = dspir_create_fft(n, false, DSPIR_PREC_FP32, 0);
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        for (int b = 0; b < batch; b++) {
            dspir_execute_f32(t, out, in);
        }
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * n);
    state.SetBytesProcessed(state.iterations() * batch * 2 * n * sizeof(float) * 2);
    
    free(in); free(out);
    dspir_destroy_transform(t);
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
    
    float* in = (float*)fftwf_malloc(n * sizeof(float));
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    
    generate_signal(in, n);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
    
    for (auto _ : state) {
        for (int b = 0; b < batch; b++) {
            fftwf_execute(plan);
        }
        benchmark::DoNotOptimize(out);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * n);
    state.SetBytesProcessed(state.iterations() * batch * n * sizeof(float) * 2);
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}
BENCHMARK(BM_Throughput_FFTW3)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Unit(benchmark::kMillisecond)
    ->Name("Throughput/FFTW3");
#endif

/* ============ Main ============ */
BENCHMARK_MAIN();
