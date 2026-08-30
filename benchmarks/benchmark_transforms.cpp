/**
 * @file benchmark_transforms.cpp
 * @brief Transform comparison benchmarks
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "bench_util.h"
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Compare FFT vs DCT performance */
static void BM_Compare_FFT(benchmark::State& state) {
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
    
    state.SetLabel("FFT");
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

static void BM_Compare_DCT(benchmark::State& state) {
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
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    state.SetLabel("DCT-II");
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

static void BM_Compare_DST(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_dst(n, 2);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* DST uses complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf((float)M_PI * (float)(i + 1) / (float)(n + 1));
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    state.SetLabel("DST-II");
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

BENCHMARK(BM_Compare_FFT)->Range(64, 1024);
BENCHMARK(BM_Compare_DCT)->Range(64, 1024);
BENCHMARK(BM_Compare_DST)->Range(64, 1024);

/* Compare wavelet transforms */
static void BM_Compare_Haar(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_haar(n, 3);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Wavelets use complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    state.SetLabel("Haar");
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

static void BM_Compare_Daubechies4(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = bench_d4(n, 3);
    if (!t) {
        state.SkipWithError("Failed to create transform");
        return;
    }
    
    /* Wavelets use complex-interleaved format */
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2*i+1] = 0.0f;
    }
    
    for (auto _ : state) {
        faf_execute_f32(t, out, in);
    }
    
    state.SetLabel("Daubechies-4");
    state.SetItemsProcessed(state.iterations() * n);
    
    free(in); free(out);
    faf_destroy_transform(t);
}

BENCHMARK(BM_Compare_Haar)->Range(64, 4096);
BENCHMARK(BM_Compare_Daubechies4)->Range(64, 4096);

static void BM_DWT_Family(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    const int fam_i = (int)state.range(1);
    const size_t levels = (state.range(2) == 0) ? 0 : (size_t)state.range(2);
    auto fam = (faf_wavelet_family)fam_i;
    const bool inverse = state.range(3) != 0;

    faf_transform* t = bench_dwt(fam, n, levels, inverse);
    if (!t) {
        state.SkipWithError("Failed to create DWT");
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
    state.SetItemsProcessed(state.iterations() * n);
    state.SetLabel(std::string(faf_wavelet_name(fam)) + (inverse ? "/idwt" : "/dwt"));

    free(in); free(out);
    faf_destroy_transform(t);
}

BENCHMARK(BM_DWT_Family)
    ->ArgsProduct({
        {64, 256, 1024, 4096},
        {0, 1, 2, 3, 4}, /* families */
        {3, 0},          /* levels: 3 or full */
        {0, 1}           /* forward / inverse */
    });

static void BM_DWT_Backend(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    const auto fam = (faf_wavelet_family)state.range(1);
    const auto backend = (faf_dwt_backend)state.range(2);

    faf_config c = faf_config_init(n);
    c.family = fam;
    c.levels = 0;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    c.dwt_backend = backend;
    faf_transform *t = faf_create_dwt(&c);
    if (!t) {
        state.SkipWithError("Failed to create DWT");
        return;
    }

    float *in = (float *)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float *)aligned_alloc(64, 2 * n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        in[2 * i] = sinf(2.0f * (float)M_PI * (float)i / (float)n);
        in[2 * i + 1] = 0.0f;
    }

    for (auto _ : state)
        faf_execute_f32(t, out, in);

    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    const char *bname = (backend == FAF_DWT_BACKEND_FIR) ? "F" : "L";
    state.SetLabel(std::string(faf_wavelet_name(fam)) + "/" + bname);

    free(in); free(out);
    faf_destroy_transform(t);
}

BENCHMARK(BM_DWT_Backend)
    ->ArgsProduct({
        {1024, 16384, 65536},
        {0, 1, 4}, /* haar, d4, sym4 */
        {1, 2}     /* LIFT, FIR */
    });
