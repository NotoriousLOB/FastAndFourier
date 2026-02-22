/**
 * @file benchmark_transforms.cpp
 * @brief Transform comparison benchmarks
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Compare FFT vs DCT performance */
static void BM_Compare_FFT(benchmark::State& state) {
    const size_t n = state.range(0);
    
    faf_transform* t = faf_create_fft(n, false, FAF_PREC_FP32, 0);
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
    
    faf_transform* t = faf_create_dct(n, 2, FAF_PREC_FP32, 0);
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
    
    faf_transform* t = faf_create_dst(n, 2, FAF_PREC_FP32, 0);
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
    
    faf_transform* t = faf_create_haar(n, 3, FAF_PREC_FP32, 0);
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
    
    faf_transform* t = faf_create_daubechies4(n, 3, FAF_PREC_FP32, 0);
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

BENCHMARK(BM_Compare_Haar)->Range(64, 1024);
BENCHMARK(BM_Compare_Daubechies4)->Range(64, 1024);
