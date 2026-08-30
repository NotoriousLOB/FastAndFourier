/**
 * @file benchmark_cwt.cpp
 * @brief CWT filter bank performance benchmarks
 */

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstring>
#include <cstdlib>

extern "C" {
#include "fastandfourier.h"
}

static void *alloc64(size_t bytes) {
    return aligned_alloc(64, ((bytes + 63) / 64) * 64);
}

static void BM_CWT_Forward_FP32(benchmark::State &state) {
    size_t n = (size_t)state.range(0);

    faf_cwt_config cfg = faf_cwt_config_init(n);
    cfg.precision = FAF_PREC_FP32;
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.voices = 10;


    faf_transform *t = faf_create_cwt(&cfg);
    if (!t) {
        state.SkipWithError("Failed to create CWT");
        return;
    }

    size_t n_rows = faf_cwt_n_rows(t);
    float *in_data = (float *)alloc64(n * sizeof(float));
    float *out_data = (float *)alloc64(n_rows * n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        in_data[i] = sinf(2.0f * 3.14159265f * 10.0f * (float)i / (float)n);

    faf_buffer in_buf = faf_buffer_real(in_data, n);
    faf_buffer out_buf;
    memset(&out_buf, 0, sizeof(out_buf));
    out_buf.re = out_data;
    out_buf.n = n;
    out_buf.layout = FAF_LAYOUT_REAL;

    for (auto _ : state) {
        faf_execute(t, &out_buf, &in_buf);
        benchmark::DoNotOptimize(out_data);
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)n);
    state.counters["scales"] = (double)faf_cwt_n_scales(t);
    state.counters["ns/sample"] = benchmark::Counter(
        (double)n, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

    free(in_data);
    free(out_data);
    faf_destroy_transform(t);
}
BENCHMARK(BM_CWT_Forward_FP32)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)
    ->Unit(benchmark::kMicrosecond);

static void BM_CWT_Forward_FP64(benchmark::State &state) {
    size_t n = (size_t)state.range(0);

    faf_cwt_config cfg = faf_cwt_config_init(n);
    cfg.precision = FAF_PREC_FP64;
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.voices = 10;


    faf_transform *t = faf_create_cwt(&cfg);
    if (!t) {
        state.SkipWithError("Failed to create CWT");
        return;
    }

    size_t n_rows = faf_cwt_n_rows(t);
    double *in_data = (double *)alloc64(n * sizeof(double));
    double *out_data = (double *)alloc64(n_rows * n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        in_data[i] = sin(2.0 * 3.14159265358979 * 10.0 * (double)i / (double)n);

    faf_buffer in_buf = faf_buffer_real(in_data, n);
    faf_buffer out_buf;
    memset(&out_buf, 0, sizeof(out_buf));
    out_buf.re = out_data;
    out_buf.n = n;
    out_buf.layout = FAF_LAYOUT_REAL;

    for (auto _ : state) {
        faf_execute(t, &out_buf, &in_buf);
        benchmark::DoNotOptimize(out_data);
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)n);
    state.counters["scales"] = (double)faf_cwt_n_scales(t);

    free(in_data);
    free(out_data);
    faf_destroy_transform(t);
}
BENCHMARK(BM_CWT_Forward_FP64)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)
    ->Unit(benchmark::kMicrosecond);

static void BM_ICWT_Dual_FP32(benchmark::State &state) {
    size_t n = (size_t)state.range(0);

    faf_cwt_config cfg = faf_cwt_config_init(n);
    cfg.precision = FAF_PREC_FP32;
    cfg.norm = FAF_CWT_NORM_L2;


    faf_transform *cwt = faf_create_cwt(&cfg);
    if (!cwt) {
        state.SkipWithError("Failed to create CWT");
        return;
    }
    faf_transform *icwt = faf_create_inverse(cwt);
    if (!icwt) {
        state.SkipWithError("Failed to create ICWT");
        faf_destroy_transform(cwt);
        return;
    }

    size_t n_rows = faf_cwt_n_rows(cwt);
    float *in_data = (float *)alloc64(n * sizeof(float));
    float *w_data = (float *)alloc64(n_rows * n * sizeof(float));
    float *out_data = (float *)alloc64(n * sizeof(float));

    for (size_t i = 0; i < n; i++)
        in_data[i] = sinf(2.0f * 3.14159265f * 10.0f * (float)i / (float)n);

    faf_buffer in_buf = faf_buffer_real(in_data, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = w_data;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;
    faf_execute(cwt, &w_buf, &in_buf);

    faf_buffer out_buf = faf_buffer_real(out_data, n);

    for (auto _ : state) {
        faf_execute(icwt, &out_buf, &w_buf);
        benchmark::DoNotOptimize(out_data);
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)n);
    state.counters["scales"] = (double)faf_cwt_n_scales(cwt);

    free(in_data);
    free(w_data);
    free(out_data);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
}
BENCHMARK(BM_ICWT_Dual_FP32)
    ->Arg(1024)->Arg(4096)->Arg(16384)
    ->Unit(benchmark::kMicrosecond);

static void BM_CWT_BankCreate(benchmark::State &state) {
    size_t n = (size_t)state.range(0);

    for (auto _ : state) {
        faf_cwt_config cfg = faf_cwt_config_init(n);
        cfg.precision = FAF_PREC_FP32;
        cfg.voices = 10;
        faf_transform *t = faf_create_cwt(&cfg);
        benchmark::DoNotOptimize(t);
        faf_destroy_transform(t);
    }

    state.SetItemsProcessed((int64_t)state.iterations());
}
BENCHMARK(BM_CWT_BankCreate)
    ->Arg(1024)->Arg(4096)->Arg(16384)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
