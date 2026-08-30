/**
 * @file benchmark_fft.cpp
 * @brief FFT performance benchmarks
 */

#include <benchmark/benchmark.h>
#include "fastandfourier.h"
#include "chirp.h"
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

/* Mixed-radix 4000 vs padded-to-4096 vs native 4096. */
static void BM_FFT_Mixed(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    faf_transform *t = faf_create_fft(&c);
    if (!t) {
        state.SkipWithError("Failed to create mixed-radix fft");
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
    free(re_in); free(im_in); free(re_out); free(im_out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FFT_Mixed)
    ->Arg(3840)->Arg(4000)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

/* rfft_3840 split Hermitian vs interleaved — layout tax. */
static void BM_RFFT_Layout(benchmark::State& state) {
    const size_t n = 3840;
    const size_t nb = n / 2 + 1;
    const int interleaved = (int)state.range(0);
    faf_config c = faf_config_init(n);
    c.layout = interleaved ? FAF_LAYOUT_INTERLEAVED : FAF_LAYOUT_HERMITIAN;
    faf_transform *t = faf_create_rfft(&c);
    if (!t) {
        state.SkipWithError("Failed to create rfft");
        return;
    }
    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x, n);
    float *re = (float *)aligned_alloc(64, (interleaved ? 2 * nb : nb) * sizeof(float));
    float *im = interleaved ? nullptr
                            : (float *)aligned_alloc(64, nb * sizeof(float));
    faf_buffer out;
    if (interleaved)
        out = faf_buffer_interleaved(re, nb);
    else
        out = faf_buffer_hermitian(re, im, nb);
    for (auto _ : state) {
        faf_execute(t, &out, &in);
        benchmark::DoNotOptimize(re);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    free(x); free(re); if (im) free(im);
    faf_destroy_transform(t);
}
BENCHMARK(BM_RFFT_Layout)
    ->Arg(0)->Arg(1)
    ->Unit(benchmark::kMicrosecond);

/* Packed Hermitian corr vs unpack-to-full C2C conjugate multiply. */
static void BM_RFFT_CorrPacked(benchmark::State& state) {
    const size_t n = 1024;
    const size_t nb = n / 2 + 1;
    const int packed = (int)state.range(0);
    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *h = (float *)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        x[i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);
        h[i] = cosf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n);
    }
    if (packed) {
        faf_config c = faf_config_init(n);
        faf_transform *fwd = faf_create_rfft(&c);
        faf_transform *inv = faf_create_inverse(fwd);
        float *Xr = (float *)aligned_alloc(64, nb * sizeof(float));
        float *Xi = (float *)aligned_alloc(64, nb * sizeof(float));
        float *Hr = (float *)aligned_alloc(64, nb * sizeof(float));
        float *Hi = (float *)aligned_alloc(64, nb * sizeof(float));
        float *y = (float *)aligned_alloc(64, n * sizeof(float));
        faf_buffer hin = faf_buffer_real(h, n);
        faf_buffer Hs = faf_buffer_hermitian(Hr, Hi, nb);
        faf_execute(fwd, &Hs, &hin);
        faf_buffer xin = faf_buffer_real(x, n);
        faf_buffer Xs = faf_buffer_hermitian(Xr, Xi, nb);
        faf_buffer yb = faf_buffer_real(y, n);
        for (auto _ : state) {
            faf_execute(fwd, &Xs, &xin);
            faf_herm_mul_conj_f32(Xr, Xi, Xr, Xi, Hr, Hi, nb);
            faf_execute(inv, &yb, &Xs);
            benchmark::DoNotOptimize(y);
        }
        free(Xr); free(Xi); free(Hr); free(Hi); free(y);
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    } else {
        faf_config c = faf_config_init(n);
        c.layout = FAF_LAYOUT_SPLIT;
        faf_transform *fwd = faf_create_fft(&c);
        faf_transform *inv = faf_create_inverse(fwd);
        float *xr = (float *)aligned_alloc(64, n * sizeof(float));
        float *xi = (float *)aligned_alloc(64, n * sizeof(float));
        float *hr = (float *)aligned_alloc(64, n * sizeof(float));
        float *hi = (float *)aligned_alloc(64, n * sizeof(float));
        float *Xr = (float *)aligned_alloc(64, n * sizeof(float));
        float *Xi = (float *)aligned_alloc(64, n * sizeof(float));
        float *Hr = (float *)aligned_alloc(64, n * sizeof(float));
        float *Hi = (float *)aligned_alloc(64, n * sizeof(float));
        float *zr = (float *)aligned_alloc(64, n * sizeof(float));
        float *zi = (float *)aligned_alloc(64, n * sizeof(float));
        memcpy(xr, x, n * sizeof(float));
        memcpy(hr, h, n * sizeof(float));
        memset(xi, 0, n * sizeof(float));
        memset(hi, 0, n * sizeof(float));
        faf_buffer hin = faf_buffer_split(hr, hi, n);
        faf_buffer Hs = faf_buffer_split(Hr, Hi, n);
        faf_execute(fwd, &Hs, &hin);
        faf_buffer xin = faf_buffer_split(xr, xi, n);
        faf_buffer Xs = faf_buffer_split(Xr, Xi, n);
        faf_buffer Ys = faf_buffer_split(zr, zi, n);
        for (auto _ : state) {
            memcpy(xr, x, n * sizeof(float));
            memset(xi, 0, n * sizeof(float));
            faf_execute(fwd, &Xs, &xin);
            for (size_t k = 0; k < n; k++) {
                float ar = Xr[k], ai = Xi[k], br = Hr[k], bi = Hi[k];
                Xr[k] = ar * br + ai * bi;
                Xi[k] = ai * br - ar * bi;
            }
            faf_execute(inv, &Ys, &Xs);
            benchmark::DoNotOptimize(zr);
        }
        free(xr); free(xi); free(hr); free(hi);
        free(Xr); free(Xi); free(Hr); free(Hi); free(zr); free(zi);
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }
    free(x); free(h);
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
}
BENCHMARK(BM_RFFT_CorrPacked)
    ->Arg(1)->Arg(0)
    ->Unit(benchmark::kMicrosecond);

/* Bluestein n=307 vs pad-to-320 vs pad-to-512. */
static void BM_FFT_Bluestein(benchmark::State& state) {
    const size_t n = (size_t)state.range(0);
    const int bluestein = (int)state.range(1);
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_SPLIT;
    if (bluestein) c.flags = FAF_FLAG_BLUESTEIN;
    faf_transform *t = faf_create_fft(&c);
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
    free(re_in); free(im_in); free(re_out); free(im_out);
    faf_destroy_transform(t);
}
BENCHMARK(BM_FFT_Bluestein)
    ->Args({307, 1})
    ->Args({320, 0})
    ->Args({512, 0})
    ->Unit(benchmark::kMicrosecond);

/* Fused rfft→spectral→irfft vs unfused (milestone B is already in tree). */
static void ident_spec(float *re, float *im, size_t n_bins, void *ctx) {
    (void)re; (void)im; (void)n_bins; (void)ctx;
}
static void BM_Pipeline_Fused(benchmark::State& state) {
    const size_t n = 1024;
    chirp_register_spectral("ident_bench", ident_spec, nullptr);
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 1024) (spectral ident_bench) (irfft))");
    if (!t) {
        state.SkipWithError("compile failed");
        return;
    }
    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *y = (float *)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_real(y, n);
    for (auto _ : state) {
        faf_execute(t, &out, &in);
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    free(x); free(y);
    faf_destroy_transform(t);
}
static void BM_Pipeline_Unfused(benchmark::State& state) {
    const size_t n = 1024;
    const size_t nb = n / 2 + 1;
    faf_config c = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&c);
    faf_transform *inv = faf_create_inverse(fwd);
    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *y = (float *)aligned_alloc(64, n * sizeof(float));
    float *re = (float *)aligned_alloc(64, nb * sizeof(float));
    float *im = (float *)aligned_alloc(64, nb * sizeof(float));
    for (size_t i = 0; i < n; i++)
        x[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer spec = faf_buffer_hermitian(re, im, nb);
    faf_buffer out = faf_buffer_real(y, n);
    for (auto _ : state) {
        faf_execute(fwd, &spec, &in);
        ident_spec(re, im, nb, nullptr);
        faf_execute(inv, &out, &spec);
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    free(x); free(y); free(re); free(im);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}
BENCHMARK(BM_Pipeline_Fused)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Pipeline_Unfused)->Unit(benchmark::kMicrosecond);
