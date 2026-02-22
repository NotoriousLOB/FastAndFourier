/**
 * @file jit_example.c
 * @brief JIT compilation example - High Performance Edition
 * 
 * Demonstrates maximum performance using JIT with:
 * - In-place transforms (no register allocation overhead)
 * - SIMD intrinsics (NEON on ARM64, AVX on x86)
 * - Pre-compiled kernels for repeated execution
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Get current time in microseconds */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Benchmark a kernel function */
static double benchmark_kernel(faf_kernel_fn kernel, void *out, const void *in, 
                                size_t n, void *twiddles, int iterations) {
    /* Warm up cache */
    for (int i = 0; i < 100; i++) {
        kernel(out, in, n, twiddles);
    }
    
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        kernel(out, in, n, twiddles);
    }
    return get_time_us() - start;
}

/* Benchmark VM execution */
static double benchmark_vm(faf_transform *t, float *out, const float *in, int iterations) {
    /* Warm up */
    for (int i = 0; i < 100; i++) {
        faf_execute_f32(t, out, in);
    }
    
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        faf_execute_f32(t, out, in);
    }
    return get_time_us() - start;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   FastAndFourier - High Performance JIT Compilation Demo     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    faf_init();
    
    const size_t n = 256;
    const int iterations = 10000;
    
    printf("Configuration:\n");
    printf("  Transform size: %zu-point complex FFT\n", n);
    printf("  Iterations: %d\n", iterations);
    printf("  Data format: Complex-interleaved (2*%zu floats)\n\n", n);
    
    /* Create transform */
    faf_transform* fft = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    if (!fft) {
        fprintf(stderr, "Failed to create FFT: %s\n", faf_get_error());
        return 1;
    }
    
    /* Allocate aligned memory for SIMD */
    float* in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float* out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    
    /* Initialize with complex sine wave */
    for (size_t i = 0; i < n; i++) {
        in[2*i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
        in[2*i + 1] = cosf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n);
    }
    
    printf("┌────────────────────────────────────────────────────────────┐\n");
    printf("│                    PERFORMANCE BENCHMARK                   │\n");
    printf("└────────────────────────────────────────────────────────────┘\n\n");
    
    /* Benchmark VM execution */
    printf("1. VM Execution (interpreted)\n");
    double vm_time = benchmark_vm(fft, out, in, iterations);
    double vm_per_fft = vm_time / iterations;
    double vm_throughput = iterations / (vm_time / 1e6);
    printf("   Total time:  %.2f ms\n", vm_time / 1000.0);
    printf("   Per FFT:     %.3f us\n", vm_per_fft);
    printf("   Throughput:  %.2f M FFTs/sec\n\n", vm_throughput / 1e6);
    
    /* Try JIT compilation with all optimizations */
    printf("2. JIT Compilation (in-place + SIMD)\n");
    
    faf_jit_ctx* jit = faf_jit_create();
    if (!jit) {
        fprintf(stderr, "   Failed to create JIT context\n");
        free(in); free(out);
        faf_destroy_transform(fft);
        return 1;
    }
    
    /* Compile with all optimizations enabled by default */
    if (faf_jit_compile(jit, fft) != 0) {
        printf("   JIT compilation not available (compiler not found)\n");
        printf("   Falling back to VM execution\n");
    } else {
        faf_kernel_fn kernel = faf_jit_get_kernel(jit);
        if (kernel) {
            double jit_time = benchmark_kernel(kernel, out, in, n, fft->twiddles[0], iterations);
            double jit_per_fft = jit_time / iterations;
            double jit_throughput = iterations / (jit_time / 1e6);
            double speedup = vm_time / jit_time;
            
            printf("   Total time:  %.2f ms\n", jit_time / 1000.0);
            printf("   Per FFT:     %.3f us\n", jit_per_fft);
            printf("   Throughput:  %.2f M FFTs/sec\n", jit_throughput / 1e6);
            printf("   Speedup:     %.1fx faster than VM\n\n", speedup);
            
            /* Benchmark in-place mode */
            printf("3. JIT In-Place Execution (overwrites input)\n");
            
            /* Copy input since in-place will overwrite */
            float* inplace_buffer = (float*)aligned_alloc(64, 2 * n * sizeof(float));
            memcpy(inplace_buffer, in, 2 * n * sizeof(float));
            
            /* Use in-place flag explicitly */
            faf_jit_ctx* jit_inplace = faf_jit_create();
            if (jit_inplace && faf_jit_compile_ex(jit_inplace, fft, 
                FAF_FLAG_JIT_INPLACE | FAF_FLAG_JIT_SIMD) == 0) {
                
                faf_kernel_fn kernel_ip = faf_jit_get_kernel(jit_inplace);
                
                /* Warm up */
                for (int i = 0; i < 100; i++) {
                    memcpy(inplace_buffer, in, 2 * n * sizeof(float));
                    kernel_ip(inplace_buffer, inplace_buffer, n, fft->twiddles[0]);
                }
                
                double ip_start = get_time_us();
                for (int i = 0; i < iterations; i++) {
                    memcpy(inplace_buffer, in, 2 * n * sizeof(float));
                    kernel_ip(inplace_buffer, inplace_buffer, n, fft->twiddles[0]);
                }
                double ip_time = get_time_us() - ip_start;
                double ip_per_fft = ip_time / iterations;
                double ip_throughput = iterations / (ip_time / 1e6);
                
                printf("   Total time:  %.2f ms (including memcpy)\n", ip_time / 1000.0);
                printf("   Per FFT:     %.3f us\n", ip_per_fft);
                printf("   Throughput:  %.2f M FFTs/sec\n", ip_throughput / 1e6);
                
                /* Pure transform time (no memcpy) */
                memcpy(inplace_buffer, in, 2 * n * sizeof(float));
                double pure_start = get_time_us();
                for (int i = 0; i < iterations; i++) {
                    kernel_ip(inplace_buffer, inplace_buffer, n, fft->twiddles[0]);
                }
                double pure_time = get_time_us() - pure_start;
                double pure_per_fft = pure_time / iterations;
                double pure_throughput = iterations / (pure_time / 1e6);
                
                printf("   Pure transform:\n");
                printf("     Per FFT:     %.3f us\n", pure_per_fft);
                printf("     Throughput:  %.2f M FFTs/sec\n", pure_throughput / 1e6);
                printf("     Speedup:     %.1fx faster than VM\n\n", vm_time / pure_time);
                
                faf_jit_destroy(jit_inplace);
            }
            free(inplace_buffer);
            
            /* Compare different transform sizes */
            printf("4. JIT Performance Across Sizes\n");
            printf("   Size    │ Time (us) │ Throughput (M FFTs/sec)\n");
            printf("   ────────┼───────────┼─────────────────────────\n");
            
            size_t sizes[] = {16, 32, 64, 128, 256};
            for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
                size_t test_n = sizes[i];
                faf_transform* test_fft = faf_create_fft(test_n, false, FAF_PREC_FP32, 0);
                if (!test_fft) continue;
                
                float* test_in = (float*)aligned_alloc(64, 2 * test_n * sizeof(float));
                float* test_out = (float*)aligned_alloc(64, 2 * test_n * sizeof(float));
                
                for (size_t j = 0; j < test_n; j++) {
                    test_in[2*j] = 1.0f;
                    test_in[2*j + 1] = 0.0f;
                }
                
                faf_jit_ctx* test_jit = faf_jit_create();
                if (test_jit && faf_jit_compile(test_jit, test_fft) == 0) {
                    faf_kernel_fn test_kernel = faf_jit_get_kernel(test_jit);
                    int test_iters = (test_n <= 64) ? 50000 : 10000;
                    
                    double t = benchmark_kernel(test_kernel, test_out, test_in, test_n, 
                                                test_fft->twiddles[0], test_iters);
                    double t_per = t / test_iters;
                    double t_thr = test_iters / (t / 1e6) / 1e6;
                    
                    printf("   %4zu    │  %7.3f  │      %7.2f\n", test_n, t_per, t_thr);
                }
                
                free(test_in); free(test_out);
                faf_jit_destroy(test_jit);
                faf_destroy_transform(test_fft);
            }
        }
    }
    
    printf("\n┌────────────────────────────────────────────────────────────┐\n");
    printf("│                         NOTES                              │\n");
    printf("└────────────────────────────────────────────────────────────┘\n");
    printf("• JIT compilation happens once, kernel is reused\n");
    printf("• In-place mode eliminates register allocation\n");
    printf("• SIMD uses NEON on ARM64, AVX/AVX2 on x86_64\n");
    printf("• First call includes compilation overhead (~1-2s)\n");
    printf("• Use aligned_alloc(64, ...) for best performance\n\n");
    
    /* Cleanup */
    free(in);
    free(out);
    faf_jit_destroy(jit);
    faf_destroy_transform(fft);
    faf_cleanup();
    
    return 0;
}
