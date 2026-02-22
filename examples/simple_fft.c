/**
 * @file simple_fft.c
 * @brief Simple FFT example
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - Simple FFT Example\n");
    printf("====================================\n\n");
    
    /* Initialize library */
    faf_init();
    printf("Library version: %s\n", faf_version());
    printf("Architecture: %s\n\n", faf_arch_name());
    
    /* Create a 256-point FFT */
    const size_t n = 256;
    faf_transform* fft = faf_create_fft(n, false, FAF_PREC_FP32, 0);
    if (!fft) {
        fprintf(stderr, "Failed to create FFT: %s\n", faf_get_error());
        return 1;
    }
    
    /* Allocate aligned buffers */
    float* in = (float*)aligned_alloc(64, n * sizeof(float));
    float* out = (float*)aligned_alloc(64, n * sizeof(float));
    
    /* Create a signal with two frequency components */
    const int f1 = 4;   /* 4 cycles */
    const int f2 = 16;  /* 16 cycles */
    
    for (size_t i = 0; i < n; i++) {
        in[i] = sinf(2.0f * (float)M_PI * f1 * (float)i / (float)n) +
                0.5f * sinf(2.0f * (float)M_PI * f2 * (float)i / (float)n);
    }
    
    printf("Input signal: sin(2*pi*%d*t) + 0.5*sin(2*pi*%d*t)\n", f1, f2);
    
    /* Execute FFT */
    int result = faf_execute_f32(fft, out, in);
    if (result != 0) {
        fprintf(stderr, "FFT execution failed\n");
        return 1;
    }
    
    /* Find peaks in spectrum */
    printf("\nFrequency spectrum (first 32 bins):\n");
    printf("Bin\tMagnitude\n");
    printf("---\t---------\n");
    
    for (size_t i = 0; i < 32 && i < n; i++) {
        float mag = fabsf(out[i]);
        if (mag > 0.1f) {
            printf("%zu\t%.2f%s\n", i, mag, 
                   (i == (size_t)f1 || i == (size_t)f2) ? " *" : "");
        }
    }
    
    printf("\n* Expected peaks at bins %d and %d\n", f1, f2);
    
    /* Cleanup */
    free(in);
    free(out);
    faf_destroy_transform(fft);
    faf_cleanup();
    
    return 0;
}
