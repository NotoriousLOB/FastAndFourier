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
    
    /* Create a 256-point FFT. Default layout is split re[]/im[]. */
    const size_t n = 256;
    faf_config cfg = faf_config_init(n);
    faf_transform* fft = faf_create_fft(&cfg);
    if (!fft) {
        fprintf(stderr, "Failed to create FFT: %s\n", faf_get_error());
        return 1;
    }
    printf("layout=%s  norm=%s\n\n",
           faf_layout_name(fft->cfg.layout), faf_norm_name(fft->cfg.norm));

    float* in_re  = (float*)aligned_alloc(64, n * sizeof(float));
    float* in_im  = (float*)aligned_alloc(64, n * sizeof(float));
    float* out_re = (float*)aligned_alloc(64, n * sizeof(float));
    float* out_im = (float*)aligned_alloc(64, n * sizeof(float));
    if (!in_re || !in_im || !out_re || !out_im) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    const int f1 = 4;
    const int f2 = 16;
    for (size_t i = 0; i < n; i++) {
        in_re[i] = sinf(2.0f * (float)M_PI * f1 * (float)i / (float)n) +
                   0.5f * sinf(2.0f * (float)M_PI * f2 * (float)i / (float)n);
        in_im[i] = 0.0f;
    }

    printf("Input signal: sin(2*pi*%d*t) + 0.5*sin(2*pi*%d*t)\n", f1, f2);

    faf_buffer in  = faf_buffer_split(in_re, in_im, n);
    faf_buffer out = faf_buffer_split(out_re, out_im, n);
    if (faf_execute(fft, &out, &in) != 0) {
        fprintf(stderr, "FFT execution failed: %s\n", faf_get_error());
        return 1;
    }

    printf("\nFrequency spectrum (first 32 bins):\n");
    printf("Bin\tMagnitude\n");
    printf("---\t---------\n");

    for (size_t i = 0; i < 32 && i < n; i++) {
        float mag = hypotf(out_re[i], out_im[i]);
        if (mag > 0.1f) {
            printf("%zu\t%.2f%s\n", i, mag,
                   (i == (size_t)f1 || i == (size_t)f2) ? " *" : "");
        }
    }

    printf("\n* Expected peaks at bins %d and %d\n", f1, f2);

    free(in_re); free(in_im); free(out_re); free(out_im);
    faf_destroy_transform(fft);
    faf_cleanup();
    
    return 0;
}
