/**
 * @file simple_rfft.c
 * @brief Real FFT example: REAL time -> packed Hermitian spectrum -> round-trip
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - Simple R2C Example\n");
    printf("====================================\n\n");

    faf_init();
    printf("Library version: %s\n", faf_version());
    printf("Architecture: %s\n\n", faf_arch_name());

    const size_t n = 256;
    const size_t nb = n / 2 + 1;
    faf_config cfg = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&cfg);
    if (!fwd) {
        fprintf(stderr, "Failed to create rfft: %s\n", faf_get_error());
        return 1;
    }
    faf_transform *inv = faf_create_inverse(fwd);
    if (!inv) {
        fprintf(stderr, "Failed to create irfft: %s\n", faf_get_error());
        faf_destroy_transform(fwd);
        return 1;
    }

    printf("n=%zu  spectrum_len=%zu  layout=%s  norm=%s\n\n",
           n, faf_spectrum_len(fwd),
           faf_layout_name(fwd->cfg.layout), faf_norm_name(fwd->cfg.norm));

    float *x  = (float *)aligned_alloc(64, n * sizeof(float));
    float *y  = (float *)aligned_alloc(64, n * sizeof(float));
    float *re = (float *)aligned_alloc(64, nb * sizeof(float));
    float *im = (float *)aligned_alloc(64, nb * sizeof(float));
    if (!x || !y || !re || !im) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    const int f1 = 4;
    const int f2 = 16;
    for (size_t i = 0; i < n; i++) {
        x[i] = sinf(2.0f * (float)M_PI * (float)f1 * (float)i / (float)n) +
               0.5f * sinf(2.0f * (float)M_PI * (float)f2 * (float)i / (float)n);
    }

    printf("Input: sin(2*pi*%d*t) + 0.5*sin(2*pi*%d*t)  (real)\n", f1, f2);

    faf_buffer in  = faf_buffer_real(x, n);
    faf_buffer spec = faf_buffer_hermitian(re, im, nb);
    faf_buffer out = faf_buffer_real(y, n);
    if (faf_execute(fwd, &spec, &in) != 0) {
        fprintf(stderr, "rfft failed: %s\n", faf_get_error());
        return 1;
    }

    printf("\nPacked Hermitian spectrum (bins with |X| > 0.1):\n");
    printf("Bin\tRe\t\tIm\t\tMagnitude\n");
    printf("---\t--\t\t--\t\t---------\n");
    for (size_t k = 0; k < nb; k++) {
        float mag = hypotf(re[k], im[k]);
        if (mag > 0.1f) {
            printf("%zu\t%8.2f\t%8.2f\t%8.2f%s\n", k, re[k], im[k], mag,
                   (k == (size_t)f1 || k == (size_t)f2) ? " *" : "");
        }
    }
    printf("\nDC imag=%.3g  Nyquist imag=%.3g  (both should be ~0)\n",
           im[0], im[nb - 1]);
    printf("* Expected peaks at bins %d and %d\n\n", f1, f2);

    if (faf_execute(inv, &out, &spec) != 0) {
        fprintf(stderr, "irfft failed: %s\n", faf_get_error());
        return 1;
    }

    float err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = y[i] - x[i];
        err += d * d;
    }
    err = sqrtf(err / (float)n);
    printf("Round-trip RMS error: %.3e\n", err);

    free(x); free(y); free(re); free(im);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    faf_cleanup();
    return err > 1e-4f ? 1 : 0;
}
