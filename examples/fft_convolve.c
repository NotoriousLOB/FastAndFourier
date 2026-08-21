/**
 * @file fft_convolve.c
 * @brief FFT convolution of real signals via mul-spectrum
 */

#include "fastandfourier.h"
#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    const size_t n = 64;
    const size_t nb = n / 2 + 1;

    /* Moving-average kernel of length 4, zero-padded to n. */
    float *h = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < 4; i++) h[i] = 0.25f;

    faf_config cfg = faf_config_init(n);
    faf_transform *rf = faf_create_rfft(&cfg);
    float *Hr = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Hi = (float *)aligned_alloc(64, nb * sizeof(float));
    faf_buffer hin = faf_buffer_real(h, n);
    faf_buffer hs = faf_buffer_hermitian(Hr, Hi, nb);
    if (faf_execute(rf, &hs, &hin) != 0) {
        fprintf(stderr, "rfft(h): %s\n", faf_get_error());
        return 1;
    }
    faf_destroy_transform(rf);

    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 64) (mul-spectrum) (irfft))");
    if (!t || chirp_bind(t, "H", Hr, Hi, nb) != 0) {
        fprintf(stderr, "pipeline: %s\n", faf_get_error());
        return 1;
    }

    float *x = (float *)calloc(n, sizeof(float));
    float *y = (float *)calloc(n, sizeof(float));
    x[10] = 1.0f; /* impulse → the kernel, circular */

    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_real(y, n);
    if (faf_execute(t, &out, &in) != 0) {
        fprintf(stderr, "execute: %s\n", faf_get_error());
        return 1;
    }

    printf("fft_convolve: impulse at 10 through 4-tap boxcar\n");
    for (size_t i = 10; i < 14; i++)
        printf("  y[%zu] = %.3f\n", i, y[i]);

    free(h); free(Hr); free(Hi); free(x); free(y);
    faf_destroy_transform(t);
    chirp_cleanup();
    return 0;
}
