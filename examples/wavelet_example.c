/**
 * @file wavelet_example.c
 * @brief Multi-family C API DWT with inverse reconstruction
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - Wavelet families (forward + inverse)\n\n");
    faf_init();

    const size_t n = 128;
    const size_t levels = 3;
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *mid = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        in[2 * i] = (i < n / 2) ? 0.0f : 1.0f;
        in[2 * i + 1] = 0.0f;
    }

    faf_wavelet_family fams[] = {
        FAF_WAVELET_HAAR, FAF_WAVELET_D4, FAF_WAVELET_CDF53,
        FAF_WAVELET_CDF97, FAF_WAVELET_SYM4
    };

    printf("%-10s  PR max|err|\n", "family");
    int rc = 0;
    for (size_t k = 0; k < sizeof(fams) / sizeof(fams[0]); k++) {
        faf_transform *fwd = faf_create_dwt(fams[k], n, levels, false, FAF_PREC_FP32, 0);
        faf_transform *inv = faf_create_dwt(fams[k], n, levels, true, FAF_PREC_FP32, 0);
        if (!fwd || !inv) {
            fprintf(stderr, "create failed: %s\n", faf_get_error());
            rc = 1;
            faf_destroy_transform(fwd);
            faf_destroy_transform(inv);
            continue;
        }
        faf_execute_f32(fwd, mid, in);
        faf_execute_f32(inv, out, mid);
        float err = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float e = fabsf(out[2 * i] - in[2 * i]);
            if (e > err) err = e;
        }
        printf("%-10s  %.3e\n", faf_wavelet_name(fams[k]), err);
        if (err > 5e-4f) rc = 1;
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }

    free(in); free(mid); free(out);
    faf_cleanup();
    return rc;
}
