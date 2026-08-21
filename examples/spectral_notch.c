/**
 * @file spectral_notch.c
 * @brief Register three lines of C, kill a tone, R2C round-trip
 */

#include "fastandfourier.h"
#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct notch_ctx { size_t bin; };

static void notch(float *re, float *im, size_t n_bins, void *ctx) {
    size_t k = ((struct notch_ctx *)ctx)->bin;
    if (k < n_bins) {
        re[k] = 0.0f;
        im[k] = 0.0f;
    }
}

int main(void) {
    const size_t n = 256;
    struct notch_ctx ctx = { .bin = 16 };
    chirp_register_spectral("notch", notch, &ctx);

    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 256) (spectral notch) (irfft))");
    if (!t) {
        fprintf(stderr, "compile: %s\n", faf_get_error());
        return 1;
    }

    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *y = (float *)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        x[i] = sinf(2.0f * (float)M_PI * 4.0f * (float)i / (float)n) +
               sinf(2.0f * (float)M_PI * 16.0f * (float)i / (float)n);
    }

    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_real(y, n);
    if (faf_execute(t, &out, &in) != 0) {
        fprintf(stderr, "execute: %s\n", faf_get_error());
        return 1;
    }

    printf("spectral notch: killed bin %zu, jit=%s\n",
           ctx.bin, t->jit_cache ? "yes" : "vm");
    printf("sample 0: in=%.3f out=%.3f\n", x[0], y[0]);

    free(x); free(y);
    faf_destroy_transform(t);
    chirp_cleanup();
    return 0;
}
