/**
 * @file matched_filter.c
 * @brief Split-plane correlation: R2C(x) * conj(R2C(template)) → C2R
 */

#include "fastandfourier.h"
#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    const size_t n = 128;
    const size_t nb = n / 2 + 1;

    float *tmpl = (float *)calloc(n, sizeof(float));
    float *x = (float *)calloc(n, sizeof(float));
    for (size_t i = 0; i < 16; i++)
        tmpl[i] = sinf(2.0f * (float)M_PI * (float)i / 16.0f);
    /* Place the template at offset 40 with a little noise. */
    for (size_t i = 0; i < n; i++)
        x[i] = ((i % 17) * 0.01f);
    for (size_t i = 0; i < 16; i++)
        x[40 + i] += tmpl[i];

    faf_config cfg = faf_config_init(n);
    faf_transform *rf = faf_create_rfft(&cfg);
    float *Hr = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Hi = (float *)aligned_alloc(64, nb * sizeof(float));
    faf_buffer tin = faf_buffer_real(tmpl, n);
    faf_buffer hs = faf_buffer_hermitian(Hr, Hi, nb);
    faf_execute(rf, &hs, &tin);
    faf_destroy_transform(rf);

    /* Correlation: X * conj(H). Conjugate the bound template. */
    faf_transform *t = chirp_compile(
        "(pipeline (rfft :size 128) (mul-spectrum) (irfft))");
    for (size_t k = 0; k < nb; k++)
        Hi[k] = -Hi[k];
    chirp_bind(t, "H", Hr, Hi, nb);

    float *y = (float *)calloc(n, sizeof(float));
    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer out = faf_buffer_real(y, n);
    if (faf_execute(t, &out, &in) != 0) {
        fprintf(stderr, "execute: %s\n", faf_get_error());
        return 1;
    }

    size_t peak = 0;
    float best = y[0];
    for (size_t i = 1; i < n; i++) {
        if (y[i] > best) { best = y[i]; peak = i; }
    }
    printf("matched_filter: peak at lag %zu (expected 40), value %.3f\n",
           peak, best);

    free(tmpl); free(x); free(y); free(Hr); free(Hi);
    faf_destroy_transform(t);
    chirp_cleanup();
    return 0;
}
