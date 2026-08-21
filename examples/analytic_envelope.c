/**
 * @file analytic_envelope.c
 * @brief Hilbert envelope via packed R2C + split-plane C2C IFFT
 *
 * Packed R2C already dropped the negative frequencies. Scale bins 1..n/2-1
 * by 2, expand into a full-length split spectrum with negatives zeroed,
 * and IFFT C2C to get the complex analytic signal. Envelope = hypot(re, im).
 * That is the split-storage payoff: no interleaved zeros for the missing half.
 */

#include "fastandfourier.h"
#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void hilbert_scale(float *re, float *im, size_t n_bins, void *ctx) {
    (void)ctx;
    for (size_t k = 1; k + 1 < n_bins; k++) {
        re[k] *= 2.0f;
        im[k] *= 2.0f;
    }
}

int main(void) {
    const size_t n = 256;
    const size_t nb = n / 2 + 1;
    chirp_register_spectral("hilbert", hilbert_scale, NULL);

    faf_transform *an = chirp_compile(
        "(pipeline (rfft :size 256) (spectral hilbert))");
    if (!an) {
        fprintf(stderr, "compile: %s\n", faf_get_error());
        return 1;
    }

    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    float *Xr = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Xi = (float *)aligned_alloc(64, nb * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        float env = 0.5f + 0.5f * sinf(2.0f * (float)M_PI * 2.0f *
                                       (float)i / (float)n);
        x[i] = env * sinf(2.0f * (float)M_PI * 20.0f * (float)i / (float)n);
    }

    faf_buffer in = faf_buffer_real(x, n);
    faf_buffer spec = faf_buffer_hermitian(Xr, Xi, nb);
    if (faf_execute(an, &spec, &in) != 0) {
        fprintf(stderr, "rfft: %s\n", faf_get_error());
        return 1;
    }

    /* Expand packed analytic spectrum into a full-length split C2C buffer. */
    float *full_re = (float *)calloc(n, sizeof(float));
    float *full_im = (float *)calloc(n, sizeof(float));
    memcpy(full_re, Xr, nb * sizeof(float));
    memcpy(full_im, Xi, nb * sizeof(float));
    /* bins n/2+1 .. n-1 stay zero — the negative frequencies */

    faf_config ic = faf_config_init(n);
    ic.dir = FAF_DIR_INVERSE;
    ic.layout = FAF_LAYOUT_SPLIT;
    faf_transform *ifft = faf_create_fft(&ic);
    float *are = (float *)aligned_alloc(64, n * sizeof(float));
    float *aim = (float *)aligned_alloc(64, n * sizeof(float));
    faf_buffer cin = faf_buffer_split(full_re, full_im, n);
    faf_buffer cout = faf_buffer_split(are, aim, n);
    if (faf_execute(ifft, &cout, &cin) != 0) {
        fprintf(stderr, "ifft: %s\n", faf_get_error());
        return 1;
    }

    printf("analytic_envelope (first 8 samples):\n");
    printf("  i\tx\tenvelope\n");
    for (size_t i = 0; i < 8; i++)
        printf("  %zu\t%7.3f\t%7.3f\n", i, x[i], hypotf(are[i], aim[i]));

    free(x); free(Xr); free(Xi); free(full_re); free(full_im);
    free(are); free(aim);
    faf_destroy_transform(an);
    faf_destroy_transform(ifft);
    chirp_cleanup();
    return 0;
}
