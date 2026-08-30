/**
 * @file split_corr.c
 * @brief Split-plane real correlation via packed Hermitian multiply
 *
 * Default fast path: REAL → HERMITIAN (re[n/2+1], im[n/2+1]) →
 * faf_herm_mul_conj → IRFFT. Interleaved storage is not used.
 *
 * Also registers a Chirp spectral builtin named "corr" so the same
 * kernel can sit inside (pipeline (rfft) (spectral corr) (irfft)).
 */

#include "fastandfourier.h"
#include "chirp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct corr_ctx {
    float *Hr;
    float *Hi;
};

static void corr_spectral_f32(float *re, float *im, size_t n_bins, void *ctx) {
    struct corr_ctx *c = (struct corr_ctx *)ctx;
    faf_herm_mul_conj_f32(re, im, re, im, c->Hr, c->Hi, n_bins);
}

int main(void) {
    const size_t n = 128;
    const size_t nb = n / 2 + 1;
    const size_t lag = 40;
    const size_t w = 16;

    float *tmpl = (float *)calloc(n, sizeof(float));
    float *x = (float *)calloc(n, sizeof(float));
    float *y = (float *)calloc(n, sizeof(float));
    float *Hr = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Hi = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Xr = (float *)aligned_alloc(64, nb * sizeof(float));
    float *Xi = (float *)aligned_alloc(64, nb * sizeof(float));
    if (!tmpl || !x || !y || !Hr || !Hi || !Xr || !Xi) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (size_t i = 0; i < w; i++)
        tmpl[i] = sinf(2.0f * (float)M_PI * (float)i / (float)w);
    for (size_t i = 0; i < n; i++)
        x[i] = ((i % 17) * 0.01f);
    for (size_t i = 0; i < w; i++)
        x[lag + i] += tmpl[i];

    faf_config cfg = faf_config_init(n);
    faf_transform *fwd = faf_create_rfft(&cfg);
    faf_transform *inv = faf_create_inverse(fwd);
    if (!fwd || !inv) {
        fprintf(stderr, "create: %s\n", faf_get_error());
        return 1;
    }

    faf_buffer tin = faf_buffer_real(tmpl, n);
    faf_buffer Hs = faf_buffer_hermitian(Hr, Hi, nb);
    faf_buffer xin = faf_buffer_real(x, n);
    faf_buffer Xs = faf_buffer_hermitian(Xr, Xi, nb);
    faf_buffer yb = faf_buffer_real(y, n);

    if (faf_execute(fwd, &Hs, &tin) != 0 ||
        faf_execute(fwd, &Xs, &xin) != 0) {
        fprintf(stderr, "rfft: %s\n", faf_get_error());
        return 1;
    }

    /* Packed path: DC and Nyquist stay real. */
    faf_herm_mul_conj_f32(Xr, Xi, Xr, Xi, Hr, Hi, nb);
    if (faf_execute(inv, &yb, &Xs) != 0) {
        fprintf(stderr, "irfft: %s\n", faf_get_error());
        return 1;
    }

    size_t peak = 0;
    float best = y[0];
    for (size_t i = 1; i < n; i++) {
        if (y[i] > best) { best = y[i]; peak = i; }
    }
    printf("split_corr C API: peak at lag %zu (expected %zu), value %.3f\n",
           peak, lag, best);
    printf("  DC imag after mul = %.3g  Nyquist imag = %.3g\n",
           Xi[0], Xi[nb - 1]);

    /* Same operator as a Chirp spectral builtin. */
    struct corr_ctx ctx = { Hr, Hi };
    chirp_register_spectral_ex("corr", corr_spectral_f32, NULL, &ctx);
    faf_transform *pipe = chirp_compile(
        "(pipeline (rfft :size 128 :layout hermitian) (spectral corr) (irfft))");
    if (!pipe) {
        fprintf(stderr, "chirp: %s\n", faf_get_error());
        return 1;
    }
    for (size_t i = 0; i < n; i++) y[i] = 0.0f;
    if (faf_execute(pipe, &yb, &xin) != 0) {
        fprintf(stderr, "pipeline: %s\n", faf_get_error());
        return 1;
    }
    peak = 0;
    best = y[0];
    for (size_t i = 1; i < n; i++) {
        if (y[i] > best) { best = y[i]; peak = i; }
    }
    printf("split_corr Chirp: peak at lag %zu (expected %zu), value %.3f\n",
           peak, lag, best);

    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    faf_destroy_transform(pipe);
    chirp_cleanup();
    free(tmpl); free(x); free(y);
    free(Hr); free(Hi); free(Xr); free(Xi);
    return (peak == lag) ? 0 : 1;
}