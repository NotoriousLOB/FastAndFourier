/**
 * @file cwt_reconstruct.c
 * @brief CWT forward + dual-frame inverse reconstruction demo
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - CWT Reconstruction\n");
    printf("====================================\n\n");

    faf_init();

    const size_t n = 4096;
    faf_cwt_config cfg = faf_cwt_config_init(n);
    cfg.precision = FAF_PREC_FP64;

    faf_transform *cwt = faf_create_cwt(&cfg);
    if (!cwt) {
        fprintf(stderr, "Failed to create CWT: %s\n", faf_get_error());
        return 1;
    }
    faf_transform *icwt = faf_create_inverse(cwt);
    if (!icwt) {
        fprintf(stderr, "Failed to create ICWT: %s\n", faf_get_error());
        faf_destroy_transform(cwt);
        return 1;
    }

    size_t n_rows = faf_cwt_n_rows(cwt);
    printf("n=%zu  scales=%zu  rows=%zu\n\n", n, faf_cwt_n_scales(cwt), n_rows);

    double *x = (double *)aligned_alloc(64, n * sizeof(double));
    double *W = (double *)aligned_alloc(64, n_rows * n * sizeof(double));
    double *y = (double *)aligned_alloc(64, n * sizeof(double));

    /* Test 1: unit impulse */
    memset(x, 0, n * sizeof(double));
    x[0] = 1.0;

    faf_buffer in_buf = faf_buffer_real(x, n);
    faf_buffer w_buf;
    memset(&w_buf, 0, sizeof(w_buf));
    w_buf.re = W;
    w_buf.n = n;
    w_buf.layout = FAF_LAYOUT_REAL;
    faf_buffer out_buf = faf_buffer_real(y, n);

    faf_execute(cwt, &w_buf, &in_buf);
    faf_execute(icwt, &out_buf, &w_buf);

    double err_impulse = 0.0, norm_impulse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = y[i] - x[i];
        err_impulse += d * d;
        norm_impulse += x[i] * x[i];
    }
    double rel_impulse = sqrt(err_impulse / norm_impulse);
    printf("Impulse reconstruction:  rel L2 = %.2e\n", rel_impulse);

    /* Test 2: sine at 10 Hz */
    for (size_t i = 0; i < n; i++)
        x[i] = sin(2.0 * M_PI * 10.0 * (double)i / (double)n);

    in_buf = faf_buffer_real(x, n);
    faf_execute(cwt, &w_buf, &in_buf);
    faf_execute(icwt, &out_buf, &w_buf);

    double err_sine = 0.0, norm_sine = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = y[i] - x[i];
        err_sine += d * d;
        norm_sine += x[i] * x[i];
    }
    double rel_sine = sqrt(err_sine / norm_sine);
    printf("Sine reconstruction:     rel L2 = %.2e\n", rel_sine);

    /* Test 3: sum of 3 tones */
    for (size_t i = 0; i < n; i++)
        x[i] = sin(2.0 * M_PI * 5.0 * (double)i / (double)n) +
               0.7 * sin(2.0 * M_PI * 20.0 * (double)i / (double)n) +
               0.3 * sin(2.0 * M_PI * 80.0 * (double)i / (double)n);

    in_buf = faf_buffer_real(x, n);
    faf_execute(cwt, &w_buf, &in_buf);
    faf_execute(icwt, &out_buf, &w_buf);

    double err_tones = 0.0, norm_tones = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = y[i] - x[i];
        err_tones += d * d;
        norm_tones += x[i] * x[i];
    }
    double rel_tones = sqrt(err_tones / norm_tones);
    printf("3-tone reconstruction:   rel L2 = %.2e\n", rel_tones);

    int ok = (rel_impulse < 1e-4 && rel_sine < 1e-4 && rel_tones < 1e-4);
    printf("\n%s\n", ok ? "All reconstructions OK" : "RECONSTRUCTION ERROR");

    free(x); free(W); free(y);
    faf_destroy_transform(icwt);
    faf_destroy_transform(cwt);
    faf_cleanup();
    return ok ? 0 : 1;
}
