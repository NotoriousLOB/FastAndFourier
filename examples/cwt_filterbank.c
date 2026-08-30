/**
 * @file cwt_filterbank.c
 * @brief Create a CWT filter bank and print the Littlewood-Paley report
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("FastAndFourier - CWT Filter Bank\n");
    printf("================================\n\n");

    faf_init();
    printf("Library version: %s\n", faf_version());
    printf("Architecture: %s\n\n", faf_arch_name());

    const size_t n = 4096;
    faf_cwt_config cfg = faf_cwt_config_init(n);
    cfg.wavelet = FAF_CWT_WAVELET_MORSE;
    cfg.voices = 10;
    cfg.precision = FAF_PREC_FP32;

    faf_transform *cwt = faf_create_cwt(&cfg);
    if (!cwt) {
        fprintf(stderr, "Failed to create CWT: %s\n", faf_get_error());
        return 1;
    }

    printf("Transform: %s\n", faf_transform_name(cwt->type));
    printf("n = %zu\n", n);
    printf("Scales: %zu\n", faf_cwt_n_scales(cwt));
    printf("Rows (scales + lowpass): %zu\n", faf_cwt_n_rows(cwt));
    printf("Has lowpass: %s\n\n", faf_cwt_has_lowpass(cwt) ? "yes" : "no");

    size_t J = faf_cwt_n_scales(cwt);
    double *hz = (double *)malloc(J * sizeof(double));
    faf_cwt_freqs(cwt, hz, J);
    printf("Center frequencies (Hz):\n");
    for (size_t j = 0; j < J && j < 10; j++)
        printf("  scale %2zu: %8.4f Hz\n", j, hz[j]);
    if (J > 10) printf("  ... (%zu more)\n", J - 10);
    printf("\n");
    free(hz);

    const faf_cwt_lp_report *rpt = faf_cwt_bank_report(cwt);
    if (rpt) {
        printf("Littlewood-Paley Report:\n");
        faf_cwt_report_fprint(stdout, rpt);
    }

    faf_destroy_transform(cwt);
    faf_cleanup();
    return 0;
}
