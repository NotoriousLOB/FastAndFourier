/**
 * @file chirp_dwt_transients.c
 * @brief Haar localizes a step; smoother families leak less into coarse bands
 */
#include "dwt_common.h"

int main(void) {
    const size_t n = 256;
    const size_t levels = 4;
    faf_init();

    float *in = dwt_alloc(n);
    float *coef = dwt_alloc(n);
    /* Linear chirp plus a unit step at n/2. */
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / (float)n;
        float inst_f = 2.0f + 18.0f * t;
        in[2 * i] = 0.4f * sinf(2.0f * (float)M_PI * inst_f * t);
        if (i >= n / 2) in[2 * i] += 1.0f;
    }

    printf("Transient localization  step at sample %zu\n", n / 2);
    printf("%-10s  finest-detail peak  peak index  coarse energy\n", "family");

    faf_wavelet_family fams[] = {
        FAF_WAVELET_HAAR, FAF_WAVELET_D4, FAF_WAVELET_CDF97
    };
    for (int k = 0; k < 3; k++) {
        faf_transform *t = faf_create_dwt(fams[k], n, levels, false, FAF_PREC_FP32, 0);
        if (!t) return 1;
        faf_execute_f32(t, coef, in);
        size_t fine0 = n / 2;
        float peak = 0.0f;
        size_t peak_i = fine0;
        for (size_t i = fine0; i < n; i++) {
            float a = fabsf(coef[2 * i]);
            if (a > peak) { peak = a; peak_i = i; }
        }
        float coarse = dwt_energy(coef, 0, n >> levels);
        printf("%-10s  %17.4f  %10zu  %13.4f\n",
               faf_wavelet_name(fams[k]), peak, peak_i, coarse);
        faf_destroy_transform(t);
    }

    free(in); free(coef);
    faf_cleanup();
    return 0;
}
