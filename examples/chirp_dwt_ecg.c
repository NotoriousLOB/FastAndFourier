/**
 * @file chirp_dwt_ecg.c
 * @brief Synthetic ECG: strip baseline wander and EMG with CDF 9/7
 */
#include "dwt_common.h"

static float gauss(float t, float mu, float s) {
    float x = (t - mu) / s;
    return expf(-0.5f * x * x);
}

int main(void) {
    const size_t n = 512;
    const size_t levels = 6;
    faf_init();
    chirp_register_standard_builtins();

    float *clean = dwt_alloc(n);
    float *noisy = dwt_alloc(n);
    float *out = dwt_alloc(n);

    /* Two beats: P, QRS, T plus slow wander and high-freq EMG. */
    for (size_t i = 0; i < n; i++) {
        float t = (float)i;
        float beat = 0.0f;
        for (int b = 0; b < 2; b++) {
            float c = 80.0f + 220.0f * (float)b;
            beat += 0.15f * gauss(t, c - 25.0f, 8.0f);
            beat += 1.00f * gauss(t, c, 4.0f);
            beat -= 0.20f * gauss(t, c + 8.0f, 5.0f);
            beat += 0.35f * gauss(t, c + 40.0f, 14.0f);
        }
        clean[2 * i] = beat;
        float wander = 0.25f * sinf(2.0f * (float)M_PI * (float)i / (float)n);
        float emg = 0.08f * sinf(2.0f * (float)M_PI * 40.0f * (float)i / (float)n);
        noisy[2 * i] = beat + wander + emg;
    }

    faf_transform *fwd = faf_create_cdf97(n, levels, FAF_PREC_FP32, 0);
    faf_transform *inv = faf_create_dwt(FAF_WAVELET_CDF97, n, levels, true, FAF_PREC_FP32, 0);
    if (!fwd || !inv) return 1;

    float *coef = dwt_alloc(n);
    faf_execute_f32(fwd, coef, noisy);

    /* Drop the coarsest approximation (baseline) and tiny finest details (EMG). */
    size_t approx = n >> levels;
    for (size_t i = 0; i < approx; i++) coef[2 * i] = 0.0f;
    for (size_t i = n / 2; i < n; i++) {
        if (fabsf(coef[2 * i]) < 0.05f) coef[2 * i] = 0.0f;
    }
    faf_execute_f32(inv, out, coef);

    /* QRS peak preservation: max of clean vs max of reconstruction. */
    float peak_c = 0.0f, peak_o = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (clean[2 * i] > peak_c) peak_c = clean[2 * i];
        if (out[2 * i] > peak_o) peak_o = out[2 * i];
    }
    float wander_in = 0.0f, wander_out = 0.0f;
    /* Low-frequency residual vs the clean beat (no wander). */
    for (size_t i = 0; i < n; i++) {
        wander_in += fabsf(noisy[2 * i] - clean[2 * i]);
        wander_out += fabsf(out[2 * i] - clean[2 * i]);
    }
    wander_in /= (float)n;
    wander_out /= (float)n;

    printf("Synthetic ECG  n=%zu  CDF97 levels=%zu\n", n, levels);
    printf("  mean |error| vs clean beat:  input %.4f  output %.4f\n",
           wander_in, wander_out);
    printf("  QRS peak:  clean %.3f  reconstructed %.3f  ratio %.3f\n",
           peak_c, peak_o, peak_o / (peak_c + 1e-8f));

    int rc = (wander_out < wander_in && peak_o / (peak_c + 1e-8f) > 0.6f) ? 0 : 1;

    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    free(clean); free(noisy); free(out); free(coef);
    chirp_cleanup();
    faf_cleanup();
    return rc;
}
