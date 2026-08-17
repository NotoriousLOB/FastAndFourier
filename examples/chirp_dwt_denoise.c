/**
 * @file chirp_dwt_denoise.c
 * @brief Soft-threshold denoising with Haar, D4, and CDF 9/7
 */
#include "dwt_common.h"

static int denoise_family(const char *family, size_t n, size_t levels,
                          float lambda, const float *clean, const float *noisy) {
    char src[320];
    snprintf(src, sizeof(src),
             "(pipeline "
             "  (dwt :family %s :size %zu :levels %zu)"
             "  (threshold :mode soft :lambda %.5f)"
             "  (idwt :family %s :size %zu :levels %zu))",
             family, n, levels, lambda, family, n, levels);
    faf_transform *t = chirp_compile(src);
    if (!t) {
        fprintf(stderr, "%s: compile failed: %s\n", family, faf_get_error());
        return -1;
    }
    float *out = dwt_alloc(n);
    if (faf_execute_f32(t, out, noisy) != 0) {
        free(out);
        faf_destroy_transform(t);
        return -1;
    }
    float snr_in = dwt_snr_db(noisy, clean, n);
    float snr_out = dwt_snr_db(out, clean, n);
    float kept = dwt_energy(out, 0, n) / (dwt_energy(noisy, 0, n) + 1e-12f);
    printf("%-8s  SNR in %6.2f dB   SNR out %6.2f dB   gain %+5.2f   energy keep %.3f\n",
           family, snr_in, snr_out, snr_out - snr_in, kept);
    int ok = snr_out > snr_in;
    free(out);
    faf_destroy_transform(t);
    return ok ? 0 : 1;
}

int main(void) {
    const size_t n = 512;
    const size_t levels = 5;
    faf_init();
    chirp_register_standard_builtins();

    float *clean = dwt_alloc(n);
    float *noisy = dwt_alloc(n);
    dwt_fill_tone(clean, n, 4.0f, 1.0f);
    memcpy(noisy, clean, 2 * n * sizeof(float));
    /* Sparse impulses + a little structured hash noise. */
    noisy[2 * 40] += 1.7f;
    noisy[2 * 200] -= 2.0f;
    noisy[2 * 350] += 1.5f;
    for (size_t i = 0; i < n; i++)
        noisy[2 * i] += 0.04f * ((float)((i * 17u) % 11) - 5.0f);

    printf("Chirp DWT denoise  n=%zu  levels=%zu  lambda=0.20\n", n, levels);
    int rc = 0;
    rc |= denoise_family("haar", n, levels, 0.20f, clean, noisy);
    rc |= denoise_family("d4", n, levels, 0.20f, clean, noisy);
    rc |= denoise_family("cdf97", n, levels, 0.20f, clean, noisy);

    free(clean); free(noisy);
    chirp_cleanup();
    faf_cleanup();
    return rc ? 1 : 0;
}
