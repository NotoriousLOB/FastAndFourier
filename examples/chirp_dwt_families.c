/**
 * @file chirp_dwt_families.c
 * @brief Compare Haar / D4 / CDF 5/3 / CDF 9/7 / Sym4 on one mixed signal
 */
#include "dwt_common.h"

int main(void) {
    const size_t n = 256;
    const size_t levels = 4;
    faf_init();
    chirp_register_standard_builtins();

    float *in = dwt_alloc(n);
    float *mid = dwt_alloc(n);
    float *out = dwt_alloc(n);
    if (!in || !mid || !out) return 1;

    /* Tone + step + ramp — a mix that stresses every family differently. */
    for (size_t i = 0; i < n; i++) {
        float x = (float)i / (float)n;
        float tone = 0.6f * sinf(2.0f * (float)M_PI * 5.0f * x);
        float step = (i < n / 2) ? 0.0f : 0.8f;
        float ramp = 0.25f * x;
        in[2 * i] = tone + step + ramp;
    }

    printf("Chirp DWT family tour  n=%zu  levels=%zu\n", n, levels);
    printf("%-10s  %12s  %14s  %s\n", "family", "PR max|err|", "detail energy", "status");

    const char *names[] = {"haar", "d4", "cdf53", "cdf97", "sym4"};
    int failed = 0;
    for (int k = 0; k < 5; k++) {
        char src[256];
        snprintf(src, sizeof(src),
                 "(pipeline (dwt :family %s :size %zu :levels %zu))",
                 names[k], n, levels);
        faf_transform *fwd = chirp_compile(src);
        snprintf(src, sizeof(src),
                 "(pipeline (idwt :family %s :size %zu :levels %zu))",
                 names[k], n, levels);
        faf_transform *inv = chirp_compile(src);
        if (!fwd || !inv) {
            printf("%-10s  compile failed: %s\n", names[k], faf_get_error());
            failed = 1;
            faf_destroy_transform(fwd);
            faf_destroy_transform(inv);
            continue;
        }
        faf_execute_f32(fwd, mid, in);
        faf_execute_f32(inv, out, mid);
        float err = dwt_max_abs(out, in, n);
        float de = dwt_energy(mid, n >> levels, n);
        int ok = err < 5e-4f;
        printf("%-10s  %12.3e  %14.4f  %s\n", names[k], err, de, ok ? "ok" : "FAIL");
        if (!ok) failed = 1;
        faf_destroy_transform(fwd);
        faf_destroy_transform(inv);
    }

    free(in); free(mid); free(out);
    chirp_cleanup();
    faf_cleanup();
    return failed;
}
