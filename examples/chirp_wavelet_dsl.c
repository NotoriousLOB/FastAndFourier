/**
 * @file chirp_wavelet_dsl.c
 * @brief Execute Haar / CDF 5/3 / CDF 9/7 via Chirp lifting and dwt forms
 */
#include "dwt_common.h"

static int run_named(const char *label, const char *src,
                     const float *in, size_t n, float *mid, float *out) {
    faf_transform *fwd = chirp_compile(src);
    if (!fwd) {
        fprintf(stderr, "%s compile failed: %s\n", label, faf_get_error());
        return 1;
    }
    char isrc[256];
    size_t levels = fwd->levels ? fwd->levels : 1;
    snprintf(isrc, sizeof(isrc),
             "(idwt :family %s :size %zu :levels %zu)",
             faf_wavelet_name(fwd->family), n, levels);
    faf_transform *inv = chirp_compile(isrc);
    if (!inv) {
        faf_destroy_transform(fwd);
        return 1;
    }
    faf_execute_f32(fwd, mid, in);
    faf_execute_f32(inv, out, mid);
    float err = dwt_max_abs(out, in, n);
    printf("%-28s  %zu inst  PR %.3e\n", label, fwd->n_inst, err);
    int rc = err < 5e-4f ? 0 : 1;
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    return rc;
}

int main(void) {
    printf("Wavelets as Chirp compositions\n\n");
    faf_init();
    chirp_register_standard_builtins();

    const size_t n = 64;
    float *in = dwt_alloc(n);
    float *mid = dwt_alloc(n);
    float *out = dwt_alloc(n);
    for (size_t i = 0; i < n; i++)
        in[2 * i] = sinf(2.0f * (float)M_PI * 3.0f * (float)i / (float)n);

    int rc = 0;
    rc |= run_named("Haar via (dwt)",
                    "(dwt :family haar :size 64 :levels 3)",
                    in, n, mid, out);
    rc |= run_named("Haar via lift",
                    "(pipeline (lift :predict haar :update haar))",
                    in, n, mid, out);
    rc |= run_named("CDF 5/3",
                    "(dwt :family cdf53 :size 64 :levels 3)",
                    in, n, mid, out);
    rc |= run_named("CDF 9/7",
                    "(dwt :family cdf97 :size 64 :levels 3)",
                    in, n, mid, out);
    rc |= run_named("linear/cubic lift aliases",
                    "(pipeline (lift :predict predict_linear :update update_linear))",
                    in, n, mid, out);

    free(in); free(mid); free(out);
    chirp_cleanup();
    faf_cleanup();
    return rc;
}
