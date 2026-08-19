/**
 * @file chirp_dwt_jpeg2000.c
 * @brief CDF 5/3 lossless round-trip and CDF 9/7 lossy quantization
 */
#include "dwt_common.h"

int main(void) {
    const size_t n = 256;
    const size_t levels = 5;
    faf_init();
    chirp_register_standard_builtins();

    float *in = dwt_alloc(n);
    float *mid = dwt_alloc(n);
    float *out = dwt_alloc(n);
    for (size_t i = 0; i < n; i++) {
        float x = (float)i / (float)n;
        in[2 * i] = 0.7f * sinf(2.0f * (float)M_PI * 3.0f * x)
                  + 0.3f * sinf(2.0f * (float)M_PI * 17.0f * x);
    }

    printf("JPEG 2000-style 1-D coding  n=%zu\n\n", n);

    /* Lossless: CDF 5/3 perfect reconstruction. */
    float err53 = 0.0f;
    if (dwt_run_pr(FAF_WAVELET_CDF53, n, levels, in, mid, out, &err53) != 0) {
        fprintf(stderr, "CDF 5/3 failed: %s\n", faf_get_error());
        return 1;
    }
    printf("CDF 5/3 lossless  PR max|err| = %.3e  %s\n",
           err53, err53 < 1e-5f ? "ok" : "FAIL");

    /* Lossy: CDF 9/7, uniform-quantize details, invert. */
    faf_transform *fwd = dwt_make(FAF_WAVELET_CDF97, n, levels, 0);
    faf_transform *inv = dwt_make(FAF_WAVELET_CDF97, n, levels, 1);
    if (!fwd || !inv) return 1;
    faf_execute_f32(fwd, mid, in);

    const float step = 0.08f;
    size_t approx = n >> levels;
    size_t kept = approx;
    for (size_t i = approx; i < n; i++) {
        float q = nearbyintf(mid[2 * i] / step) * step;
        if (q != 0.0f) kept++;
        mid[2 * i] = q;
    }
    faf_execute_f32(inv, out, mid);
    float err97 = dwt_max_abs(out, in, n);
    float ratio = (float)n / (float)kept;
    printf("CDF 9/7 lossy     qstep=%.2f  keep %zu/%zu (%.1fx)  max|err|=%.4f\n",
           step, kept, n, ratio, err97);

    int rc = (err53 < 1e-5f && err97 < 0.5f) ? 0 : 1;
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
    free(in); free(mid); free(out);
    chirp_cleanup();
    faf_cleanup();
    return rc;
}
