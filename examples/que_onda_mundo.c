/**
 * @file que_onda_mundo.c
 * @brief Canonical Chirp hello world — ¿Qué onda mundo?
 *
 * The smallest complete Chirp path: register, compile, execute, print.
 * An 8-point impulse FFT should come back flat (all ones).
 */

#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("¿Qué onda mundo?\n");
    printf("Chirp's hello world: compile an S-expression, run an FFT.\n\n");

    faf_init();
    chirp_register_standard_builtins();

    faf_transform *t = chirp_compile("(fft :size 8)");
    if (!t) {
        fprintf(stderr, "compile failed: %s\n", faf_get_error());
        return 1;
    }

    const size_t n = t->n;
    float *in = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    float *out = (float*)aligned_alloc(64, 2 * n * sizeof(float));
    if (!in || !out) {
        faf_destroy_transform(t);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        in[2 * i]     = (i == 0) ? 1.0f : 0.0f;
        in[2 * i + 1] = 0.0f;
    }

    if (faf_execute_f32(t, out, in) != 0) {
        fprintf(stderr, "execute failed: %s\n", faf_get_error());
        free(in); free(out);
        faf_destroy_transform(t);
        return 1;
    }

    printf("  (fft :size %zu)  impulse in, spectrum out:\n", n);
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        printf("    bin %zu  %+.3f%+.3fi\n", i, out[2 * i], out[2 * i + 1]);
        if (fabsf(out[2 * i] - 1.0f) > 1e-4f || fabsf(out[2 * i + 1]) > 1e-4f)
            ok = 0;
    }

    printf(ok ? "\nOnda. Flat spectrum, as promised.\n"
              : "\nNot the onda we wanted — impulse FFT should be all ones.\n");

    free(in);
    free(out);
    faf_destroy_transform(t);
    chirp_cleanup();
    faf_cleanup();
    return ok ? 0 : 1;
}
