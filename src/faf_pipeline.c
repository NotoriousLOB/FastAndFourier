/**
 * @file faf_pipeline.c
 * @brief Fused Chirp pipeline: R2C → spectral C → C2R
 */

#include "faf.h"
#include "chirp.h"
#include <string.h>

int faf_pipeline_execute(const faf_transform *t, faf_buffer *out,
                         const faf_buffer *in) {
    if (!t || !out || !in || !t->inner) {
        faf_set_error("pipeline execute: missing transform");
        return -1;
    }
    if (!t->scratch) {
        faf_set_error("pipeline execute: missing scratch");
        return -1;
    }

    int has_inv = (t->inner_inv != NULL);

    /* JIT the fused REAL→REAL path: one generated kernel with a live
     * split-plane spectral call. Skip analysis-only (no irfft) and FP64. */
    if (has_inv && t->precision == FAF_PREC_FP32 &&
        t->cfg.backend != FAF_BACKEND_VM &&
        in->layout == FAF_LAYOUT_REAL && out->layout == FAF_LAYOUT_REAL &&
        in->re && out->re) {
        if (faf_execute_jit_cached(t, out->re, in->re) == 0)
            return 0;
    }
    size_t n = t->n;
    size_t nbins = n / 2 + 1;

    if (in->layout != FAF_LAYOUT_REAL || in->n != n || !in->re) {
        faf_set_error("pipeline input must be REAL of length %zu", n);
        return -1;
    }
    if (has_inv) {
        if (out->layout != FAF_LAYOUT_REAL || out->n != n || !out->re) {
            faf_set_error("pipeline output must be REAL of length %zu", n);
            return -1;
        }
    } else {
        if (out->layout != FAF_LAYOUT_HERMITIAN || out->n != nbins ||
            !out->re || !out->im) {
            faf_set_error("analysis pipeline output must be HERMITIAN "
                          "(%zu bins)", nbins);
            return -1;
        }
    }

    faf_buffer spec;
    memset(&spec, 0, sizeof(spec));
    spec.layout = FAF_LAYOUT_HERMITIAN;
    spec.n = nbins;
    if (t->precision == FAF_PREC_FP64) {
        spec.re = t->scratch;
        spec.im = (double *)t->scratch + nbins;
    } else {
        spec.re = t->scratch;
        spec.im = (float *)t->scratch + nbins;
    }

    if (faf_execute(t->inner, &spec, in) != 0)
        return -1;

    for (size_t i = 0; i < t->n_inst; i++) {
        const faf_inst *inst = &t->code[i];
        if (t->precision == FAF_PREC_FP64) {
            chirp_apply_split_f64(t, inst->a0, inst->a1, inst->a2,
                                  (double *)spec.re, (double *)spec.im, nbins);
        } else {
            chirp_apply_split_f32(t, inst->a0, inst->a1, inst->a2,
                                  (float *)spec.re, (float *)spec.im, nbins);
        }
    }

    if (has_inv)
        return faf_execute(t->inner_inv, out, &spec);

    if (t->precision == FAF_PREC_FP64) {
        memcpy(out->re, spec.re, nbins * sizeof(double));
        memcpy(out->im, spec.im, nbins * sizeof(double));
    } else {
        memcpy(out->re, spec.re, nbins * sizeof(float));
        memcpy(out->im, spec.im, nbins * sizeof(float));
    }
    return 0;
}
