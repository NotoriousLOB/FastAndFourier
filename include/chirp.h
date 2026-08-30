#ifndef CHIRP_H
#define CHIRP_H

#include "fastandfourier.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Builtin calling conventions.
 *
 * UNARY is the slow kind: a scalar loop on the real plane.
 * SPECTRAL / HERMITIAN are the Fourier-domain C ABI: one call with live
 * split-plane pointers. OTHER is the legacy "not a unary scalar" bucket.
 */
typedef enum {
    CHIRP_KIND_UNARY = 0,     /* float(float) / double(double), elementwise */
    CHIRP_KIND_OTHER = 1,     /* not safe to call as a unary scalar */
    CHIRP_KIND_COMPLEX,       /* per-bin in-place (re, im) */
    CHIRP_KIND_VECTOR,        /* void(float *x, size_t n) on the real plane */
    CHIRP_KIND_SPECTRAL,      /* void(re, im, n_bins, ctx) split-plane */
    CHIRP_KIND_HERMITIAN      /* SPECTRAL, then force DC/Nyquist imag = 0 */
} chirp_kind;

typedef void (*chirp_spectral_fn)(float *re, float *im, size_t n_bins, void *ctx);
typedef void (*chirp_spectral_fn_f64)(double *re, double *im, size_t n_bins,
                                      void *ctx);
typedef void (*chirp_complex_fn)(float *re, float *im);
typedef void (*chirp_complex_fn_f64)(double *re, double *im);
typedef void (*chirp_vector_fn)(float *x, size_t n);
typedef void (*chirp_vector_fn_f64)(double *x, size_t n);

/* Reserved CALL_BUILTIN.a0 values for pipeline-only ops (not in the registry) */
#define CHIRP_OP_CONJ      0x7ffffffdu
#define CHIRP_OP_MUL       0x7ffffffeu
#define CHIRP_OP_BANDPASS  0x7fffffffu

/* Register a custom C function (gaussian, softmax, etc.) */
int chirp_register(const char *name, void (*fn)(void));
int chirp_register_ex(const char *name, void (*fn)(void), int kind);
int chirp_register_unary(const char *name, float (*f32)(float),
                         double (*f64)(double));
int chirp_register_spectral(const char *name, chirp_spectral_fn fn, void *ctx);
int chirp_register_spectral_ex(const char *name, chirp_spectral_fn f32,
                               chirp_spectral_fn_f64 f64, void *ctx);
int chirp_builtin_kind(int idx);
void *chirp_builtin_ctx(int idx);

/* Bind a named auxiliary spectrum (e.g. "H") for mul-spectrum. */
int chirp_bind(faf_transform *t, const char *name, void *re, void *im,
               size_t n_bins);

/**
 * Register a named real vector (caller owns the data) for `(bind … :h NAME)`.
 */
int chirp_register_vector(const char *name, const float *data, int len);

/**
 * Register a named DWT tap set for `(dwt :taps NAME …)`.
 * ht/gt NULL ⇒ analysis-only.
 */
int chirp_register_taps(const char *name,
                        const float *h, const float *g, int len_hg,
                        const float *ht, const float *gt, int len_syn);

/* Free all registered builtin names and reset the registry */
void chirp_cleanup(void);

/* Return the number of currently registered builtins */
int chirp_count(void);

/* Lookup helpers for VM/JIT dispatch */
const char* chirp_builtin_name(int idx);
void* chirp_builtin_fn(int idx);
int chirp_builtin_count(void);
void* chirp_builtin_fn_for_precision(int idx, faf_precision precision);

/* Apply a CALL_BUILTIN (or reserved pipeline op) to split-plane buffers. */
void chirp_apply_split_f32(const faf_transform *t, uint32_t a0, uint32_t a1,
                           uint32_t a2, float *re, float *im, size_t n);
void chirp_apply_split_f64(const faf_transform *t, uint32_t a0, uint32_t a1,
                           uint32_t a2, double *re, double *im, size_t n);

/* Compile a Chirp S-expression to a transform */
faf_transform* chirp_compile(const char *source);

#ifdef __cplusplus
}
#endif

#endif

