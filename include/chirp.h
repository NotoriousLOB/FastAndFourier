#ifndef CHIRP_H
#define CHIRP_H

#include "fastandfourier.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Builtin calling-convention kinds for VM dispatch */
#define CHIRP_KIND_UNARY  0  /* float(float) / double(double), elementwise */
#define CHIRP_KIND_OTHER  1  /* not safe to call as a unary scalar */

/* Register a custom C function (gaussian, softmax, etc.) */
int chirp_register(const char *name, void (*fn)(void));
int chirp_register_ex(const char *name, void (*fn)(void), int kind);
int chirp_builtin_kind(int idx);

/* Free all registered builtin names and reset the registry */
void chirp_cleanup(void);

/* Return the number of currently registered builtins */
int chirp_count(void);

/* Lookup helpers for VM/JIT dispatch */
const char* chirp_builtin_name(int idx);
void* chirp_builtin_fn(int idx);
int chirp_builtin_count(void);
void* chirp_builtin_fn_for_precision(int idx, faf_precision precision);

/* Compile a Chirp S-expression to a transform */
faf_transform* chirp_compile(const char *source);

/* Example: chirp_compile("(pipeline (fft :size 1024) twiddle (bfly 4) (lift :predict gaussian :update softmax) (custom softmax) reduce-sum)"); */

#ifdef __cplusplus
}
#endif

#endif

