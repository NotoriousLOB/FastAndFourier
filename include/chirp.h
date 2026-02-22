#ifndef CHIRP_H
#define CHIRP_H

#include "fastandfourier.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Register a custom C function (gaussian, softmax, etc.) */
int chirp_register(const char *name, void (*fn)(void));

/* Free all registered builtin names and reset the registry */
void chirp_cleanup(void);

/* Compile a Chirp S-expression to a transform */
faf_transform* chirp_compile(const char *source);

/* Example: chirp_compile("(pipeline (fft :size 1024) twiddle (bfly 4) (lift :predict gaussian :update softmax) (custom softmax) reduce-sum)"); */

#ifdef __cplusplus
}
#endif

#endif

