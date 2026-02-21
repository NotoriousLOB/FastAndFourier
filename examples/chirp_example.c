/**
 * @file chirp_example.c
 * @brief Example demonstrating the Chirp DSL for DSP pipelines
 * 
 * Chirp is a Scheme-like DSL with Smalltalk-style keyword arguments:
 * 
 *   (pipeline
 *     (fft :size 1024)
 *     twiddle
 *     (bfly 4)
 *     (lift :predict gaussian :update softmax)
 *     (custom softmax)
 *     reduce-sum)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fastandfourier.h"
#include "chirp.h"

/* Example custom function implementations */
static void my_gaussian_impl(void) {
    printf("  [builtin] Gaussian function called\n");
}

static void my_softmax_impl(void) {
    printf("  [builtin] Softmax function called\n");
}

static void my_normalize_impl(void) {
    printf("  [builtin] Normalize function called\n");
}

/* Helper to print IR instructions */
static void print_ir(dspir_transform *t) {
    printf("Generated IR (%zu instructions):\n", t->n_inst);
    for (size_t i = 0; i < t->n_inst && i < 20; i++) {
        dspir_inst *inst = &t->code[i];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        const char *opname = "?";
        switch (op) {
            case DSPIR_NOP: opname = "NOP"; break;
            case DSPIR_LOAD: opname = "LOAD"; break;
            case DSPIR_STORE: opname = "STORE"; break;
            case DSPIR_BFLY2: opname = "BFLY2"; break;
            case DSPIR_BFLY4: opname = "BFLY4"; break;
            case DSPIR_BFLY8: opname = "BFLY8"; break;
            case DSPIR_TWIDDLE_MUL: opname = "TWIDDLE_MUL"; break;
            case DSPIR_FFT_STAGE: opname = "FFT_STAGE"; break;
            case DSPIR_LIFT_PRED: opname = "LIFT_PRED"; break;
            case DSPIR_LIFT_UPD: opname = "LIFT_UPD"; break;
            case DSPIR_LIFT_SCALE: opname = "LIFT_SCALE"; break;
            case DSPIR_REDUCE_SUM: opname = "REDUCE_SUM"; break;
            case DSPIR_REDUCE_MAX: opname = "REDUCE_MAX"; break;
            case DSPIR_REDUCE_MIN: opname = "REDUCE_MIN"; break;
            case DSPIR_CALL_BUILTIN: opname = "CALL_BUILTIN"; break;
            case DSPIR_END: opname = "END"; break;
        }
        printf("  [%2zu] %s a0=%u a1=%u a2=%u\n", i, opname, inst->a0, inst->a1, inst->a2);
        if (op == DSPIR_END) break;
    }
    if (t->n_inst > 20) {
        printf("  ... (%zu more instructions)\n", t->n_inst - 20);
    }
}

int main(void) {
    printf("=== Chirp DSL Example ===\n\n");
    
    /* Initialize library */
    dspir_init();
    
    /* Register custom functions (like OpenCL kernels or CUDA device functions) */
    printf("Registering custom builtins...\n");
    int gaussian_id = chirp_register("gaussian", (void(*)(void))my_gaussian_impl);
    int softmax_id = chirp_register("softmax", (void(*)(void))my_softmax_impl);
    int normalize_id = chirp_register("normalize", (void(*)(void))my_normalize_impl);
    
    printf("  gaussian -> id %d\n", gaussian_id);
    printf("  softmax  -> id %d\n", softmax_id);
    printf("  normalize-> id %d\n\n", normalize_id);
    
    /* Example 1: Simple FFT pipeline */
    printf("Example 1: Simple FFT Pipeline\n");
    printf("Program: (fft :size 256)\n\n");
    
    dspir_transform *t1 = chirp_compile("(fft :size 256)");
    if (t1) {
        print_ir(t1);
        printf("Transform size: %zu\n\n", t1->n);
        dspir_destroy_transform(t1);
    }
    
    /* Example 2: Complex pipeline with custom functions */
    printf("\nExample 2: Complex DSP Pipeline\n");
    printf("Program:\n");
    printf("  (pipeline\n");
    printf("    (fft :size 1024)\n");
    printf("    twiddle\n");
    printf("    (bfly 4)\n");
    printf("    (lift :predict gaussian :update softmax)\n");
    printf("    (custom softmax)\n");
    printf("    reduce-sum)\n\n");
    
    const char *complex_program = 
        "(pipeline "
        "  (fft :size 1024) "
        "  twiddle "
        "  (bfly 4) "
        "  (lift :predict gaussian :update softmax) "
        "  (custom softmax) "
        "  reduce-sum)";
    
    dspir_transform *t2 = chirp_compile(complex_program);
    if (t2) {
        print_ir(t2);
        printf("Transform size: %zu\n\n", t2->n);
        dspir_destroy_transform(t2);
    }
    
    /* Example 3: Wavelet transform */
    printf("\nExample 3: Wavelet Transform\n");
    printf("Program: (lift :predict gaussian :update normalize)\n\n");
    
    dspir_transform *t3 = chirp_compile("(lift :predict gaussian :update normalize)");
    if (t3) {
        print_ir(t3);
        dspir_destroy_transform(t3);
    }
    
    /* Example 4: Butterfly cascade */
    printf("\nExample 4: Butterfly Cascade\n");
    printf("Program: (pipeline (bfly 2) (bfly 4) (bfly 8))\n\n");
    
    dspir_transform *t4 = chirp_compile("(pipeline (bfly 2) (bfly 4) (bfly 8))");
    if (t4) {
        print_ir(t4);
        dspir_destroy_transform(t4);
    }
    
    printf("\n=== All examples completed successfully! ===\n");
    
    dspir_cleanup();
    return 0;
}
