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
#ifndef isfinite
#define isfinite(x) ((x) == (x) && (x) != HUGE_VALF && (x) != -HUGE_VALF)
#endif
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

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
static void print_ir(faf_transform *t) {
    printf("Generated IR (%zu instructions):\n", t->n_inst);
    for (size_t i = 0; i < t->n_inst && i < 20; i++) {
        faf_inst *inst = &t->code[i];
        uint8_t op = FAF_GET_OP(inst->packed);
        const char *opname = "?";
        switch (op) {
            case FAF_NOP: opname = "NOP"; break;
            case FAF_LOAD: opname = "LOAD"; break;
            case FAF_STORE: opname = "STORE"; break;
            case FAF_BFLY2: opname = "BFLY2"; break;
            case FAF_BFLY4: opname = "BFLY4"; break;
            case FAF_BFLY8: opname = "BFLY8"; break;
            case FAF_TWIDDLE_MUL: opname = "TWIDDLE_MUL"; break;
            case FAF_FFT_STAGE: opname = "FFT_STAGE"; break;
            case FAF_LIFT_PRED: opname = "LIFT_PRED"; break;
            case FAF_LIFT_UPD: opname = "LIFT_UPD"; break;
            case FAF_LIFT_SCALE: opname = "LIFT_SCALE"; break;
            case FAF_REDUCE_SUM: opname = "REDUCE_SUM"; break;
            case FAF_REDUCE_MAX: opname = "REDUCE_MAX"; break;
            case FAF_REDUCE_MIN: opname = "REDUCE_MIN"; break;
            case FAF_CALL_BUILTIN: opname = "CALL_BUILTIN"; break;
            case FAF_END: opname = "END"; break;
        }
        printf("  [%2zu] %s a0=%u a1=%u a2=%u\n", i, opname, inst->a0, inst->a1, inst->a2);
        if (op == FAF_END) break;
    }
    if (t->n_inst > 20) {
        printf("  ... (%zu more instructions)\n", t->n_inst - 20);
    }
}

int main(void) {
    printf("=== Chirp DSL Example ===\n\n");
    
    /* Initialize library */
    faf_init();
    
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
    
    faf_transform *t1 = chirp_compile("(fft :size 256)");
    if (t1) {
        print_ir(t1);
        printf("Transform size: %zu\n\n", t1->n);
        faf_destroy_transform(t1);
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
    
    faf_transform *t2 = chirp_compile(complex_program);
    if (t2) {
        print_ir(t2);
        printf("Transform size: %zu\n\n", t2->n);
        faf_destroy_transform(t2);
    }
    
    /* Example 3: Wavelet transform */
    printf("\nExample 3: Wavelet Transform\n");
    printf("Program: (lift :predict gaussian :update normalize)\n\n");
    
    faf_transform *t3 = chirp_compile("(lift :predict gaussian :update normalize)");
    if (t3) {
        print_ir(t3);
        faf_destroy_transform(t3);
    }
    
    /* Example 4: Butterfly cascade */
    printf("\nExample 4: Butterfly Cascade\n");
    printf("Program: (pipeline (bfly 2) (bfly 4) (bfly 8))\n\n");
    
    faf_transform *t4 = chirp_compile("(pipeline (bfly 2) (bfly 4) (bfly 8))");
    if (t4) {
        print_ir(t4);
        faf_destroy_transform(t4);
    }
    
    /* Example 5: actually execute a Haar DWT */
    printf("\nExample 5: Execute Haar DWT\n");
    chirp_register_standard_builtins();
    faf_transform *t5 = chirp_compile("(dwt :family haar :size 32 :levels 3)");
    if (t5) {
        float in[64] = {0}, out[64] = {0};
        for (int i = 0; i < 32; i++) in[2 * i] = (i < 16) ? 0.0f : 1.0f;
        if (faf_execute_f32(t5, out, in) == 0) {
            int finite = 1;
            for (int i = 0; i < 64; i++) if (!isfinite(out[i])) finite = 0;
            printf("Executed Haar DWT: %s  approx[0]=%.4f\n",
                   finite ? "finite output" : "NON-FINITE", out[0]);
        }
        faf_destroy_transform(t5);
    }

    printf("\n=== All examples completed successfully! ===\n");
    
    chirp_cleanup();
    faf_cleanup();
    return 0;
}
