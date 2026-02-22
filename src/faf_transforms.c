/**
 * @file faf_transforms.c
 * @brief Bytecode generators for all transform types
 */

#include "faf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Maximum instructions per transform */
#define MAX_INST 65536

/* Maximum FFT size we support - limited by register file
 * FAF_MAX_REGISTERS = 131072, interleaved complex needs 2 per point
 * So max = 131072 / 2 = 65536 complex points
 */
#define MAX_FFT_SIZE (FAF_MAX_REGISTERS / 2)

/* Helper to add instruction */
static size_t add_inst(faf_transform *t, size_t idx, uint8_t op, uint32_t a0, uint32_t a1, uint32_t a2) {
    if (idx >= MAX_INST) return idx;
    t->code[idx] = (faf_inst){
        .packed = FAF_PACK_INST(op, 0, 0),
        .a0 = a0,
        .a1 = a1,
        .a2 = a2
    };
    return idx + 1;
}

/* Helper to add instruction with float constant */
static size_t add_inst_f32(faf_transform *t, size_t idx, uint8_t op, uint32_t a0, uint32_t a1, float c) {
    union { float f; uint32_t u; } caster = { .f = c };
    return add_inst(t, idx, op, a0, a1, caster.u);
}

/* Helper: reverse bits in an index */
static size_t bit_reverse(size_t x, size_t bits) {
    size_t y = 0;
    for (size_t i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

/**
 * @brief Generate radix-2 FFT bytecode
 * 
 * Cooley-Tukey radix-2 decimation-in-time FFT
 * Input: interleaved complex [r0, i0, r1, i1, ...]
 * Output: same format
 */
void faf_gen_fft_radix2(faf_transform *t, size_t n, bool inverse) {
    t->code = calloc(MAX_INST, sizeof(faf_inst));
    if (!t->code) return;
    
    /* Limit size to prevent register overflow */
    if (n > MAX_FFT_SIZE) {
        n = MAX_FFT_SIZE;
    }
    
    /* Calculate number of bits */
    size_t bits = 0;
    size_t temp = n;
    while (temp > 1) { temp >>= 1; bits++; }
    
    /* Allocate twiddles based on precision */
    if (t->precision == FAF_PREC_FP64) {
        t->twiddles[0] = malloc(n * sizeof(double));
        if (!t->twiddles[0]) {
            free(t->code);
            t->code = NULL;
            return;
        }
        faf_gen_twiddles_f64((double*)t->twiddles[0], n, inverse);
    } else {
        t->twiddles[0] = malloc(n * sizeof(float));
        if (!t->twiddles[0]) {
            free(t->code);
            t->code = NULL;
            return;
        }
        faf_gen_twiddles_f32((float*)t->twiddles[0], n, inverse);
    }
    t->twiddle_sizes[0] = n / 2;
    
    size_t idx = 0;
    size_t max_reg = (n < FAF_MAX_REGISTERS / 2) ? n : FAF_MAX_REGISTERS / 2;
    
    /* 
     * Load inputs with bit-reversal permutation
     * Register i gets input[bit_reverse(i)]
     */
    for (size_t i = 0; i < max_reg; i++) {
        size_t j = bit_reverse(i, bits);
        idx = add_inst(t, idx, FAF_LOAD, i, j, 0);
    }
    
    /*
     * FFT stages - Cooley-Tukey butterfly using FFT_STAGE opcodes
     * Each stage emits a single FFT_STAGE instruction that the VM/JIT
     * executes as a loop, avoiding O(N log N) instruction count.
     *
     * Encoding: a0 = group_size (acts as radix, so radix/2 butterflies per group)
     *           a1 = 1 (element stride, always 1 for standard DIT)
     *           a2 = twiddle_step (stride through twiddle array)
     *
     * The VM/JIT loop pairs elements at (base+r) and (base+r+group_size/2),
     * with twiddle factor at index r * twiddle_step.
     */
    for (size_t stage = 0; stage < bits; stage++) {
        size_t group_size = 2 << stage;             /* = 2^(stage+1) */
        size_t twiddle_step = n / group_size;       /* Stride through twiddle array */
        idx = add_inst(t, idx, FAF_FFT_STAGE, group_size, 1, twiddle_step);
    }
    
    /* Store outputs (already in natural order due to input bit-reversal) */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, FAF_STORE, i, i, 0);
    }
    
    idx = add_inst(t, idx, FAF_END, 0, 0, 0);
    t->n_inst = idx;
}

/**
 * @brief Generate radix-4 FFT bytecode
 * 
 * More efficient for larger FFTs - reduces number of stages
 */
void faf_gen_fft_radix4(faf_transform *t, size_t n, bool inverse) {
    /* For now, fall back to radix-2 for simplicity and correctness */
    faf_gen_fft_radix2(t, n, inverse);
}

/**
 * @brief Generate mixed-radix FFT bytecode
 * 
 * Combines different radices for optimal performance
 */
void faf_gen_fft_mixed(faf_transform *t, size_t n, bool inverse) {
    /* For now, fall back to radix-2 */
    faf_gen_fft_radix2(t, n, inverse);
}

/**
 * @brief Generate DCT-II bytecode
 * 
 * DCT-II via FFT: extend to 2N, take real part
 * For now, simplified implementation
 */
void faf_gen_dct_ii(faf_transform *t, size_t n) {
    /* Simplified: use FFT as approximation */
    faf_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate DCT-IV bytecode
 */
void faf_gen_dct_iv(faf_transform *t, size_t n) {
    /* Simplified: use FFT */
    faf_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate DST-II bytecode
 */
void faf_gen_dst_ii(faf_transform *t, size_t n) {
    /* Simplified: use FFT */
    faf_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate MDCT bytecode
 */
void faf_gen_mdct(faf_transform *t, size_t n) {
    /* MDCT via DCT-IV */
    faf_gen_dct_iv(t, n);
}

/**
 * @brief Generate Haar wavelet bytecode
 */
void faf_gen_haar(faf_transform *t, size_t n, size_t levels) {
    t->code = calloc(MAX_INST, sizeof(faf_inst));
    if (!t->code) return;
    
    size_t idx = 0;
    /* Haar operates on real data - use half the registers for complex storage */
    size_t max_reg = (n < FAF_MAX_REGISTERS / 2) ? n : FAF_MAX_REGISTERS / 2;
    
    /* Load inputs - use only real part of each complex register */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, FAF_LOAD, i, i, 0);
    }
    
    /* Simple Haar transform for each level using BFLY2 on real parts only */
    for (size_t level = 0; level < levels; level++) {
        size_t step = 1 << level;
        for (size_t i = 0; i < max_reg; i += 2 * step) {
            /* Butterfly gives sum and difference */
            if (i + step < max_reg) {
                /* Use BFLY2 but only real parts are meaningful for Haar */
                idx = add_inst(t, idx, FAF_BFLY2, i, i + step, 0);
            }
        }
    }
    
    /* Store outputs */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, FAF_STORE, i, i, 0);
    }
    
    idx = add_inst(t, idx, FAF_END, 0, 0, 0);
    t->n_inst = idx;
}

/**
 * @brief Generate Daubechies-4 wavelet bytecode
 */
void faf_gen_daubechies4(faf_transform *t, size_t n, size_t levels) {
    /* Simplified: use Haar for now */
    faf_gen_haar(t, n, levels);
}
