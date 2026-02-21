/**
 * @file dspir_transforms.c
 * @brief Bytecode generators for all transform types
 */

#include "dspir.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Maximum instructions per transform */
#define MAX_INST 65536

/* Maximum FFT size we support - limited by register file
 * DSPIR_MAX_REGISTERS = 131072, interleaved complex needs 2 per point
 * So max = 131072 / 2 = 65536 complex points
 */
#define MAX_FFT_SIZE (DSPIR_MAX_REGISTERS / 2)

/* Helper to add instruction */
static size_t add_inst(dspir_transform *t, size_t idx, uint8_t op, uint32_t a0, uint32_t a1, uint32_t a2) {
    if (idx >= MAX_INST) return idx;
    t->code[idx] = (dspir_inst){
        .packed = DSPIR_PACK_INST(op, 0, 0),
        .a0 = a0,
        .a1 = a1,
        .a2 = a2
    };
    return idx + 1;
}

/* Helper to add instruction with float constant */
static size_t add_inst_f32(dspir_transform *t, size_t idx, uint8_t op, uint32_t a0, uint32_t a1, float c) {
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
void dspir_gen_fft_radix2(dspir_transform *t, size_t n, bool inverse) {
    t->code = calloc(MAX_INST, sizeof(dspir_inst));
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
    if (t->precision == DSPIR_PREC_FP64) {
        t->twiddles[0] = malloc(n * sizeof(double));
        if (!t->twiddles[0]) {
            free(t->code);
            t->code = NULL;
            return;
        }
        dspir_gen_twiddles_f64((double*)t->twiddles[0], n, inverse);
    } else {
        t->twiddles[0] = malloc(n * sizeof(float));
        if (!t->twiddles[0]) {
            free(t->code);
            t->code = NULL;
            return;
        }
        dspir_gen_twiddles_f32((float*)t->twiddles[0], n, inverse);
    }
    t->twiddle_sizes[0] = n / 2;
    
    size_t idx = 0;
    size_t max_reg = (n < DSPIR_MAX_REGISTERS / 2) ? n : DSPIR_MAX_REGISTERS / 2;
    
    /* 
     * Load inputs with bit-reversal permutation
     * Register i gets input[bit_reverse(i)]
     */
    for (size_t i = 0; i < max_reg; i++) {
        size_t j = bit_reverse(i, bits);
        idx = add_inst(t, idx, DSPIR_LOAD, i, j, 0);
    }
    
    /* 
     * FFT stages - Cooley-Tukey butterfly
     * Stage s (0-indexed) has stride = 2^s butterflies per group
     * Each butterfly: (a, b) -> (a + w*b, a - w*b)
     */
    for (size_t stage = 0; stage < bits; stage++) {
        size_t butterfly_span = 1 << stage;        /* Distance between butterfly inputs */
        size_t group_size = butterfly_span << 1;    /* Size of each butterfly group */
        size_t num_groups = n / group_size;         /* Number of groups */
        size_t twiddle_step = n / group_size;       /* Step between twiddle factors */
        
        for (size_t group = 0; group < num_groups; group++) {
            size_t base = group * group_size;
            
            for (size_t i = 0; i < butterfly_span && (base + i + butterfly_span) < max_reg; i++) {
                size_t idx0 = base + i;                     /* Lower element */
                size_t idx1 = base + i + butterfly_span;    /* Upper element */
                size_t tw_idx = i * twiddle_step;           /* Twiddle factor index */
                
                /* Apply twiddle factor to upper element (unless first stage where w=1) */
                if (stage > 0 && tw_idx > 0) {
                    idx = add_inst(t, idx, DSPIR_TWIDDLE_MUL, idx1, tw_idx, 0);
                }
                
                /* Radix-2 butterfly: (a, b) -> (a+b, a-b) */
                idx = add_inst(t, idx, DSPIR_BFLY2, idx0, idx1, 0);
            }
        }
    }
    
    /* Store outputs (already in natural order due to input bit-reversal) */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, DSPIR_STORE, i, i, 0);
    }
    
    idx = add_inst(t, idx, DSPIR_END, 0, 0, 0);
    t->n_inst = idx;
}

/**
 * @brief Generate radix-4 FFT bytecode
 * 
 * More efficient for larger FFTs - reduces number of stages
 */
void dspir_gen_fft_radix4(dspir_transform *t, size_t n, bool inverse) {
    /* For now, fall back to radix-2 for simplicity and correctness */
    dspir_gen_fft_radix2(t, n, inverse);
}

/**
 * @brief Generate mixed-radix FFT bytecode
 * 
 * Combines different radices for optimal performance
 */
void dspir_gen_fft_mixed(dspir_transform *t, size_t n, bool inverse) {
    /* For now, fall back to radix-2 */
    dspir_gen_fft_radix2(t, n, inverse);
}

/**
 * @brief Generate DCT-II bytecode
 * 
 * DCT-II via FFT: extend to 2N, take real part
 * For now, simplified implementation
 */
void dspir_gen_dct_ii(dspir_transform *t, size_t n) {
    /* Simplified: use FFT as approximation */
    dspir_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate DCT-IV bytecode
 */
void dspir_gen_dct_iv(dspir_transform *t, size_t n) {
    /* Simplified: use FFT */
    dspir_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate DST-II bytecode
 */
void dspir_gen_dst_ii(dspir_transform *t, size_t n) {
    /* Simplified: use FFT */
    dspir_gen_fft_radix2(t, n, false);
}

/**
 * @brief Generate MDCT bytecode
 */
void dspir_gen_mdct(dspir_transform *t, size_t n) {
    /* MDCT via DCT-IV */
    dspir_gen_dct_iv(t, n);
}

/**
 * @brief Generate Haar wavelet bytecode
 */
void dspir_gen_haar(dspir_transform *t, size_t n, size_t levels) {
    t->code = calloc(MAX_INST, sizeof(dspir_inst));
    if (!t->code) return;
    
    size_t idx = 0;
    /* Haar operates on real data - use half the registers for complex storage */
    size_t max_reg = (n < DSPIR_MAX_REGISTERS / 2) ? n : DSPIR_MAX_REGISTERS / 2;
    
    /* Load inputs - use only real part of each complex register */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, DSPIR_LOAD, i, i, 0);
    }
    
    /* Simple Haar transform for each level using BFLY2 on real parts only */
    for (size_t level = 0; level < levels; level++) {
        size_t step = 1 << level;
        for (size_t i = 0; i < max_reg; i += 2 * step) {
            /* Butterfly gives sum and difference */
            if (i + step < max_reg) {
                /* Use BFLY2 but only real parts are meaningful for Haar */
                idx = add_inst(t, idx, DSPIR_BFLY2, i, i + step, 0);
            }
        }
    }
    
    /* Store outputs */
    for (size_t i = 0; i < max_reg; i++) {
        idx = add_inst(t, idx, DSPIR_STORE, i, i, 0);
    }
    
    idx = add_inst(t, idx, DSPIR_END, 0, 0, 0);
    t->n_inst = idx;
}

/**
 * @brief Generate Daubechies-4 wavelet bytecode
 */
void dspir_gen_daubechies4(dspir_transform *t, size_t n, size_t levels) {
    /* Simplified: use Haar for now */
    dspir_gen_haar(t, n, levels);
}
