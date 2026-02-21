/**
 * @file dspir_vm.c
 * @brief Direct-threaded virtual machine implementation
 * 
 * This implements a high-performance interpreter using direct-threaded dispatch
 * (computed goto) for minimal dispatch overhead. Falls back to switch-based
 * dispatch on compilers that don't support computed goto.
 */

#include "dspir.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Enable direct-threaded dispatch on GCC/Clang */
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define DSPIR_USE_DIRECT_THREADED 1
#endif

/* Maximum register file size we can allocate on stack for small transforms */
#define DSPIR_STACK_THRESHOLD (256 * 1024)  /* 256KB */

/**
 * @brief Compute maximum register index used by a transform
 * 
 * Scans all instructions to find the highest register index referenced.
 * This allows allocating only the register file space actually needed.
 */
static size_t compute_max_registers(const dspir_transform *t) {
    size_t max_reg = 0;
    
    for (size_t i = 0; i < t->n_inst; i++) {
        const dspir_inst *inst = &t->code[i];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        
        /* Skip NOP and END */
        if (op == DSPIR_NOP || op == DSPIR_END) continue;
        
        /* Check a0, a1, a2 based on opcode type */
        if (inst->a0 > max_reg) max_reg = inst->a0;
        
        /* For opcodes that use a1 as register index */
        if (op != DSPIR_LOAD_CONST && op != DSPIR_FFT_STAGE && 
            op != DSPIR_DCT_STAGE && op != DSPIR_DST_STAGE) {
            if (inst->a1 > max_reg) max_reg = inst->a1;
        }
        
        /* For opcodes that use a2 */
        if (op == DSPIR_FMA || op == DSPIR_FMUL || op == DSPIR_FADD || 
            op == DSPIR_FSUB || op == DSPIR_POLY_FIR || op == DSPIR_POLY_IIR) {
            if (inst->a2 > max_reg) max_reg = inst->a2;
        }
    }
    
    /* Add padding for complex operations and alignment */
    /* For interleaved layout, we need 2 * (max_reg + 1) entries */
    /* Round up to multiple of 64 for cache line alignment */
    size_t num_regs = (max_reg + 1) * 2;
    num_regs = (num_regs + 63) & ~63;  /* Align to 64 */
    
    /* Minimum size for small transforms */
    if (num_regs < 64) num_regs = 64;
    
    /* Cap at maximum to prevent overflow */
    if (num_regs > DSPIR_MAX_REGISTERS) num_regs = DSPIR_MAX_REGISTERS;
    
    return num_regs;
}

/**
 * @brief Allocate aligned memory for register file
 * 
 * For small transforms, uses stack allocation.
 * For large transforms, uses heap allocation.
 */
static inline void* alloc_regs(size_t size) {
    return aligned_alloc(64, size);
}

static inline void free_regs(void* ptr) {
    free(ptr);
}

/**
 * @brief Execute transform using direct-threaded VM (FP32)
 * 
 * This is the fallback interpreter that uses direct-threaded dispatch
 * for minimal overhead. Each opcode directly jumps to its handler.
 */
int dspir_vm_execute_f32(const dspir_transform *t,
                          float *restrict out,
                          const float *restrict in) {
    if (!t || !out || !in) return -1;
    if (!t->code || t->n_inst == 0) return -1;
    
#ifdef DSPIR_USE_DIRECT_THREADED
    /* Direct-threaded dispatch table - 256 entries for fast indexing
     * Unused opcodes (31-254) map to END handler for safety
     */
    #define DT_END_8 &&label_END, &&label_END, &&label_END, &&label_END, &&label_END, &&label_END, &&label_END, &&label_END
    #define DT_END_32 DT_END_8, DT_END_8, DT_END_8, DT_END_8
    #define DT_END_128 DT_END_32, DT_END_32, DT_END_32, DT_END_32
    #define DT_END_224 DT_END_128, DT_END_32, DT_END_32, DT_END_32
    static const void *dispatch_table[256] = {
        /* 0-30: Active opcodes */
        &&label_NOP, &&label_LOAD, &&label_STORE, &&label_LOAD_CONST,
        &&label_BFLY2, &&label_BFLY4, &&label_BFLY8,
        &&label_TWIDDLE_MUL, &&label_TWIDDLE_MUL_CONJ,
        &&label_FMA, &&label_FMUL, &&label_FADD, &&label_FSUB,
        &&label_BARRIER, &&label_FFT_STAGE, &&label_DCT_STAGE,
        &&label_DST_STAGE, &&label_LIFT_PRED, &&label_LIFT_UPD,
        &&label_LIFT_SCALE, &&label_DOWN2, &&label_UP2,
        &&label_POLY_FIR, &&label_POLY_IIR, &&label_PERMUTE,
        &&label_SHUFFLE, &&label_BLEND, &&label_REDUCE_SUM,
        &&label_REDUCE_MAX, &&label_REDUCE_MIN, &&label_END,
        /* 31-254: Unused opcodes -> END handler */
        DT_END_224,
        /* 255: DSPIR_END */
        &&label_END
    };
    #undef DT_END_8
    #undef DT_END_32
    #undef DT_END_128
    #undef DT_END_224
#endif

    /* Compute and allocate register file based on actual transform needs */
    size_t num_regs = compute_max_registers(t);
    float *regs = (float*)alloc_regs(num_regs * sizeof(float));
    if (!regs) return -1;
    memset(regs, 0, num_regs * sizeof(float));
    
    /* Twiddle pointer - may be NULL for transforms that don't need twiddles */
    const float *tw = (const float *)t->twiddles[0];
    
    /* Instruction pointer */
    size_t ip = 0;
    
    /* Current instruction */
    const dspir_inst *inst;
    uint8_t op;
    
#ifdef DSPIR_USE_DIRECT_THREADED
    /* Direct-threaded: fetch first opcode and dispatch */
    inst = &t->code[ip++];
    op = DSPIR_GET_OP(inst->packed);
    goto *dispatch_table[op];
#else
    /* Switch-based dispatch */
    while (ip < t->n_inst) {
        inst = &t->code[ip++];
        op = DSPIR_GET_OP(inst->packed);
        switch (op) {
#endif

#ifdef DSPIR_USE_DIRECT_THREADED
#define DISPATCH() do { \
    inst = &t->code[ip++]; \
    op = DSPIR_GET_OP(inst->packed); \
    goto *dispatch_table[op]; \
} while(0)
#else
#define DISPATCH() break
#endif

#ifdef DSPIR_USE_DIRECT_THREADED
label_NOP:
    DISPATCH();
#else
    case DSPIR_NOP:
        DISPATCH();
#endif

#ifdef DSPIR_USE_DIRECT_THREADED
label_LOAD:
#else
    case DSPIR_LOAD:
#endif
        /* Load complex value from interleaved input
         * a0 = register index 
         * a1 = input index (complex element index)
         * Both input and register file are interleaved: [r0, i0, r1, i1, ...]
         */
        {
            size_t in_idx = inst->a1;
            size_t reg_idx = inst->a0 * 2;  /* Interleaved: real at even, imag at odd */
            if (reg_idx + 1 < DSPIR_MAX_REGISTERS) {
                regs[reg_idx] = in[in_idx * 2];      /* Real part */
                regs[reg_idx + 1] = in[in_idx * 2 + 1];  /* Imag part */
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_STORE:
#else
    case DSPIR_STORE:
#endif
        /* Store complex value to interleaved output
         * a0 = output index (complex element index)
         * a1 = register index
         * Register file is interleaved: regs[2*i]=ri, regs[2*i+1]=ii
         */
        {
            size_t out_idx = inst->a0;
            size_t reg_idx = inst->a1 * 2;  /* Interleaved */
            if (reg_idx + 1 < DSPIR_MAX_REGISTERS) {
                out[out_idx * 2] = regs[reg_idx];      /* Real part */
                out[out_idx * 2 + 1] = regs[reg_idx + 1];  /* Imag part */
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_LOAD_CONST:
#else
    case DSPIR_LOAD_CONST:
#endif
        /* a1 contains the bits of a float constant */
        if (inst->a0 < DSPIR_MAX_REGISTERS) {
            union { uint32_t u; float f; } caster;
            caster.u = inst->a1;
            regs[inst->a0] = caster.f;
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_BFLY2:
#else
    case DSPIR_BFLY2:
#endif
        {
            /* Complex radix-2 butterfly
             * Register file is interleaved: regs[2*i]=ri, regs[2*i+1]=ii
             * a0, a1 are register indices (not memory offsets)
             * Input: regs[2*a0]=ar, regs[2*a0+1]=ai, regs[2*a1]=br, regs[2*a1+1]=bi
             * Output: (ar+br, ai+bi) at a0, (ar-br, ai-bi) at a1
             */
            size_t ra = inst->a0 * 2;
            size_t rb = inst->a1 * 2;
            if (ra + 1 < DSPIR_MAX_REGISTERS && rb + 1 < DSPIR_MAX_REGISTERS) {
                float ar = regs[ra];
                float ai = regs[ra + 1];
                float br = regs[rb];
                float bi = regs[rb + 1];
                
                regs[ra] = ar + br;
                regs[ra + 1] = ai + bi;
                regs[rb] = ar - br;
                regs[rb + 1] = ai - bi;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_BFLY4:
#else
    case DSPIR_BFLY4:
#endif
        {
            /* Radix-4 butterfly for complex data
             * Register file is interleaved: regs[2*i]=real, regs[2*i+1]=imag
             * a0 = base register index (complex element index)
             * Operates on 4 consecutive complex elements at base, base+1, base+2, base+3
             */
            uint32_t base = inst->a0;
            if (base * 2 + 7 < DSPIR_MAX_REGISTERS) {
                float r0 = regs[base * 2],     i0 = regs[base * 2 + 1];
                float r1 = regs[base * 2 + 2], i1 = regs[base * 2 + 3];
                float r2 = regs[base * 2 + 4], i2 = regs[base * 2 + 5];
                float r3 = regs[base * 2 + 6], i3 = regs[base * 2 + 7];

                /* Stage 1: Two radix-2 butterflies */
                float t0r = r0 + r2, t0i = i0 + i2;
                float t1r = r0 - r2, t1i = i0 - i2;
                float t2r = r1 + r3, t2i = i1 + i3;
                float t3r = r1 - r3, t3i = i1 - i3;

                /* Stage 2: Final butterfly with twiddle (j multiplication) */
                regs[base * 2]     = t0r + t2r;
                regs[base * 2 + 1] = t0i + t2i;
                regs[base * 2 + 2] = t1r + t3i;  /* t3 * j */
                regs[base * 2 + 3] = t1i - t3r;
                regs[base * 2 + 4] = t0r - t2r;
                regs[base * 2 + 5] = t0i - t2i;
                regs[base * 2 + 6] = t1r - t3i;
                regs[base * 2 + 7] = t1i + t3r;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_BFLY8:
#else
    case DSPIR_BFLY8:
#endif
        {
            /* Radix-8 butterfly for complex data
             * Register file is interleaved: regs[2*i]=real, regs[2*i+1]=imag
             * a0 = base register index (complex element index)
             * Operates on 8 consecutive complex elements
             */
            uint32_t base = inst->a0;
            if (base * 2 + 15 < DSPIR_MAX_REGISTERS) {
                /* Load 8 complex values from interleaved layout */
                float r[8], im[8];
                for (int k = 0; k < 8; k++) {
                    r[k]  = regs[base * 2 + 2 * k];
                    im[k] = regs[base * 2 + 2 * k + 1];
                }

                /* 8-point DFT using Cooley-Tukey */
                /* Stage 1: 4 radix-2 butterflies */
                float s1r[8], s1i[8];
                for (int k = 0; k < 4; k++) {
                    s1r[k]     = r[k] + r[k + 4];
                    s1i[k]     = im[k] + im[k + 4];
                    s1r[k + 4] = r[k] - r[k + 4];
                    s1i[k + 4] = im[k] - im[k + 4];
                }

                /* Stage 2: 2 radix-4 butterflies with twiddles */
                float s2r[8], s2i[8];
                /* First radix-4 (no twiddle) */
                s2r[0] = s1r[0] + s1r[2]; s2i[0] = s1i[0] + s1i[2];
                s2r[1] = s1r[1] + s1i[3]; s2i[1] = s1i[1] - s1r[3];
                s2r[2] = s1r[0] - s1r[2]; s2i[2] = s1i[0] - s1i[2];
                s2r[3] = s1r[1] - s1i[3]; s2i[3] = s1i[1] + s1r[3];

                /* Second radix-4 (with W_8 twiddles) */
                const float c = 0.7071067811865476f; /* cos(pi/4) */
                float t4r = c * (s1r[4] + s1i[4]);
                float t4i = c * (s1i[4] - s1r[4]);
                float t5r = -s1i[5];
                float t5i = s1r[5];
                float t6r = c * (s1i[6] - s1r[6]);
                float t6i = -c * (s1r[6] + s1i[6]);
                float t7r = s1r[7];
                float t7i = s1i[7];

                s2r[4] = t4r + t6r; s2i[4] = t4i + t6i;
                s2r[5] = t5r + t7i; s2i[5] = t5i - t7r;
                s2r[6] = t4r - t6r; s2i[6] = t4i - t6i;
                s2r[7] = t5r - t7i; s2i[7] = t5i + t7r;

                /* Stage 3: Final combination — write back interleaved */
                for (int k = 0; k < 4; k++) {
                    regs[base * 2 + 4 * k]     = s2r[k] + s2r[k + 4];
                    regs[base * 2 + 4 * k + 1] = s2i[k] + s2i[k + 4];
                    regs[base * 2 + 4 * k + 2] = s2r[k] - s2r[k + 4];
                    regs[base * 2 + 4 * k + 3] = s2i[k] - s2i[k + 4];
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_TWIDDLE_MUL:
#else
    case DSPIR_TWIDDLE_MUL:
#endif
        {
            /* Complex multiplication: (a+ib) * (c+id) = (ac-bd) + i(ad+bc) */
            size_t ra = inst->a0 * 2;
            if (ra + 1 < DSPIR_MAX_REGISTERS) {
                float re = regs[ra];
                float im = regs[ra + 1];
                if (tw) {
                    float wr = tw[inst->a1 * 2];
                    float wi = tw[inst->a1 * 2 + 1];
                    regs[ra] = re * wr - im * wi;
                    regs[ra + 1] = re * wi + im * wr;
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_TWIDDLE_MUL_CONJ:
#else
    case DSPIR_TWIDDLE_MUL_CONJ:
#endif
        {
            /* Complex multiplication by conjugate: (a+ib) * (c-id) = (ac+bd) + i(bc-ad) */
            size_t ra = inst->a0 * 2;
            if (ra + 1 < DSPIR_MAX_REGISTERS) {
                float re = regs[ra];
                float im = regs[ra + 1];
                if (tw) {
                    float wr = tw[inst->a1 * 2];
                    float wi = -tw[inst->a1 * 2 + 1];  /* Conjugate */
                    regs[ra] = re * wr - im * wi;
                    regs[ra + 1] = re * wi + im * wr;
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_FMA:
#else
    case DSPIR_FMA:
#endif
        if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
            regs[inst->a0] = regs[inst->a0] * regs[inst->a1] + regs[inst->a2];
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_FMUL:
#else
    case DSPIR_FMUL:
#endif
        if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
            regs[inst->a0] = regs[inst->a1] * regs[inst->a2];
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_FADD:
#else
    case DSPIR_FADD:
#endif
        if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
            regs[inst->a0] = regs[inst->a1] + regs[inst->a2];
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_FSUB:
#else
    case DSPIR_FSUB:
#endif
        if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
            regs[inst->a0] = regs[inst->a1] - regs[inst->a2];
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_BARRIER:
#else
    case DSPIR_BARRIER:
#endif
        /* Memory barrier / prefetch hint */
        __asm__ volatile("" ::: "memory");
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_FFT_STAGE:
#else
    case DSPIR_FFT_STAGE:
#endif
        {
            /* Execute complete FFT stage with complex butterflies
             * Register file is interleaved: regs[2*i]=real, regs[2*i+1]=imag
             */
            size_t radix = inst->a0;
            size_t stride = inst->a1;
            size_t tw_base = inst->a2;
            size_t ngroups = t->n / (radix * stride);

            for (size_t g = 0; g < ngroups; g++) {
                size_t base = g * radix * stride;

                for (size_t r = 0; r < radix / 2; r++) {
                    size_t idx1 = base + r * stride;
                    size_t idx2 = base + (r + radix / 2) * stride;

                    /* Read complex values from interleaved layout */
                    float ar = regs[idx1 * 2];
                    float ai = regs[idx1 * 2 + 1];
                    float br = regs[idx2 * 2];
                    float bi = regs[idx2 * 2 + 1];

                    if (tw && tw_base + r < t->twiddle_sizes[0]) {
                        float wr = tw[(tw_base + r) * 2];
                        float wi = tw[(tw_base + r) * 2 + 1];
                        /* Complex multiply: (br + i*bi) * (wr + i*wi) */
                        float tw_re = br * wr - bi * wi;
                        float tw_im = br * wi + bi * wr;
                        regs[idx1 * 2]     = ar + tw_re;
                        regs[idx1 * 2 + 1] = ai + tw_im;
                        regs[idx2 * 2]     = ar - tw_re;
                        regs[idx2 * 2 + 1] = ai - tw_im;
                    } else {
                        regs[idx1 * 2]     = ar + br;
                        regs[idx1 * 2 + 1] = ai + bi;
                        regs[idx2 * 2]     = ar - br;
                        regs[idx2 * 2 + 1] = ai - bi;
                    }
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_DCT_STAGE:
#else
    case DSPIR_DCT_STAGE:
#endif
        {
            /* DCT stage using lifting or direct computation */
            size_t stride = inst->a1;
            size_t nstage = t->n / stride;
            
            for (size_t i = 0; i < nstage / 2; i++) {
                size_t j = nstage - 1 - i;
                if (i * stride < DSPIR_MAX_REGISTERS && j * stride < DSPIR_MAX_REGISTERS) {
                    float a = regs[i * stride];
                    float b = regs[j * stride];
                    
                    /* DCT-II butterfly */
                    regs[i * stride] = a + b;
                    regs[j * stride] = (a - b) * tw[i];
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_DST_STAGE:
#else
    case DSPIR_DST_STAGE:
#endif
        {
            /* DST stage — differs from DCT by negated twiddle for alternating elements */
            size_t stride = inst->a1;
            size_t nstage = t->n / stride;

            for (size_t i = 0; i < nstage / 2; i++) {
                size_t j = nstage - 1 - i;
                if (i * stride < DSPIR_MAX_REGISTERS && j * stride < DSPIR_MAX_REGISTERS) {
                    float a = regs[i * stride];
                    float b = regs[j * stride];

                    /* DST-II butterfly — negate twiddle for sine basis */
                    regs[i * stride] = a + b;
                    regs[j * stride] = (a - b) * (-tw[i]);
                }
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_LIFT_PRED:
#else
    case DSPIR_LIFT_PRED:
#endif
        {
            /* Lifting scheme prediction step */
            /* a2 contains the coefficient as raw bits */
            if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS) {
                union { uint32_t u; float f; } c;
                c.u = inst->a2;
                regs[inst->a0] += regs[inst->a1] * c.f;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_LIFT_UPD:
#else
    case DSPIR_LIFT_UPD:
#endif
        {
            /* Lifting scheme update step */
            if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS) {
                union { uint32_t u; float f; } c;
                c.u = inst->a2;
                regs[inst->a0] += regs[inst->a1] * c.f;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_LIFT_SCALE:
#else
    case DSPIR_LIFT_SCALE:
#endif
        {
            /* Lifting scheme scaling step */
            if (inst->a0 < DSPIR_MAX_REGISTERS) {
                union { uint32_t u; float f; } c;
                c.u = inst->a2;
                regs[inst->a0] *= c.f;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_DOWN2:
#else
    case DSPIR_DOWN2:
#endif
        {
            /* Downsampling by 2 */
            size_t offset = inst->a0;
            for (size_t j = 0; j < t->n / 2; j++) {
                if (j < DSPIR_MAX_REGISTERS && j * 2 + offset < DSPIR_MAX_REGISTERS)
                    regs[j] = regs[j * 2 + offset];
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_UP2:
#else
    case DSPIR_UP2:
#endif
        {
            /* Upsampling by 2 */
            size_t offset = inst->a0;
            for (size_t j = t->n / 2; j > 0; j--) {
                if (j * 2 - 1 + offset < DSPIR_MAX_REGISTERS && j - 1 < DSPIR_MAX_REGISTERS)
                    regs[j * 2 - 1 + offset] = regs[j - 1];
                if (j * 2 - 2 + offset < DSPIR_MAX_REGISTERS)
                    regs[j * 2 - 2 + offset] = 0;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_POLY_FIR:
#else
    case DSPIR_POLY_FIR:
#endif
        {
            /* Polyphase FIR filter */
            size_t taps = inst->a0;
            size_t stride = inst->a1;
            size_t nch = 8;  /* 8 channels */
            
            for (size_t ch = 0; ch < nch; ch++) {
                float acc = 0.0f;
                for (size_t k = 0; k < taps; k++) {
                    acc += in[ch * stride + k] * tw[k];
                }
                out[ch * stride] = acc;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_POLY_IIR:
#else
    case DSPIR_POLY_IIR:
#endif
        {
            /* Polyphase IIR filter (simplified) */
            size_t taps = inst->a0;
            size_t stride = inst->a1;
            (void)taps;
            (void)stride;
            /* IIR implementation would use feedback */
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_PERMUTE:
#else
    case DSPIR_PERMUTE:
#endif
        {
            /* Register permutation */
            uint32_t pattern = inst->a2;
            float temp[4];
            for (int k = 0; k < 4; k++) {
                int src = (pattern >> (k * 8)) & 0xFF;
                if (inst->a0 + src < DSPIR_MAX_REGISTERS)
                    temp[k] = regs[inst->a0 + src];
            }
            for (int k = 0; k < 4; k++) {
                if (inst->a0 + k < DSPIR_MAX_REGISTERS)
                    regs[inst->a0 + k] = temp[k];
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_SHUFFLE:
#else
    case DSPIR_SHUFFLE:
#endif
        {
            /* Vector shuffle */
            /* a2 contains shuffle mask */
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_BLEND:
#else
    case DSPIR_BLEND:
#endif
        {
            /* Vector blend */
            /* a2 contains blend mask */
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_REDUCE_SUM:
#else
    case DSPIR_REDUCE_SUM:
#endif
        {
            /* Horizontal sum reduction */
            float sum = 0.0f;
            for (size_t k = 0; k < inst->a1; k++) {
                if (inst->a0 + k < DSPIR_MAX_REGISTERS)
                    sum += regs[inst->a0 + k];
            }
            if (inst->a0 < DSPIR_MAX_REGISTERS)
                regs[inst->a0] = sum;
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_REDUCE_MAX:
#else
    case DSPIR_REDUCE_MAX:
#endif
        {
            /* Horizontal max reduction */
            if (inst->a0 < DSPIR_MAX_REGISTERS) {
                float max_val = regs[inst->a0];
                for (size_t k = 1; k < inst->a1; k++) {
                    if (inst->a0 + k < DSPIR_MAX_REGISTERS && regs[inst->a0 + k] > max_val) {
                        max_val = regs[inst->a0 + k];
                    }
                }
                regs[inst->a0] = max_val;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_REDUCE_MIN:
#else
    case DSPIR_REDUCE_MIN:
#endif
        {
            /* Horizontal min reduction */
            if (inst->a0 < DSPIR_MAX_REGISTERS) {
                float min_val = regs[inst->a0];
                for (size_t k = 1; k < inst->a1; k++) {
                    if (inst->a0 + k < DSPIR_MAX_REGISTERS && regs[inst->a0 + k] < min_val) {
                        min_val = regs[inst->a0 + k];
                    }
                }
                regs[inst->a0] = min_val;
            }
        }
        DISPATCH();

#ifdef DSPIR_USE_DIRECT_THREADED
label_END:
#else
    case DSPIR_END:
#endif
        goto done;

#ifndef DSPIR_USE_DIRECT_THREADED
        default:
            /* Unknown opcode - skip */
            break;
        }  /* end switch */
    }  /* end while */
#endif

done:
    /* Cleanup */
    free_regs(regs);
    return 0;
}

/* Wrapper for core API compatibility - uses split-plane by default */
int dspir_execute_f32(const dspir_transform *t,
                      float *restrict out,
                      const float *restrict in) {
    /* Use split-plane execution for better performance */
    const size_t n = t->n;
    float *in_re = (float*)alloc_regs(n * sizeof(float));
    float *in_im = (float*)alloc_regs(n * sizeof(float));
    float *out_re = (float*)alloc_regs(n * sizeof(float));
    float *out_im = (float*)alloc_regs(n * sizeof(float));
    
    if (!in_re || !in_im || !out_re || !out_im) {
        free_regs(in_re); free_regs(in_im);
        free_regs(out_re); free_regs(out_im);
        return dspir_vm_execute_f32(t, out, in);  /* Fallback to interleaved */
    }
    
    /* Deinterleave input */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = in[i * 2];
        in_im[i] = in[i * 2 + 1];
    }
    
    /* Execute split-plane */
    int result = dspir_execute_split_f32(t, out_re, out_im, in_re, in_im);
    
    /* Reinterleave output */
    if (result == 0) {
        for (size_t i = 0; i < n; i++) {
            out[i * 2] = out_re[i];
            out[i * 2 + 1] = out_im[i];
        }
    }
    
    free_regs(in_re); free_regs(in_im);
    free_regs(out_re); free_regs(out_im);
    return result;
}

/**
 * @brief FP64 VM execution
 */
int dspir_vm_execute_f64(const dspir_transform *t,
                          double *restrict out,
                          const double *restrict in) {
    if (!t || !out || !in) return -1;
    if (!t->code || t->n_inst == 0) return -1;
    
    /* Compute and allocate register file based on actual transform needs */
    size_t num_regs = compute_max_registers(t);
    double *regs = (double*)alloc_regs(num_regs * sizeof(double));
    if (!regs) return -1;
    memset(regs, 0, num_regs * sizeof(double));
    
    const double *tw = (const double *)t->twiddles[0];
    
    for (size_t ip = 0; ip < t->n_inst; ip++) {
        const dspir_inst *inst = &t->code[ip];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        
        switch (op) {
            case DSPIR_LOAD: {
                size_t in_idx = inst->a1;
                size_t reg_idx = inst->a0 * 2;
                if (reg_idx + 1 < DSPIR_MAX_REGISTERS) {
                    regs[reg_idx] = in[in_idx * 2];
                    regs[reg_idx + 1] = in[in_idx * 2 + 1];
                }
                break;
            }
            case DSPIR_STORE: {
                size_t out_idx = inst->a0;
                size_t reg_idx = inst->a1 * 2;
                if (reg_idx + 1 < DSPIR_MAX_REGISTERS) {
                    out[out_idx * 2] = regs[reg_idx];
                    out[out_idx * 2 + 1] = regs[reg_idx + 1];
                }
                break;
            }
            case DSPIR_LOAD_CONST: {
                if (inst->a0 < DSPIR_MAX_REGISTERS) {
                    union { uint32_t u; float f; } caster;
                    caster.u = inst->a1;
                    regs[inst->a0] = (double)caster.f;
                }
                break;
            }
            case DSPIR_BFLY2: {
                size_t ra = inst->a0 * 2;
                size_t rb = inst->a1 * 2;
                if (ra + 1 < DSPIR_MAX_REGISTERS && rb + 1 < DSPIR_MAX_REGISTERS) {
                    double ar = regs[ra], ai = regs[ra + 1];
                    double br = regs[rb], bi = regs[rb + 1];
                    regs[ra] = ar + br; regs[ra + 1] = ai + bi;
                    regs[rb] = ar - br; regs[rb + 1] = ai - bi;
                }
                break;
            }
            case DSPIR_BFLY4: {
                uint32_t base = inst->a0;
                if (base * 2 + 7 < DSPIR_MAX_REGISTERS) {
                    double r0 = regs[base * 2],     i0 = regs[base * 2 + 1];
                    double r1 = regs[base * 2 + 2], i1 = regs[base * 2 + 3];
                    double r2 = regs[base * 2 + 4], i2 = regs[base * 2 + 5];
                    double r3 = regs[base * 2 + 6], i3 = regs[base * 2 + 7];
                    double t0r = r0 + r2, t0i = i0 + i2;
                    double t1r = r0 - r2, t1i = i0 - i2;
                    double t2r = r1 + r3, t2i = i1 + i3;
                    double t3r = r1 - r3, t3i = i1 - i3;
                    regs[base * 2]     = t0r + t2r;
                    regs[base * 2 + 1] = t0i + t2i;
                    regs[base * 2 + 2] = t1r + t3i;
                    regs[base * 2 + 3] = t1i - t3r;
                    regs[base * 2 + 4] = t0r - t2r;
                    regs[base * 2 + 5] = t0i - t2i;
                    regs[base * 2 + 6] = t1r - t3i;
                    regs[base * 2 + 7] = t1i + t3r;
                }
                break;
            }
            case DSPIR_BFLY8: {
                uint32_t base = inst->a0;
                if (base * 2 + 15 < DSPIR_MAX_REGISTERS) {
                    double r[8], im[8];
                    for (int k = 0; k < 8; k++) {
                        r[k]  = regs[base * 2 + 2 * k];
                        im[k] = regs[base * 2 + 2 * k + 1];
                    }
                    double s1r[8], s1i[8];
                    for (int k = 0; k < 4; k++) {
                        s1r[k]     = r[k] + r[k + 4];
                        s1i[k]     = im[k] + im[k + 4];
                        s1r[k + 4] = r[k] - r[k + 4];
                        s1i[k + 4] = im[k] - im[k + 4];
                    }
                    double s2r[8], s2i[8];
                    s2r[0] = s1r[0] + s1r[2]; s2i[0] = s1i[0] + s1i[2];
                    s2r[1] = s1r[1] + s1i[3]; s2i[1] = s1i[1] - s1r[3];
                    s2r[2] = s1r[0] - s1r[2]; s2i[2] = s1i[0] - s1i[2];
                    s2r[3] = s1r[1] - s1i[3]; s2i[3] = s1i[1] + s1r[3];
                    const double c = 0.7071067811865476;
                    double t4r = c * (s1r[4] + s1i[4]);
                    double t4i = c * (s1i[4] - s1r[4]);
                    double t5r = -s1i[5];
                    double t5i = s1r[5];
                    double t6r = c * (s1i[6] - s1r[6]);
                    double t6i = -c * (s1r[6] + s1i[6]);
                    double t7r = s1r[7];
                    double t7i = s1i[7];
                    s2r[4] = t4r + t6r; s2i[4] = t4i + t6i;
                    s2r[5] = t5r + t7i; s2i[5] = t5i - t7r;
                    s2r[6] = t4r - t6r; s2i[6] = t4i - t6i;
                    s2r[7] = t5r - t7i; s2i[7] = t5i + t7r;
                    for (int k = 0; k < 4; k++) {
                        regs[base * 2 + 4 * k]     = s2r[k] + s2r[k + 4];
                        regs[base * 2 + 4 * k + 1] = s2i[k] + s2i[k + 4];
                        regs[base * 2 + 4 * k + 2] = s2r[k] - s2r[k + 4];
                        regs[base * 2 + 4 * k + 3] = s2i[k] - s2i[k + 4];
                    }
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL: {
                size_t ra = inst->a0 * 2;
                if (ra + 1 < DSPIR_MAX_REGISTERS && tw) {
                    double re = regs[ra], im = regs[ra + 1];
                    double wr = tw[inst->a1 * 2], wi = tw[inst->a1 * 2 + 1];
                    regs[ra] = re * wr - im * wi;
                    regs[ra + 1] = re * wi + im * wr;
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL_CONJ: {
                size_t ra = inst->a0 * 2;
                if (ra + 1 < DSPIR_MAX_REGISTERS && tw) {
                    double re = regs[ra], im = regs[ra + 1];
                    double wr = tw[inst->a1 * 2], wi = -tw[inst->a1 * 2 + 1];
                    regs[ra] = re * wr - im * wi;
                    regs[ra + 1] = re * wi + im * wr;
                }
                break;
            }
            case DSPIR_FMA:
                if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
                    regs[inst->a0] = regs[inst->a0] * regs[inst->a1] + regs[inst->a2];
                break;
            case DSPIR_FMUL:
                if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
                    regs[inst->a0] = regs[inst->a1] * regs[inst->a2];
                break;
            case DSPIR_FADD:
                if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
                    regs[inst->a0] = regs[inst->a1] + regs[inst->a2];
                break;
            case DSPIR_FSUB:
                if (inst->a0 < DSPIR_MAX_REGISTERS && inst->a1 < DSPIR_MAX_REGISTERS && inst->a2 < DSPIR_MAX_REGISTERS)
                    regs[inst->a0] = regs[inst->a1] - regs[inst->a2];
                break;
            case DSPIR_END:
                goto done_f64;
            default:
                break;
        }
    }
    
done_f64:
    free_regs(regs);
    return 0;
}

/* Wrapper for core API compatibility - uses split-plane by default */
int dspir_execute_f64(const dspir_transform *t,
                      double *restrict out,
                      const double *restrict in) {
    /* Use split-plane execution for better performance */
    const size_t n = t->n;
    double *in_re = (double*)alloc_regs(n * sizeof(double));
    double *in_im = (double*)alloc_regs(n * sizeof(double));
    double *out_re = (double*)alloc_regs(n * sizeof(double));
    double *out_im = (double*)alloc_regs(n * sizeof(double));
    
    if (!in_re || !in_im || !out_re || !out_im) {
        free_regs(in_re); free_regs(in_im);
        free_regs(out_re); free_regs(out_im);
        return dspir_vm_execute_f64(t, out, in);  /* Fallback to interleaved */
    }
    
    /* Deinterleave input */
    for (size_t i = 0; i < n; i++) {
        in_re[i] = in[i * 2];
        in_im[i] = in[i * 2 + 1];
    }
    
    /* Execute split-plane */
    int result = dspir_execute_split_f64(t, out_re, out_im, in_re, in_im);
    
    /* Reinterleave output */
    if (result == 0) {
        for (size_t i = 0; i < n; i++) {
            out[i * 2] = out_re[i];
            out[i * 2 + 1] = out_im[i];
        }
    }
    
    free_regs(in_re); free_regs(in_im);
    free_regs(out_re); free_regs(out_im);
    return result;
}

/**
 * @brief Split-plane FP32 execution (separate real/imag arrays)
 * 
 * This is more efficient for SIMD on x86 with AVX-512.
 * Input/output are separate real and imaginary arrays instead of interleaved.
 */
int dspir_execute_split_f32(const dspir_transform *t,
                             float *restrict out_re,
                             float *restrict out_im,
                             const float *restrict in_re,
                             const float *restrict in_im) {
    if (!t || !out_re || !out_im || !in_re || !in_im) return -1;
    if (!t->code || t->n_inst == 0) return -1;
    
    /* Allocate separate register planes */
    float *regs_re = (float*)alloc_regs(t->n * sizeof(float));
    float *regs_im = (float*)alloc_regs(t->n * sizeof(float));
    if (!regs_re || !regs_im) {
        free_regs(regs_re);
        free_regs(regs_im);
        return -1;
    }
    memset(regs_re, 0, t->n * sizeof(float));
    memset(regs_im, 0, t->n * sizeof(float));
    
    const float *tw = (const float *)t->twiddles[0];
    
    for (size_t ip = 0; ip < t->n_inst; ip++) {
        const dspir_inst *inst = &t->code[ip];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        
        switch (op) {
            case DSPIR_NOP:
                break;
            case DSPIR_LOAD: {
                size_t in_idx = inst->a1;
                size_t reg_idx = inst->a0;
                if (reg_idx < t->n && in_idx < t->n) {
                    regs_re[reg_idx] = in_re[in_idx];
                    regs_im[reg_idx] = in_im[in_idx];
                }
                break;
            }
            case DSPIR_STORE: {
                size_t out_idx = inst->a0;
                size_t reg_idx = inst->a1;
                if (out_idx < t->n && reg_idx < t->n) {
                    out_re[out_idx] = regs_re[reg_idx];
                    out_im[out_idx] = regs_im[reg_idx];
                }
                break;
            }
            case DSPIR_LOAD_CONST: {
                if (inst->a0 < t->n) {
                    union { uint32_t u; float f; } caster;
                    caster.u = inst->a1;
                    regs_re[inst->a0] = caster.f;
                }
                break;
            }
            case DSPIR_BFLY2: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && a1 < t->n) {
                    float ar = regs_re[a0], ai = regs_im[a0];
                    float br = regs_re[a1], bi = regs_im[a1];
                    regs_re[a0] = ar + br; regs_im[a0] = ai + bi;
                    regs_re[a1] = ar - br; regs_im[a1] = ai - bi;
                }
                break;
            }
            case DSPIR_BFLY4: {
                uint32_t base = inst->a0;
                if (base + 3 < t->n) {
                    float r0 = regs_re[base],     i0 = regs_im[base];
                    float r1 = regs_re[base + 1], i1 = regs_im[base + 1];
                    float r2 = regs_re[base + 2], i2 = regs_im[base + 2];
                    float r3 = regs_re[base + 3], i3 = regs_im[base + 3];
                    float t0r = r0 + r2, t0i = i0 + i2;
                    float t1r = r0 - r2, t1i = i0 - i2;
                    float t2r = r1 + r3, t2i = i1 + i3;
                    float t3r = r1 - r3, t3i = i1 - i3;
                    regs_re[base]     = t0r + t2r; regs_im[base]     = t0i + t2i;
                    regs_re[base + 1] = t1r + t3i; regs_im[base + 1] = t1i - t3r;
                    regs_re[base + 2] = t0r - t2r; regs_im[base + 2] = t0i - t2i;
                    regs_re[base + 3] = t1r - t3i; regs_im[base + 3] = t1i + t3r;
                }
                break;
            }
            case DSPIR_BFLY8: {
                uint32_t base = inst->a0;
                if (base + 7 < t->n) {
                    float r[8], im[8];
                    for (int k = 0; k < 8; k++) {
                        r[k]  = regs_re[base + k];
                        im[k] = regs_im[base + k];
                    }
                    float s1r[8], s1i[8];
                    for (int k = 0; k < 4; k++) {
                        s1r[k]     = r[k] + r[k + 4]; s1i[k]     = im[k] + im[k + 4];
                        s1r[k + 4] = r[k] - r[k + 4]; s1i[k + 4] = im[k] - im[k + 4];
                    }
                    float s2r[8], s2i[8];
                    s2r[0] = s1r[0] + s1r[2]; s2i[0] = s1i[0] + s1i[2];
                    s2r[1] = s1r[1] + s1i[3]; s2i[1] = s1i[1] - s1r[3];
                    s2r[2] = s1r[0] - s1r[2]; s2i[2] = s1i[0] - s1i[2];
                    s2r[3] = s1r[1] - s1i[3]; s2i[3] = s1i[1] + s1r[3];
                    const float c = 0.7071067811865476f;
                    float t4r = c * (s1r[4] + s1i[4]), t4i = c * (s1i[4] - s1r[4]);
                    float t5r = -s1i[5], t5i = s1r[5];
                    float t6r = c * (s1i[6] - s1r[6]), t6i = -c * (s1r[6] + s1i[6]);
                    float t7r = s1r[7], t7i = s1i[7];
                    s2r[4] = t4r + t6r; s2i[4] = t4i + t6i;
                    s2r[5] = t5r + t7i; s2i[5] = t5i - t7r;
                    s2r[6] = t4r - t6r; s2i[6] = t4i - t6i;
                    s2r[7] = t5r - t7i; s2i[7] = t5i + t7r;
                    for (int k = 0; k < 4; k++) {
                        regs_re[base + 2*k]     = s2r[k] + s2r[k + 4];
                        regs_im[base + 2*k]     = s2i[k] + s2i[k + 4];
                        regs_re[base + 2*k + 1] = s2r[k] - s2r[k + 4];
                        regs_im[base + 2*k + 1] = s2i[k] - s2i[k + 4];
                    }
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && tw) {
                    float r = regs_re[a0], i = regs_im[a0];
                    float wr = tw[a1 * 2], wi = tw[a1 * 2 + 1];
                    regs_re[a0] = r * wr - i * wi;
                    regs_im[a0] = r * wi + i * wr;
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL_CONJ: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && tw) {
                    float r = regs_re[a0], i = regs_im[a0];
                    float wr = tw[a1 * 2], wi = -tw[a1 * 2 + 1];
                    regs_re[a0] = r * wr - i * wi;
                    regs_im[a0] = r * wi + i * wr;
                }
                break;
            }
            case DSPIR_FMA:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a0] * regs_re[inst->a1] + regs_re[inst->a2];
                break;
            case DSPIR_FMUL:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] * regs_re[inst->a2];
                break;
            case DSPIR_FADD:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] + regs_re[inst->a2];
                break;
            case DSPIR_FSUB:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] - regs_re[inst->a2];
                break;
            case DSPIR_BARRIER:
                __asm__ volatile("" ::: "memory");
                break;
            case DSPIR_FFT_STAGE: {
                size_t radix = inst->a0;
                size_t stride = inst->a1;
                size_t tw_base = inst->a2;
                size_t ngroups = t->n / (radix * stride);
                for (size_t g = 0; g < ngroups; g++) {
                    size_t gbase = g * radix * stride;
                    for (size_t r = 0; r < radix / 2; r++) {
                        size_t idx1 = gbase + r * stride;
                        size_t idx2 = gbase + (r + radix / 2) * stride;
                        if (idx1 < t->n && idx2 < t->n) {
                            float ar = regs_re[idx1], ai = regs_im[idx1];
                            float br = regs_re[idx2], bi = regs_im[idx2];
                            if (tw && tw_base + r < t->twiddle_sizes[0]) {
                                float wr = tw[(tw_base + r) * 2];
                                float wi = tw[(tw_base + r) * 2 + 1];
                                float tw_re = br * wr - bi * wi;
                                float tw_im = br * wi + bi * wr;
                                regs_re[idx1] = ar + tw_re; regs_im[idx1] = ai + tw_im;
                                regs_re[idx2] = ar - tw_re; regs_im[idx2] = ai - tw_im;
                            } else {
                                regs_re[idx1] = ar + br; regs_im[idx1] = ai + bi;
                                regs_re[idx2] = ar - br; regs_im[idx2] = ai - bi;
                            }
                        }
                    }
                }
                break;
            }
            case DSPIR_DCT_STAGE: {
                size_t stride = inst->a1;
                size_t nstage = t->n / stride;
                for (size_t i = 0; i < nstage / 2; i++) {
                    size_t j = nstage - 1 - i;
                    if (i * stride < t->n && j * stride < t->n && tw) {
                        float a = regs_re[i * stride];
                        float b = regs_re[j * stride];
                        regs_re[i * stride] = a + b;
                        regs_re[j * stride] = (a - b) * tw[i];
                    }
                }
                break;
            }
            case DSPIR_DST_STAGE: {
                size_t stride = inst->a1;
                size_t nstage = t->n / stride;
                for (size_t i = 0; i < nstage / 2; i++) {
                    size_t j = nstage - 1 - i;
                    if (i * stride < t->n && j * stride < t->n && tw) {
                        float a = regs_re[i * stride];
                        float b = regs_re[j * stride];
                        regs_re[i * stride] = a + b;
                        regs_re[j * stride] = (a - b) * (-tw[i]);
                    }
                }
                break;
            }
            case DSPIR_LIFT_PRED: {
                if (inst->a0 < t->n && inst->a1 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] += regs_re[inst->a1] * cv.f;
                }
                break;
            }
            case DSPIR_LIFT_UPD: {
                if (inst->a0 < t->n && inst->a1 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] += regs_re[inst->a1] * cv.f;
                }
                break;
            }
            case DSPIR_LIFT_SCALE: {
                if (inst->a0 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] *= cv.f;
                }
                break;
            }
            case DSPIR_END:
                goto done_split;
            default:
                break;
        }
    }
    
done_split:
    free_regs(regs_re);
    free_regs(regs_im);
    return 0;
}


/**
 * @brief Split-plane FP64 execution (separate real/imag arrays)
 * 
 * Double precision variant for higher accuracy requirements.
 */
int dspir_execute_split_f64(const dspir_transform *t,
                             double *restrict out_re,
                             double *restrict out_im,
                             const double *restrict in_re,
                             const double *restrict in_im) {
    if (!t || !out_re || !out_im || !in_re || !in_im) return -1;
    if (!t->code || t->n_inst == 0) return -1;
    
    /* Allocate separate register planes */
    double *regs_re = (double*)alloc_regs(t->n * sizeof(double));
    double *regs_im = (double*)alloc_regs(t->n * sizeof(double));
    if (!regs_re || !regs_im) {
        free_regs(regs_re);
        free_regs(regs_im);
        return -1;
    }
    memset(regs_re, 0, t->n * sizeof(double));
    memset(regs_im, 0, t->n * sizeof(double));
    
    const double *tw = (const double *)t->twiddles[0];
    
    for (size_t ip = 0; ip < t->n_inst; ip++) {
        const dspir_inst *inst = &t->code[ip];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        
        switch (op) {
            case DSPIR_NOP:
                break;
            case DSPIR_LOAD: {
                size_t in_idx = inst->a1;
                size_t reg_idx = inst->a0;
                if (reg_idx < t->n && in_idx < t->n) {
                    regs_re[reg_idx] = in_re[in_idx];
                    regs_im[reg_idx] = in_im[in_idx];
                }
                break;
            }
            case DSPIR_STORE: {
                size_t out_idx = inst->a0;
                size_t reg_idx = inst->a1;
                if (out_idx < t->n && reg_idx < t->n) {
                    out_re[out_idx] = regs_re[reg_idx];
                    out_im[out_idx] = regs_im[reg_idx];
                }
                break;
            }
            case DSPIR_LOAD_CONST: {
                if (inst->a0 < t->n) {
                    union { uint32_t u; float f; } caster;
                    caster.u = inst->a1;
                    regs_re[inst->a0] = (double)caster.f;
                }
                break;
            }
            case DSPIR_BFLY2: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && a1 < t->n) {
                    double ar = regs_re[a0], ai = regs_im[a0];
                    double br = regs_re[a1], bi = regs_im[a1];
                    regs_re[a0] = ar + br; regs_im[a0] = ai + bi;
                    regs_re[a1] = ar - br; regs_im[a1] = ai - bi;
                }
                break;
            }
            case DSPIR_BFLY4: {
                uint32_t base = inst->a0;
                if (base + 3 < t->n) {
                    double r0 = regs_re[base],     i0 = regs_im[base];
                    double r1 = regs_re[base + 1], i1 = regs_im[base + 1];
                    double r2 = regs_re[base + 2], i2 = regs_im[base + 2];
                    double r3 = regs_re[base + 3], i3 = regs_im[base + 3];
                    double t0r = r0 + r2, t0i = i0 + i2;
                    double t1r = r0 - r2, t1i = i0 - i2;
                    double t2r = r1 + r3, t2i = i1 + i3;
                    double t3r = r1 - r3, t3i = i1 - i3;
                    regs_re[base]     = t0r + t2r; regs_im[base]     = t0i + t2i;
                    regs_re[base + 1] = t1r + t3i; regs_im[base + 1] = t1i - t3r;
                    regs_re[base + 2] = t0r - t2r; regs_im[base + 2] = t0i - t2i;
                    regs_re[base + 3] = t1r - t3i; regs_im[base + 3] = t1i + t3r;
                }
                break;
            }
            case DSPIR_BFLY8: {
                uint32_t base = inst->a0;
                if (base + 7 < t->n) {
                    double r[8], im[8];
                    for (int k = 0; k < 8; k++) {
                        r[k]  = regs_re[base + k];
                        im[k] = regs_im[base + k];
                    }
                    double s1r[8], s1i[8];
                    for (int k = 0; k < 4; k++) {
                        s1r[k]     = r[k] + r[k + 4]; s1i[k]     = im[k] + im[k + 4];
                        s1r[k + 4] = r[k] - r[k + 4]; s1i[k + 4] = im[k] - im[k + 4];
                    }
                    double s2r[8], s2i[8];
                    s2r[0] = s1r[0] + s1r[2]; s2i[0] = s1i[0] + s1i[2];
                    s2r[1] = s1r[1] + s1i[3]; s2i[1] = s1i[1] - s1r[3];
                    s2r[2] = s1r[0] - s1r[2]; s2i[2] = s1i[0] - s1i[2];
                    s2r[3] = s1r[1] - s1i[3]; s2i[3] = s1i[1] + s1r[3];
                    const double c = 0.7071067811865476;
                    double t4r = c * (s1r[4] + s1i[4]), t4i = c * (s1i[4] - s1r[4]);
                    double t5r = -s1i[5], t5i = s1r[5];
                    double t6r = c * (s1i[6] - s1r[6]), t6i = -c * (s1r[6] + s1i[6]);
                    double t7r = s1r[7], t7i = s1i[7];
                    s2r[4] = t4r + t6r; s2i[4] = t4i + t6i;
                    s2r[5] = t5r + t7i; s2i[5] = t5i - t7r;
                    s2r[6] = t4r - t6r; s2i[6] = t4i - t6i;
                    s2r[7] = t5r - t7i; s2i[7] = t5i + t7r;
                    for (int k = 0; k < 4; k++) {
                        regs_re[base + 2*k]     = s2r[k] + s2r[k + 4];
                        regs_im[base + 2*k]     = s2i[k] + s2i[k + 4];
                        regs_re[base + 2*k + 1] = s2r[k] - s2r[k + 4];
                        regs_im[base + 2*k + 1] = s2i[k] - s2i[k + 4];
                    }
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && tw) {
                    double r = regs_re[a0], i = regs_im[a0];
                    double wr = tw[a1 * 2], wi = tw[a1 * 2 + 1];
                    regs_re[a0] = r * wr - i * wi;
                    regs_im[a0] = r * wi + i * wr;
                }
                break;
            }
            case DSPIR_TWIDDLE_MUL_CONJ: {
                uint32_t a0 = inst->a0;
                uint32_t a1 = inst->a1;
                if (a0 < t->n && tw) {
                    double r = regs_re[a0], i = regs_im[a0];
                    double wr = tw[a1 * 2], wi = -tw[a1 * 2 + 1];
                    regs_re[a0] = r * wr - i * wi;
                    regs_im[a0] = r * wi + i * wr;
                }
                break;
            }
            case DSPIR_FMA:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a0] * regs_re[inst->a1] + regs_re[inst->a2];
                break;
            case DSPIR_FMUL:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] * regs_re[inst->a2];
                break;
            case DSPIR_FADD:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] + regs_re[inst->a2];
                break;
            case DSPIR_FSUB:
                if (inst->a0 < t->n && inst->a1 < t->n && inst->a2 < t->n)
                    regs_re[inst->a0] = regs_re[inst->a1] - regs_re[inst->a2];
                break;
            case DSPIR_BARRIER:
                __asm__ volatile("" ::: "memory");
                break;
            case DSPIR_FFT_STAGE: {
                size_t radix = inst->a0;
                size_t stride = inst->a1;
                size_t tw_base = inst->a2;
                size_t ngroups = t->n / (radix * stride);
                for (size_t g = 0; g < ngroups; g++) {
                    size_t gbase = g * radix * stride;
                    for (size_t r = 0; r < radix / 2; r++) {
                        size_t idx1 = gbase + r * stride;
                        size_t idx2 = gbase + (r + radix / 2) * stride;
                        if (idx1 < t->n && idx2 < t->n) {
                            double ar = regs_re[idx1], ai = regs_im[idx1];
                            double br = regs_re[idx2], bi = regs_im[idx2];
                            if (tw && tw_base + r < t->twiddle_sizes[0]) {
                                double wr = tw[(tw_base + r) * 2];
                                double wi = tw[(tw_base + r) * 2 + 1];
                                double tw_re = br * wr - bi * wi;
                                double tw_im = br * wi + bi * wr;
                                regs_re[idx1] = ar + tw_re; regs_im[idx1] = ai + tw_im;
                                regs_re[idx2] = ar - tw_re; regs_im[idx2] = ai - tw_im;
                            } else {
                                regs_re[idx1] = ar + br; regs_im[idx1] = ai + bi;
                                regs_re[idx2] = ar - br; regs_im[idx2] = ai - bi;
                            }
                        }
                    }
                }
                break;
            }
            case DSPIR_DCT_STAGE: {
                size_t stride = inst->a1;
                size_t nstage = t->n / stride;
                for (size_t i = 0; i < nstage / 2; i++) {
                    size_t j = nstage - 1 - i;
                    if (i * stride < t->n && j * stride < t->n && tw) {
                        double a = regs_re[i * stride];
                        double b = regs_re[j * stride];
                        regs_re[i * stride] = a + b;
                        regs_re[j * stride] = (a - b) * tw[i];
                    }
                }
                break;
            }
            case DSPIR_DST_STAGE: {
                size_t stride = inst->a1;
                size_t nstage = t->n / stride;
                for (size_t i = 0; i < nstage / 2; i++) {
                    size_t j = nstage - 1 - i;
                    if (i * stride < t->n && j * stride < t->n && tw) {
                        double a = regs_re[i * stride];
                        double b = regs_re[j * stride];
                        regs_re[i * stride] = a + b;
                        regs_re[j * stride] = (a - b) * (-tw[i]);
                    }
                }
                break;
            }
            case DSPIR_LIFT_PRED: {
                if (inst->a0 < t->n && inst->a1 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] += regs_re[inst->a1] * (double)cv.f;
                }
                break;
            }
            case DSPIR_LIFT_UPD: {
                if (inst->a0 < t->n && inst->a1 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] += regs_re[inst->a1] * (double)cv.f;
                }
                break;
            }
            case DSPIR_LIFT_SCALE: {
                if (inst->a0 < t->n) {
                    union { uint32_t u; float f; } cv;
                    cv.u = inst->a2;
                    regs_re[inst->a0] *= (double)cv.f;
                }
                break;
            }
            case DSPIR_END:
                goto done_split_f64;
            default:
                break;
        }
    }
    
done_split_f64:
    free_regs(regs_re);
    free_regs(regs_im);
    return 0;
}
