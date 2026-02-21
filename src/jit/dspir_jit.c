/**
 * @file dspir_jit.c
 * @brief JIT compilation system for FastAndFourier
 * 
 * Compiles IR bytecode to native machine code for maximum performance.
 * Uses system compiler (GCC/Clang) for portability.
 */

/* For popen/pclose */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dspir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>

/* Platform-specific dynamic library extension */
#ifdef __APPLE__
    #define DSPIR_SO_EXT ".dylib"
#elif defined(_WIN32)
    #define DSPIR_SO_EXT ".dll"
#else
    #define DSPIR_SO_EXT ".so"
#endif

/**
 * @brief JIT compilation context
 */
struct dspir_jit_ctx {
    void *handle;              /**< Dynamic library handle */
    dspir_kernel_fn fn;        /**< Compiled kernel function */
    char so_path[256];         /**< Path to compiled shared object */
    char c_path[256];          /**< Path to generated C source */
    int use_native_jit;        /**< Use native code generator instead of compiler */
    uint32_t flags;            /**< JIT compilation flags */
    
    /* Native JIT state (for future implementation) */
    void *native_code;         /**< Directly generated machine code */
    size_t native_size;        /**< Size of native code */
};

/**
 * @brief Create a JIT compilation context
 */
dspir_jit_ctx* dspir_jit_create(void) {
    dspir_jit_ctx *ctx = calloc(1, sizeof(dspir_jit_ctx));
    if (!ctx) return NULL;
    
    ctx->use_native_jit = 0;  /* Default to compiler-based JIT */
    ctx->flags = 0;
    
    return ctx;
}

/**
 * @brief Destroy JIT context and cleanup resources
 */
void dspir_jit_destroy(dspir_jit_ctx *ctx) {
    if (!ctx) return;
    
    if (ctx->handle) {
        dlclose(ctx->handle);
    }
    
    /* Remove temporary files */
    if (ctx->so_path[0]) {
        unlink(ctx->so_path);
    }
    if (ctx->c_path[0]) {
        unlink(ctx->c_path);
    }
    
    /* Free native code if present */
    if (ctx->native_code) {
        free(ctx->native_code);
    }
    
    free(ctx);
}

/**
 * @brief Detect architecture type
 */
static dspir_arch_type detect_arch(void) {
#if defined(__aarch64__)
    return DSPIR_ARCH_TYPE_AARCH64;
#elif defined(__x86_64__) || defined(_M_X64)
    return DSPIR_ARCH_TYPE_X86_64;
#else
    return DSPIR_ARCH_TYPE_GENERIC;
#endif
}

/**
 * @brief Find best base size for mixed-radix decomposition
 * 
 * For large transforms, we decompose N = N1 * N2 and use a base kernel
 * of size N1, calling it N2 times. This dramatically reduces code size.
 * 
 * Returns 0 if no decomposition is beneficial (use full unroll)
 */
static size_t find_best_base_size(size_t n) {
    /* For small transforms, full unroll is faster */
    if (n <= 256) {
        return 0;  /* No decomposition - use full unroll */
    }
    
    /* Prefer power-of-2 base sizes that divide n evenly */
    /* Try sizes in order of preference (larger = fewer calls) */
    size_t candidates[] = {256, 128, 64, 32, 16};
    
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        size_t base = candidates[i];
        if (n % base == 0 && n / base >= 2) {
            return base;  /* Found good decomposition */
        }
    }
    
    return 0;  /* No good decomposition found */
}

/**
 * @brief Generate inlined FFT kernel for small base size
 * 
 * This generates a static inline function that can be called
 * multiple times by the main kernel, avoiding code duplication.
 */
static void emit_base_fft_kernel(FILE *f, size_t base_size, int use_simd, int inplace) {
    const char *ptr = inplace ? "data" : "regs";
    
    fprintf(f, "/* Base FFT kernel for size %zu */\n", base_size);
    fprintf(f, "static inline void fft_base_%zu(float *restrict %s, size_t offset, size_t stride) {\n", 
            base_size, ptr);
    fprintf(f, "    (void)offset; (void)stride;\n");
    
    /* Generate butterfly operations for base_size FFT */
    /* This is a simplified radix-2 Cooley-Tukey */
    size_t stages = 0;
    size_t temp = base_size;
    while (temp > 1) {
        temp >>= 1;
        stages++;
    }
    
    /* For each stage */
    for (size_t stage = 0; stage < stages; stage++) {
        size_t butterfly_size = 1 << stage;  /* Distance between butterfly pairs */
        size_t group_size = butterfly_size << 1;
        size_t num_groups = base_size / group_size;
        
        fprintf(f, "    /* Stage %zu: butterfly size %zu */\n", stage, butterfly_size);
        
        /* Generate butterflies for this stage */
        for (size_t g = 0; g < num_groups; g++) {
            for (size_t b = 0; b < butterfly_size; b++) {
                size_t idx0 = g * group_size + b;
                size_t idx1 = idx0 + butterfly_size;
                
                /* Twiddle index for this butterfly */
                size_t twiddle_idx = b * num_groups;
                
                if (stage == 0) {
                    /* First stage: no twiddle multiplication needed (twiddle = 1) */
                    fprintf(f, "    { /* BFLY2 %zu,%zu */\n", idx0, idx1);
                    fprintf(f, "        float ar = %s[%zu], ai = %s[%zu];\n", 
                            ptr, idx0 * 2, ptr, idx0 * 2 + 1);
                    fprintf(f, "        float br = %s[%zu], bi = %s[%zu];\n", 
                            ptr, idx1 * 2, ptr, idx1 * 2 + 1);
                    fprintf(f, "        %s[%zu] = ar + br; %s[%zu] = ai + bi;\n", 
                            ptr, idx0 * 2, ptr, idx0 * 2 + 1);
                    fprintf(f, "        %s[%zu] = ar - br; %s[%zu] = ai - bi;\n", 
                            ptr, idx1 * 2, ptr, idx1 * 2 + 1);
                    fprintf(f, "    }\n");
                } else {
                    /* Later stages: include twiddle multiplication */
                    fprintf(f, "    { /* BFLY2 %zu,%zu with twiddle %zu */\n", idx0, idx1, twiddle_idx);
                    fprintf(f, "        float ar = %s[%zu], ai = %s[%zu];\n", 
                            ptr, idx0 * 2, ptr, idx0 * 2 + 1);
                    fprintf(f, "        float br = %s[%zu], bi = %s[%zu];\n", 
                            ptr, idx1 * 2, ptr, idx1 * 2 + 1);
                    fprintf(f, "        float wr = tw[%zu*2], wi = tw[%zu*2+1];\n", 
                            twiddle_idx, twiddle_idx);
                    fprintf(f, "        float t_re = br * wr - bi * wi;\n");
                    fprintf(f, "        float t_im = br * wi + bi * wr;\n");
                    fprintf(f, "        %s[%zu] = ar + t_re; %s[%zu] = ai + t_im;\n", 
                            ptr, idx0 * 2, ptr, idx0 * 2 + 1);
                    fprintf(f, "        %s[%zu] = ar - t_re; %s[%zu] = ai - t_im;\n", 
                            ptr, idx1 * 2, ptr, idx1 * 2 + 1);
                    fprintf(f, "    }\n");
                }
            }
        }
    }
    
    fprintf(f, "}\n\n");
}

/**
 * @brief Generate scalar C code for BFLY2 (interleaved complex)
 */
static void emit_scalar_bfly2(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    if (inplace) {
        fprintf(f, "    {\n");
        fprintf(f, "        float ar = data[%u], ai = data[%u];\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        float br = data[%u], bi = data[%u];\n", a1 * 2, a1 * 2 + 1);
        fprintf(f, "        data[%u] = ar + br; data[%u] = ai + bi;\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        data[%u] = ar - br; data[%u] = ai - bi;\n", a1 * 2, a1 * 2 + 1);
        fprintf(f, "    }\n");
    } else {
        fprintf(f, "    {\n");
        fprintf(f, "        float ar = regs[%u], ai = regs[%u];\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        float br = regs[%u], bi = regs[%u];\n", a1 * 2, a1 * 2 + 1);
        fprintf(f, "        regs[%u] = ar + br; regs[%u] = ai + bi;\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        regs[%u] = ar - br; regs[%u] = ai - bi;\n", a1 * 2, a1 * 2 + 1);
        fprintf(f, "    }\n");
    }
}

/**
 * @brief Generate NEON SIMD code for BFLY2 (2 complex butterflies at once)
 */
static void emit_neon_bfly2(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    /* 
     * NEON optimized butterfly for 2 complex pairs:
     * Load: [a0_r, a0_i, a1_r, a1_i] and [b0_r, b0_i, b1_r, b1_i]
     * Result: [a0+b0, a1+b1] and [a0-b0, a1-b1]
     */
    const char *ptr = inplace ? "data" : "regs";
    
    fprintf(f, "    {\n");
    fprintf(f, "        float32x2_t a = vld1_f32(&%s[%u]);\n", ptr, a0 * 2);
    fprintf(f, "        float32x2_t b = vld1_f32(&%s[%u]);\n", ptr, a1 * 2);
    fprintf(f, "        vst1_f32(&%s[%u], vadd_f32(a, b));\n", ptr, a0 * 2);
    fprintf(f, "        vst1_f32(&%s[%u], vsub_f32(a, b));\n", ptr, a1 * 2);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate scalar C code for twiddle multiply
 */
static void emit_scalar_twiddle_mul(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    if (inplace) {
        fprintf(f, "    {\n");
        fprintf(f, "        float re = data[%u], im = data[%u];\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        float wr = tw[%u*2], wi = tw[%u*2+1];\n", a1, a1);
        fprintf(f, "        data[%u] = re*wr - im*wi;\n", a0 * 2);
        fprintf(f, "        data[%u] = re*wi + im*wr;\n", a0 * 2 + 1);
        fprintf(f, "    }\n");
    } else {
        fprintf(f, "    {\n");
        fprintf(f, "        float re = regs[%u], im = regs[%u];\n", a0 * 2, a0 * 2 + 1);
        fprintf(f, "        float wr = tw[%u*2], wi = tw[%u*2+1];\n", a1, a1);
        fprintf(f, "        regs[%u] = re*wr - im*wi;\n", a0 * 2);
        fprintf(f, "        regs[%u] = re*wi + im*wr;\n", a0 * 2 + 1);
        fprintf(f, "    }\n");
    }
}

/**
 * @brief Generate split-plane BFLY2 (separate real/imag arrays)
 * 
 * This enables much better auto-vectorization on x86 with AVX-512.
 * Real and imaginary arrays are separate 64-byte aligned streams.
 */
static void emit_split_bfly2(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    const char *re = inplace ? "data_re" : "regs_re";
    const char *im = inplace ? "data_im" : "regs_im";
    fprintf(f, "    {\n");
    fprintf(f, "        float ar = %s[%u], ai = %s[%u];\n", re, a0, im, a0);
    fprintf(f, "        float br = %s[%u], bi = %s[%u];\n", re, a1, im, a1);
    fprintf(f, "        %s[%u] = ar + br; %s[%u] = ai + bi;\n", re, a0, im, a0);
    fprintf(f, "        %s[%u] = ar - br; %s[%u] = ai - bi;\n", re, a1, im, a1);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate split-plane twiddle multiply
 */
static void emit_split_twiddle_mul(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    const char *re = inplace ? "data_re" : "regs_re";
    const char *im = inplace ? "data_im" : "regs_im";
    fprintf(f, "    {\n");
    fprintf(f, "        float r = %s[%u], i = %s[%u];\n", re, a0, im, a0);
    fprintf(f, "        float wr = tw_re[%u], wi = tw_im[%u];\n", a1, a1);
    fprintf(f, "        %s[%u] = r*wr - i*wi;\n", re, a0);
    fprintf(f, "        %s[%u] = r*wi + i*wr;\n", im, a0);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate NEON SIMD code for twiddle multiply
 * 
 * Uses NEON complex multiplication:
     * (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
 * Can process 2 complex values at once with float32x4_t
 */
static void emit_neon_twiddle_mul(FILE *f, uint8_t a0, uint8_t a1, int inplace) {
    const char *ptr = inplace ? "data" : "regs";
    
    /* 
     * NEON complex multiply:
     * Let v = [re, im, x, x] (we only use first 2 elements)
     * Let w = [wr, wi, x, x] (twiddle)
     * 
     * result_real = re*wr - im*wi
     * result_imag = re*wi + im*wr
     */
    fprintf(f, "    {\n");
    fprintf(f, "        float32x2_t v = vld1_f32(&%s[%u]);\n", ptr, a0 * 2);
    fprintf(f, "        float wr = tw[%u*2], wi = tw[%u*2+1];\n", a1, a1);
    fprintf(f, "        float re = vget_lane_f32(v, 0), im = vget_lane_f32(v, 1);\n");
    fprintf(f, "        float32x2_t r;\n");
    fprintf(f, "        r = vset_lane_f32(re*wr - im*wi, r, 0);\n");
    fprintf(f, "        r = vset_lane_f32(re*wi + im*wr, r, 1);\n");
    fprintf(f, "        vst1_f32(&%s[%u], r);\n", ptr, a0 * 2);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate C source code from IR with SIMD and in-place support
 * 
 * This is the compiler-based JIT approach - generates C code
 * and compiles it to a shared library.
 */
static void dspir_jit_generate_c_ex(dspir_transform *t, 
                                     const char *path,
                                     dspir_arch_type arch,
                                     uint32_t flags) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    
    int use_simd = (flags & DSPIR_FLAG_JIT_SIMD) && (arch == DSPIR_ARCH_TYPE_AARCH64);
    int inplace = flags & DSPIR_FLAG_JIT_INPLACE;
    
    /* Header */
    fprintf(f, "/* Auto-generated JIT kernel for FastAndFourier */\n");
    fprintf(f, "/* Flags: %s%s */\n", 
            use_simd ? "SIMD " : "",
            inplace ? "INPLACE" : "");
    fprintf(f, "#include <stddef.h>\n");
    fprintf(f, "#include <stdint.h>\n");
    fprintf(f, "#include <stdlib.h>\n");
    fprintf(f, "#include <math.h>\n\n");
    
    /* Architecture-specific includes */
    if (use_simd) {
        switch (arch) {
            case DSPIR_ARCH_TYPE_X86_64:
                fprintf(f, "#include <immintrin.h>\n");
                break;
            case DSPIR_ARCH_TYPE_AARCH64:
                fprintf(f, "#include <arm_neon.h>\n");
                break;
            default:
                break;
        }
        fprintf(f, "\n");
    }
    
    /* Kernel function signature with aggressive optimization hints */
    fprintf(f, "__attribute__((visibility(\"default\"), flatten, hot, aligned(64)))\n");
    fprintf(f, "void dspir_jit_kernel(void *restrict out,\n");
    fprintf(f, "                      const void *restrict in,\n");
    fprintf(f, "                      size_t n,\n");
    fprintf(f, "                      const void *twiddles) {\n");
    
    /* Cast pointers */
    if (inplace) {
        /* In-place mode: work directly on output buffer */
        fprintf(f, "    float *restrict const data = (float*)out;\n");
        fprintf(f, "    const float *restrict const tw = (const float*)twiddles;\n");
        fprintf(f, "    (void)in; (void)n;\n\n");
    } else {
        fprintf(f, "    float *restrict const out_f = (float*)out;\n");
        fprintf(f, "    const float *restrict const in_f = (const float*)in;\n");
        fprintf(f, "    const float *restrict const tw = (const float*)twiddles;\n");
        fprintf(f, "    (void)n;\n\n");
    }
    
    /* Register file - only if not in-place mode */
    size_t needed_regs = t->n * 2;
    int use_stack = 0;
    
    if (!inplace) {
        /* Stack threshold: 64KB for stack allocation */
        const size_t STACK_THRESHOLD = 64 * 1024;
        use_stack = (needed_regs * sizeof(float) <= STACK_THRESHOLD);
        
        fprintf(f, "    /* Register file for %zu complex points (%s) */\n", 
                t->n, use_stack ? "stack" : "heap");
        if (use_stack) {
            fprintf(f, "    float regs[%zu] __attribute__((aligned(64))) = {0};\n\n", 
                    needed_regs);
        } else {
            fprintf(f, "    float *regs = (float*)aligned_alloc(64, %zu * sizeof(float));\n",
                    needed_regs);
            fprintf(f, "    if (!regs) return;\n");
            fprintf(f, "    for (size_t i = 0; i < %zu; i++) regs[i] = 0.0f;\n\n",
                    needed_regs);
        }
    }
    
    /* Generate code for each instruction */
    for (size_t i = 0; i < t->n_inst; i++) {
        const dspir_inst *inst = &t->code[i];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        
        fprintf(f, "    /* Inst %zu: opcode %d */\n", i, op);
        
        switch (op) {
            case DSPIR_NOP:
                fprintf(f, "    /* NOP */\n");
                break;
                
            case DSPIR_LOAD:
                if (inplace) {
                    /* In-place: no load needed, data already in place */
                    fprintf(f, "    /* In-place: data[%u] already loaded */\n", inst->a1 * 2);
                } else {
                    /* Interleaved complex: regs[2*a0] = real, regs[2*a0+1] = imag */
                    fprintf(f, "    regs[%u] = in_f[%u];\n", inst->a0 * 2, inst->a1 * 2);
                    fprintf(f, "    regs[%u] = in_f[%u];\n", inst->a0 * 2 + 1, inst->a1 * 2 + 1);
                }
                break;
                
            case DSPIR_STORE:
                if (inplace) {
                    /* In-place: no store needed, data already in place */
                    fprintf(f, "    /* In-place: data[%u] already stored */\n", inst->a0 * 2);
                } else {
                    /* Interleaved complex */
                    fprintf(f, "    out_f[%u] = regs[%u];\n", inst->a0 * 2, inst->a1 * 2);
                    fprintf(f, "    out_f[%u] = regs[%u];\n", inst->a0 * 2 + 1, inst->a1 * 2 + 1);
                }
                break;
                
            case DSPIR_LOAD_CONST: {
                union { uint32_t u; float f; } c = { .u = inst->a1 };
                if (inplace) {
                    fprintf(f, "    data[%u] = %.9ef;\n", inst->a0, c.f);
                } else {
                    fprintf(f, "    regs[%u] = %.9ef;\n", inst->a0, c.f);
                }
                break;
            }
                
            case DSPIR_BFLY2:
                /* Complex radix-2 butterfly with interleaved registers */
                if (use_simd && arch == DSPIR_ARCH_TYPE_AARCH64) {
                    emit_neon_bfly2(f, inst->a0, inst->a1, inplace);
                } else {
                    emit_scalar_bfly2(f, inst->a0, inst->a1, inplace);
                }
                break;
                
            case DSPIR_BFLY4:
                if (inplace) {
                    fprintf(f, "    {\n");
                    fprintf(f, "        float r0 = data[%u], r1 = data[%u];\n", 
                            inst->a0, inst->a0 + 1);
                    fprintf(f, "        float r2 = data[%u], r3 = data[%u];\n",
                            inst->a0 + 2, inst->a0 + 3);
                    fprintf(f, "        float t0 = r0 + r2, t1 = r0 - r2;\n");
                    fprintf(f, "        float t2 = r1 + r3, t3 = r1 - r3;\n");
                    fprintf(f, "        data[%u] = t0 + t2;\n", inst->a0);
                    fprintf(f, "        data[%u] = t0 - t2;\n", inst->a0 + 1);
                    fprintf(f, "        data[%u] = t1 + t3;\n", inst->a0 + 2);
                    fprintf(f, "        data[%u] = t1 - t3;\n", inst->a0 + 3);
                    fprintf(f, "    }\n");
                } else {
                    fprintf(f, "    {\n");
                    fprintf(f, "        float r0 = regs[%u], r1 = regs[%u];\n", 
                            inst->a0, inst->a0 + 1);
                    fprintf(f, "        float r2 = regs[%u], r3 = regs[%u];\n",
                            inst->a0 + 2, inst->a0 + 3);
                    fprintf(f, "        float t0 = r0 + r2, t1 = r0 - r2;\n");
                    fprintf(f, "        float t2 = r1 + r3, t3 = r1 - r3;\n");
                    fprintf(f, "        regs[%u] = t0 + t2;\n", inst->a0);
                    fprintf(f, "        regs[%u] = t0 - t2;\n", inst->a0 + 1);
                    fprintf(f, "        regs[%u] = t1 + t3;\n", inst->a0 + 2);
                    fprintf(f, "        regs[%u] = t1 - t3;\n", inst->a0 + 3);
                    fprintf(f, "    }\n");
                }
                break;
                
            case DSPIR_TWIDDLE_MUL:
                /* Complex multiply: (a+ib) * (c+id) with interleaved registers */
                if (use_simd && arch == DSPIR_ARCH_TYPE_AARCH64) {
                    emit_neon_twiddle_mul(f, inst->a0, inst->a1, inplace);
                } else {
                    emit_scalar_twiddle_mul(f, inst->a0, inst->a1, inplace);
                }
                break;
                
            case DSPIR_FMA:
                if (inplace) {
                    fprintf(f, "    data[%u] = data[%u] * data[%u] + data[%u];\n",
                            inst->a0, inst->a0, inst->a1, inst->a2);
                } else {
                    fprintf(f, "    regs[%u] = regs[%u] * regs[%u] + regs[%u];\n",
                            inst->a0, inst->a0, inst->a1, inst->a2);
                }
                break;
                
            case DSPIR_FFT_STAGE:
                if (inplace) {
                    fprintf(f, "    {\n");
                    fprintf(f, "        size_t radix = %u;\n", inst->a0);
                    fprintf(f, "        size_t stride = %u;\n", inst->a1);
                    fprintf(f, "        size_t tw_base = %u;\n", inst->a2);
                    fprintf(f, "        size_t ngroups = %zu / (radix * stride);\n", t->n);
                    fprintf(f, "        for (size_t g = 0; g < ngroups; g++) {\n");
                    fprintf(f, "            for (size_t r = 0; r < radix/2; r++) {\n");
                    fprintf(f, "                float a = data[g*stride + r];\n");
                    fprintf(f, "                float b = data[g*stride + r + stride/2];\n");
                    fprintf(f, "                float wr = tw[tw_base + r*2];\n");
                    fprintf(f, "                float wi = tw[tw_base + r*2+1];\n");
                    fprintf(f, "                float br = b * wr, bi = b * wi;\n");
                    fprintf(f, "                data[g*stride + r] = a + br;\n");
                    fprintf(f, "                data[g*stride + r + stride/2] = a - br;\n");
                    fprintf(f, "            }\n");
                    fprintf(f, "        }\n");
                    fprintf(f, "    }\n");
                } else {
                    fprintf(f, "    {\n");
                    fprintf(f, "        size_t radix = %u;\n", inst->a0);
                    fprintf(f, "        size_t stride = %u;\n", inst->a1);
                    fprintf(f, "        size_t tw_base = %u;\n", inst->a2);
                    fprintf(f, "        size_t ngroups = %zu / (radix * stride);\n", t->n);
                    fprintf(f, "        for (size_t g = 0; g < ngroups; g++) {\n");
                    fprintf(f, "            for (size_t r = 0; r < radix/2; r++) {\n");
                    fprintf(f, "                float a = regs[g*stride + r];\n");
                    fprintf(f, "                float b = regs[g*stride + r + stride/2];\n");
                    fprintf(f, "                float wr = tw[tw_base + r*2];\n");
                    fprintf(f, "                float wi = tw[tw_base + r*2+1];\n");
                    fprintf(f, "                float br = b * wr, bi = b * wi;\n");
                    fprintf(f, "                regs[g*stride + r] = a + br;\n");
                    fprintf(f, "                regs[g*stride + r + stride/2] = a - br;\n");
                    fprintf(f, "            }\n");
                    fprintf(f, "        }\n");
                    fprintf(f, "    }\n");
                }
                break;
                
            case DSPIR_LIFT_PRED:
                if (inplace) {
                    fprintf(f, "    {\n");
                    fprintf(f, "        union { uint32_t u; float f; } c = { .u = %u };\n", 
                            inst->a2);
                    fprintf(f, "        data[%u] += data[%u] * c.f;\n", inst->a0, inst->a1);
                    fprintf(f, "    }\n");
                } else {
                    fprintf(f, "    {\n");
                    fprintf(f, "        union { uint32_t u; float f; } c = { .u = %u };\n", 
                            inst->a2);
                    fprintf(f, "        regs[%u] += regs[%u] * c.f;\n", inst->a0, inst->a1);
                    fprintf(f, "    }\n");
                }
                break;
                
            case DSPIR_LIFT_UPD:
                if (inplace) {
                    fprintf(f, "    {\n");
                    fprintf(f, "        union { uint32_t u; float f; } c = { .u = %u };\n",
                            inst->a2);
                    fprintf(f, "        data[%u] += data[%u] * c.f;\n", inst->a0, inst->a1);
                    fprintf(f, "    }\n");
                } else {
                    fprintf(f, "    {\n");
                    fprintf(f, "        union { uint32_t u; float f; } c = { .u = %u };\n",
                            inst->a2);
                    fprintf(f, "        regs[%u] += regs[%u] * c.f;\n", inst->a0, inst->a1);
                    fprintf(f, "    }\n");
                }
                break;
                
            case DSPIR_DOWN2:
                if (inplace) {
                    fprintf(f, "    for (size_t j = 0; j < %zu/2; j++) {\n", t->n);
                    fprintf(f, "        data[j] = data[j*2 + %u];\n", inst->a0);
                    fprintf(f, "    }\n");
                } else {
                    fprintf(f, "    for (size_t j = 0; j < %zu/2; j++) {\n", t->n);
                    fprintf(f, "        regs[j] = regs[j*2 + %u];\n", inst->a0);
                    fprintf(f, "    }\n");
                }
                break;
                
            case DSPIR_POLY_FIR:
                fprintf(f, "    {\n");
                fprintf(f, "        size_t taps = %u;\n", inst->a0);
                fprintf(f, "        size_t stride = %u;\n", inst->a1);
                fprintf(f, "        for (size_t ch = 0; ch < 8; ch++) {\n");
                fprintf(f, "            float acc = 0.0f;\n");
                fprintf(f, "            for (size_t k = 0; k < taps; k++) {\n");
                fprintf(f, "                acc += in_f[ch*stride + k] * tw[k];\n");
                fprintf(f, "            }\n");
                fprintf(f, "            out_f[ch*stride] = acc;\n");
                fprintf(f, "        }\n");
                fprintf(f, "    }\n");
                break;
                
            case DSPIR_END:
                fprintf(f, "    goto cleanup;\n");
                break;
                
            default:
                fprintf(f, "    /* Unknown opcode %d */\n", op);
                break;
        }
    }
    
    /* Cleanup code for heap-allocated register files */
    fprintf(f, "cleanup:\n");
    if (!inplace && !use_stack) {
        fprintf(f, "    free(regs);\n");
    }
    fprintf(f, "    return;\n");
    fprintf(f, "}\n");
    fclose(f);
}

/* Backward compatibility wrapper */
static void dspir_jit_generate_c(dspir_transform *t, 
                                  const char *path,
                                  dspir_arch_type arch) {
    dspir_jit_generate_c_ex(t, path, arch, 0);
}

/**
 * @brief Compile generated C code to shared library
 */
static int dspir_jit_compile_c(dspir_jit_ctx *ctx, const char *c_path, 
                                const char *so_path, dspir_arch_type arch,
                                uint32_t flags) {
    char cmd[2048];
    
    /* Determine compiler flags based on architecture */
    const char *arch_flags = "";
    switch (arch) {
        case DSPIR_ARCH_TYPE_X86_64:
            #if defined(DSPIR_HAVE_AVX512)
                arch_flags = "-march=native -mavx512f -mavx512dq";
            #elif defined(DSPIR_HAVE_AVX2)
                arch_flags = "-march=native -mavx2 -mfma";
            #else
                arch_flags = "-march=native";
            #endif
            break;
        case DSPIR_ARCH_TYPE_AARCH64:
            /* AArch64 has NEON enabled by default, but enable full ISA */
            if (flags & DSPIR_FLAG_JIT_SIMD) {
                arch_flags = "-march=armv8.2-a+fp16+simd+crypto -mtune=cortex-a78";
            } else {
                arch_flags = "-march=native";
            }
            break;
        default:
            arch_flags = "-O3";
            break;
    }
    
    /* Construct compile command with optimized flags for ARM */
    snprintf(cmd, sizeof(cmd),
             "gcc -std=c11 -pedantic -Wall -Wextra -Werror "
             "%s "
             "-O3 -ffast-math -funroll-loops -fno-math-errno -fomit-frame-pointer "
             "%s "  /* Additional SIMD flags */
             "-fPIC -shared -w "
             "-fvisibility=hidden "
             "-o %s %s",
             arch_flags,
             (flags & DSPIR_FLAG_JIT_SIMD) ? "-DSPIR_SIMD" : "",
             so_path, c_path);
    
    int ret = system(cmd);
    if (ret != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief Compile a transform to native code with flags
 */
int dspir_jit_compile_ex(dspir_jit_ctx *ctx, dspir_transform *t, uint32_t flags) {
    if (!ctx || !t) return -1;
    
    /* Store flags for this compilation */
    ctx->flags = flags;
    
    /* Detect architecture */
    dspir_arch_type arch = detect_arch();
    
    /* Generate unique temp file names */
    static int counter = 0;
    int id = counter++;
    snprintf(ctx->c_path, sizeof(ctx->c_path), "/tmp/dspir_jit_%d_%d.c", getpid(), id);
    snprintf(ctx->so_path, sizeof(ctx->so_path), "/tmp/dspir_jit_%d_%d%s", getpid(), id, DSPIR_SO_EXT);
    
    /* Generate C source */
    dspir_jit_generate_c_ex(t, ctx->c_path, arch, flags);
    
    /* Get file size for debugging */
    struct stat st;
    if (stat(ctx->c_path, &st) == 0) {
        printf("[JIT] Generated C source: %.1f KB\n", st.st_size / 1024.0);
    }
    
    /* Compile to shared library */
    printf("[JIT] Compiling (this may take a moment)...\n");
    if (dspir_jit_compile_c(ctx, ctx->c_path, ctx->so_path, arch, flags) != 0) {
        fprintf(stderr, "[JIT] Compilation failed\n");
        return -1;
    }
    printf("[JIT] Compilation successful\n");
    
    /* Load the compiled library */
    ctx->handle = dlopen(ctx->so_path, RTLD_NOW | RTLD_LOCAL);
    if (!ctx->handle) {
        fprintf(stderr, "[JIT] Failed to load: %s\n", dlerror());
        return -1;
    }
    
    /* Get the kernel function */
    ctx->fn = (dspir_kernel_fn)dlsym(ctx->handle, "dspir_jit_kernel");
    if (!ctx->fn) {
        fprintf(stderr, "[JIT] Failed to find kernel: %s\n", dlerror());
        dlclose(ctx->handle);
        ctx->handle = NULL;
        return -1;
    }
    
    return 0;
}

/**
 * @brief Default JIT flags - enable SIMD on supported platforms
 * 
 * Note: We don't enable INPLACE by default because the user may pass
 * different input and output buffers. INPLACE should be explicitly
 * requested when the caller knows input == output.
 */
static uint32_t get_default_jit_flags(void) {
    uint32_t flags = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    flags |= DSPIR_FLAG_JIT_SIMD;  /* Enable NEON on ARM64 */
#elif defined(__x86_64__) || defined(_M_X64)
    flags |= DSPIR_FLAG_JIT_SIMD;  /* Enable SIMD on x86_64 */
#endif
    return flags;
}

/**
 * @brief Compile a transform to native code (backward compatible)
 */
int dspir_jit_compile(dspir_jit_ctx *ctx, dspir_transform *t) {
    return dspir_jit_compile_ex(ctx, t, get_default_jit_flags());
}

/**
 * @brief Get the compiled kernel function
 */
dspir_kernel_fn dspir_jit_get_kernel(dspir_jit_ctx *ctx) {
    if (!ctx) return NULL;
    return ctx->fn;
}

/**
 * @brief Execute transform using JIT compilation with flags
 */
int dspir_execute_jit_ex(const dspir_transform *t, void *out, const void *in, uint32_t flags) {
    if (!t || !out || !in) return -1;
    
    /* Use default flags if none specified */
    if (flags == 0) {
        flags = get_default_jit_flags();
    }
    
    /* Compile with flags */
    dspir_jit_ctx *ctx = dspir_jit_create();
    if (!ctx) return -1;
    
    if (dspir_jit_compile_ex(ctx, t, flags) != 0) {
        dspir_jit_destroy(ctx);
        return -1;
    }
    
    /* Get and execute the kernel */
    dspir_kernel_fn fn = dspir_jit_get_kernel(ctx);
    if (!fn) {
        dspir_jit_destroy(ctx);
        return -1;
    }
    
    fn(out, in, t->n, t->twiddles[0]);
    
    dspir_jit_destroy(ctx);
    return 0;
}

/**
 * @brief Execute transform using JIT compilation
 */
int dspir_execute_jit(const dspir_transform *t, void *out, const void *in) {
    return dspir_execute_jit_ex(t, out, in, 0);
}
