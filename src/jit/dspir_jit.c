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
#include <stdatomic.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>

/* JIT logging control */
static int dspir_jit_verbose_level = 0;

void dspir_jit_set_verbose(int level) {
    dspir_jit_verbose_level = level;
}

#define JIT_LOG(...) do { if (dspir_jit_verbose_level) fprintf(stderr, __VA_ARGS__); } while(0)

/* Platform-specific dynamic library extension */
#ifdef __APPLE__
    #define DSPIR_SO_EXT ".dylib"
#elif defined(_WIN32)
    #define DSPIR_SO_EXT ".dll"
#else
    #define DSPIR_SO_EXT ".so"
#endif

/* Cache directory */
#define DSPIR_CACHE_DIR ".cache/fastandfourier"

/**
 * @brief Simple hash function for IR bytecode (FNV-1a variant)
 * 
 * This generates a unique identifier for a transform based on its
 * IR bytecode, size, type, and flags. Used for cache filenames.
 */
static uint64_t hash_transform(const dspir_transform *t, uint32_t flags) {
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    
    uint64_t hash = FNV_OFFSET;
    
    /* Hash transform parameters */
    hash ^= t->n;
    hash *= FNV_PRIME;
    hash ^= t->type;
    hash *= FNV_PRIME;
    hash ^= t->precision;
    hash *= FNV_PRIME;
    hash ^= flags;
    hash *= FNV_PRIME;
    
    /* Hash the IR bytecode */
    for (size_t i = 0; i < t->n_inst; i++) {
        hash ^= t->code[i].packed;
        hash *= FNV_PRIME;
        hash ^= t->code[i].a0;
        hash *= FNV_PRIME;
        hash ^= t->code[i].a1;
        hash *= FNV_PRIME;
        hash ^= t->code[i].a2;
        hash *= FNV_PRIME;
    }
    
    return hash;
}

/**
 * @brief Get the cache directory path
 * @param buf Buffer to store path
 * @param bufsize Size of buffer
 * @return 0 on success, -1 on failure
 */
static int get_cache_dir(char *buf, size_t bufsize) {
    const char *home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");  /* Windows */
    if (!home) return -1;
    
    snprintf(buf, bufsize, "%s/%s", home, DSPIR_CACHE_DIR);
    return 0;
}

/**
 * @brief Ensure cache directory exists
 */
static void ensure_cache_dir(void) {
    char path[256];
    if (get_cache_dir(path, sizeof(path)) != 0) return;

    /* Create directory if it doesn't exist (mkdir -p equivalent) */
    struct stat st;
    if (stat(path, &st) != 0) {
        /* Derive parent from cache path by trimming last component */
        char parent[256];
        strncpy(parent, path, sizeof(parent) - 1);
        parent[sizeof(parent) - 1] = '\0';
        char *last_slash = strrchr(parent, '/');
        if (last_slash) {
            *last_slash = '\0';
            mkdir(parent, 0755);
        }
        mkdir(path, 0755);
    }
}

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
    int from_cache;            /**< Loaded from persistent cache */
    
    /* Native JIT state (for future implementation) */
    void *native_code;         /**< Directly generated machine code */
    size_t native_size;        /**< Size of native code */
};

/**
 * @brief Try to load a cached JIT kernel
 * @param ctx JIT context
 * @param t Transform to load kernel for
 * @param flags JIT flags
 * @return 1 if loaded from cache, 0 if not found
 */
static int try_load_cached_kernel(dspir_jit_ctx *ctx, const dspir_transform *t, uint32_t flags) {
    char cache_dir[256];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) return 0;
    
    uint64_t hash = hash_transform(t, flags);
    snprintf(ctx->so_path, sizeof(ctx->so_path), 
             "%s/jit_%016lx%s", cache_dir, hash, DSPIR_SO_EXT);
    
    /* Check if cached file exists */
    struct stat st;
    if (stat(ctx->so_path, &st) != 0) return 0;
    
    /* Try to load the cached library */
    ctx->handle = dlopen(ctx->so_path, RTLD_NOW | RTLD_LOCAL);
    if (!ctx->handle) return 0;
    
    ctx->fn = (dspir_kernel_fn)dlsym(ctx->handle, "dspir_jit_kernel");
    if (!ctx->fn) {
        dlclose(ctx->handle);
        ctx->handle = NULL;
        return 0;
    }
    
    ctx->from_cache = 1;
    ctx->c_path[0] = '\0';  /* No C source for cached kernels */
    JIT_LOG("[JIT] Loaded cached kernel: %s\n", ctx->so_path);
    return 1;
}

/**
 * @brief Save compiled kernel to cache
 * @param ctx JIT context
 * @param t Transform that was compiled
 * @param flags JIT flags
 */
static void save_kernel_to_cache(dspir_jit_ctx *ctx, const dspir_transform *t, uint32_t flags) {
    if (ctx->from_cache) return;  /* Already cached */
    if (!ctx->so_path[0]) return;  /* No SO to cache */
    
    ensure_cache_dir();
    
    char cache_dir[256];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) return;
    
    uint64_t hash = hash_transform(t, flags);
    char cached_so[256];
    snprintf(cached_so, sizeof(cached_so), 
             "%s/jit_%016lx%s", cache_dir, hash, DSPIR_SO_EXT);
    
    /* Copy the compiled .so to cache */
    FILE *src = fopen(ctx->so_path, "rb");
    if (!src) return;
    
    FILE *dst = fopen(cached_so, "wb");
    if (!dst) {
        fclose(src);
        return;
    }
    
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
        fwrite(buf, 1, n, dst);
    }
    
    fclose(src);
    fclose(dst);
    JIT_LOG("[JIT] Saved kernel to cache: %s\n", cached_so);
}

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

/* Dead code removed: find_best_base_size, emit_base_fft_kernel, emit_scalar_bfly2,
 * emit_neon_bfly2, emit_scalar_twiddle_mul were never called from the live path.
 * The live JIT path uses emit_split_bfly2() and emit_split_twiddle_mul(). */

/**
 * @brief Generate split-plane BFLY2 (separate real/imag arrays)
 * 
 * This enables much better auto-vectorization on x86 with AVX-512.
 * Real and imaginary arrays are separate 64-byte aligned streams.
 */
/* SPLIT-PLANE EMITTER v2 – Simplified, always split-plane internally */

static void emit_split_bfly2(FILE *f, uint8_t a0, uint8_t a1) {
    fprintf(f, "    {\n");
    fprintf(f, "        float ar = regs_re[%u], ai = regs_im[%u];\n", a0, a0);
    fprintf(f, "        float br = regs_re[%u], bi = regs_im[%u];\n", a1, a1);
    fprintf(f, "        regs_re[%u] = ar + br; regs_im[%u] = ai + bi;\n", a0, a0);
    fprintf(f, "        regs_re[%u] = ar - br; regs_im[%u] = ai - bi;\n", a1, a1);
    fprintf(f, "    }\n");
}

static void emit_split_twiddle_mul(FILE *f, uint8_t a0, uint8_t a1) {
    fprintf(f, "    {\n");
    fprintf(f, "        float r = regs_re[%u], i = regs_im[%u];\n", a0, a0);
    fprintf(f, "        float wr = tw_re[%u], wi = tw_im[%u];\n", a1, a1);
    fprintf(f, "        regs_re[%u] = r*wr - i*wi;\n", a0);
    fprintf(f, "        regs_im[%u] = r*wi + i*wr;\n", a0);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate NEON split-plane twiddle multiply using FMA (FP32)
 * 
 * Uses NEON fused multiply-accumulate for optimal performance:
 * - vfms_f32 for multiply-subtract (real part)
 * - vfma_f32 for multiply-accumulate (imag part)
 * 
 * Reads twiddle factors from interleaved array.
 */
static void emit_neon_split_twiddle_mul(FILE *f, uint8_t base_idx) {
    fprintf(f, "    { /* NEON split-plane twiddle FP32 at %u */\n", base_idx);
    fprintf(f, "        float32x2_t r = vld1_f32(&regs_re[%u]);\n", base_idx);
    fprintf(f, "        float32x2_t i = vld1_f32(&regs_im[%u]);\n", base_idx);
    /* Load interleaved twiddles and unzip */
    fprintf(f, "        float32x2x2_t tw = vld2_f32(&twiddles[%u*2]);\n", base_idx);
    fprintf(f, "        float32x2_t wr = tw.val[0];\n");
    fprintf(f, "        float32x2_t wi = tw.val[1];\n");
    /* Real part: r*wr - i*wi using FMA */
    fprintf(f, "        float32x2_t t1 = vmul_f32(r, wr);\n");
    fprintf(f, "        float32x2_t res_re = vfms_f32(t1, i, wi);\n");
    /* Imag part: r*wi + i*wr using FMA */
    fprintf(f, "        float32x2_t t2 = vmul_f32(r, wi);\n");
    fprintf(f, "        float32x2_t res_im = vfma_f32(t2, i, wr);\n");
    fprintf(f, "        vst1_f32(&regs_re[%u], res_re);\n", base_idx);
    fprintf(f, "        vst1_f32(&regs_im[%u], res_im);\n", base_idx);
    fprintf(f, "    }\n");
}

/**
 * @brief Generate NEON split-plane twiddle multiply using FMA (FP64)
 * 
 * Uses 128-bit registers with 2 doubles each for higher precision.
 * Matches the reference implementation pattern:
 *   bre = vmlsq_f64(vmulq_f64(xr, cr), xi, ci);  // xr*cr - xi*ci
 *   bim = vfmaq_f64(vmulq_f64(xr, ci), xi, cr);  // xr*ci + xi*cr
 * 
 * Reads twiddle factors from interleaved array.
 */
static void emit_neon_split_twiddle_mul_f64(FILE *f, uint8_t base_idx) {
    fprintf(f, "    { /* NEON split-plane twiddle FP64 at %u */\n", base_idx);
    fprintf(f, "        float64x2_t r = vld1q_f64(&regs_re[%u]);\n", base_idx);
    fprintf(f, "        float64x2_t i = vld1q_f64(&regs_im[%u]);\n", base_idx);
    /* Load interleaved twiddles and unzip */
    fprintf(f, "        float64x2x2_t tw = vld2q_f64(&twiddles[%u*2]);\n", base_idx);
    fprintf(f, "        float64x2_t wr = tw.val[0];\n");
    fprintf(f, "        float64x2_t wi = tw.val[1];\n");
    /* Real part: r*wr - i*wi using FMA */
    fprintf(f, "        float64x2_t t1 = vmulq_f64(r, wr);\n");
    fprintf(f, "        float64x2_t res_re = vfmsq_f64(t1, i, wi);\n");
    /* Imag part: r*wi + i*wr using FMA */
    fprintf(f, "        float64x2_t t2 = vmulq_f64(r, wi);\n");
    fprintf(f, "        float64x2_t res_im = vfmaq_f64(t2, i, wr);\n");
    fprintf(f, "        vst1q_f64(&regs_re[%u], res_re);\n", base_idx);
    fprintf(f, "        vst1q_f64(&regs_im[%u], res_im);\n", base_idx);
    fprintf(f, "    }\n");
}

/* Dead code removed: emit_avx512_bfly2, emit_avx512_twiddle_mul, emit_neon_twiddle_mul
 * were never called from dspir_jit_generate_c_ex(). The live path uses split-plane emitters. */

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
    int split_plane = flags & DSPIR_FLAG_JIT_SPLIT_PLANE;
    
    /* Header */
    fprintf(f, "/* Auto-generated JIT kernel for FastAndFourier */\n");
    fprintf(f, "/* Flags: %s%s%s */\n", 
            use_simd ? "SIMD " : "",
            inplace ? "INPLACE " : "",
            split_plane ? "SPLIT_PLANE" : "");
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
    
    /* Compute maximum register index needed */
    size_t max_reg_idx = 0;
    for (size_t i = 0; i < t->n_inst; i++) {
        const dspir_inst *inst = &t->code[i];
        uint8_t op = DSPIR_GET_OP(inst->packed);
        if (op == DSPIR_NOP || op == DSPIR_END) continue;
        /* FFT_STAGE/DCT_STAGE/DST_STAGE: a0=group_size, a1=stride, a2=tw_step
         * None are register indices; the loop accesses regs 0..t->n-1 */
        if (op == DSPIR_FFT_STAGE || op == DSPIR_DCT_STAGE || op == DSPIR_DST_STAGE) {
            if (t->n > 0 && t->n - 1 > max_reg_idx) max_reg_idx = t->n - 1;
            continue;
        }
        if (inst->a0 > max_reg_idx) max_reg_idx = inst->a0;
        if (op != DSPIR_LOAD_CONST) {
            if (inst->a1 > max_reg_idx) max_reg_idx = inst->a1;
        }
    }
    size_t needed_regs = max_reg_idx + 1;
    
    /* Kernel function signature - always interleaved I/O for API compatibility */
    fprintf(f, "__attribute__((visibility(\"default\"), flatten, hot, aligned(64)))\n");
    fprintf(f, "void dspir_jit_kernel(void *restrict out,\n");
    fprintf(f, "                      const void *restrict in,\n");
    fprintf(f, "                      size_t n,\n");
    fprintf(f, "                      const void *restrict twiddles) {\n");
    fprintf(f, "    (void)n;\n");
    fprintf(f, "    float *restrict const out_f = (float*)out;\n");
    fprintf(f, "    const float *restrict const in_f = (const float*)in;\n");
    fprintf(f, "    const float *restrict const tw = (const float*)twiddles;\n\n");
    
    /* Stack threshold: 64KB for stack allocation */
    const size_t STACK_THRESHOLD = 64 * 1024;
    int use_stack = (needed_regs * sizeof(float) * 2 <= STACK_THRESHOLD);
    
    /* Split-plane register file */
    fprintf(f, "    /* Split-plane register file for %zu complex points */\n", t->n);
    if (use_stack) {
        fprintf(f, "    float regs_re[%zu] __attribute__((aligned(64))) = {0};\n", needed_regs);
        fprintf(f, "    float regs_im[%zu] __attribute__((aligned(64))) = {0};\n\n", needed_regs);
    } else {
        fprintf(f, "    float *regs_re = (float*)aligned_alloc(64, %zu * sizeof(float));\n", needed_regs);
        fprintf(f, "    float *regs_im = (float*)aligned_alloc(64, %zu * sizeof(float));\n", needed_regs);
        fprintf(f, "    if (!regs_re || !regs_im) return;\n");
        fprintf(f, "    for (size_t i = 0; i < %zu; i++) { regs_re[i] = 0.0f; regs_im[i] = 0.0f; }\n\n", needed_regs);
    }
    
    /* Detect runs of LOAD (bit-reversal) and STORE (sequential) for looped emission.
     * Instead of emitting N individual load/store statements, emit compact loops. */
    size_t load_run_end = 0;   /* End of initial LOAD run (0 = no run detected) */
    size_t store_run_start = 0; /* Start of trailing STORE run */
    size_t store_run_end = 0;  /* End of trailing STORE run */
    int load_is_bitrev = 1;    /* Whether LOADs form a bit-reversal pattern */
    size_t load_bits = 0;      /* Number of bits for bit-reversal */

    /* Detect leading LOAD run */
    {
        size_t run = 0;
        while (run < t->n_inst && DSPIR_GET_OP(t->code[run].packed) == DSPIR_LOAD) {
            if (t->code[run].a0 != run) load_is_bitrev = 0;
            run++;
        }
        if (run >= 2) {
            load_run_end = run;
            /* Compute bits for bit-reversal */
            size_t tmp = run;
            while (tmp > 1) { tmp >>= 1; load_bits++; }
            /* Verify it's actually a power of 2 */
            if ((1UL << load_bits) != run) load_is_bitrev = 0;
        }
    }

    /* Detect trailing STORE run */
    if (t->n_inst > 0) {
        size_t end = t->n_inst;
        /* Find END instruction */
        if (DSPIR_GET_OP(t->code[end - 1].packed) == DSPIR_END) end--;
        size_t run_start = end;
        while (run_start > 0 && DSPIR_GET_OP(t->code[run_start - 1].packed) == DSPIR_STORE)
            run_start--;
        if (end - run_start >= 2) {
            /* Verify sequential pattern: a0=i, a1=i for i=0..N-1 */
            int sequential = 1;
            for (size_t j = run_start; j < end; j++) {
                size_t idx = j - run_start;
                if (t->code[j].a0 != idx || t->code[j].a1 != idx) {
                    sequential = 0;
                    break;
                }
            }
            if (sequential) {
                store_run_start = run_start;
                store_run_end = end;
            }
        }
    }

    /* Generate code for each instruction */
    for (size_t i = 0; i < t->n_inst; i++) {
        const dspir_inst *inst = &t->code[i];
        uint8_t op = DSPIR_GET_OP(inst->packed);

        /* Emit looped LOAD for bit-reversal pattern */
        if (i == 0 && load_run_end > 0 && load_is_bitrev) {
            fprintf(f, "    /* Bit-reversal load: %zu points, %zu bits */\n", load_run_end, load_bits);
            fprintf(f, "    for (size_t _i = 0; _i < %zu; _i++) {\n", load_run_end);
            fprintf(f, "        size_t _j = 0, _x = _i;\n");
            fprintf(f, "        for (int _b = 0; _b < %zu; _b++) { _j = (_j << 1) | (_x & 1); _x >>= 1; }\n", load_bits);
            fprintf(f, "        regs_re[_i] = in_f[_j * 2];\n");
            fprintf(f, "        regs_im[_i] = in_f[_j * 2 + 1];\n");
            fprintf(f, "    }\n");
            i = load_run_end - 1;  /* Skip individual LOADs */
            continue;
        }

        /* Emit looped STORE for sequential pattern */
        if (i == store_run_start && store_run_end > store_run_start) {
            size_t count = store_run_end - store_run_start;
            fprintf(f, "    /* Sequential store: %zu points */\n", count);
            fprintf(f, "    for (size_t _i = 0; _i < %zu; _i++) {\n", count);
            fprintf(f, "        out_f[_i * 2] = regs_re[_i];\n");
            fprintf(f, "        out_f[_i * 2 + 1] = regs_im[_i];\n");
            fprintf(f, "    }\n");
            i = store_run_end - 1;  /* Skip individual STOREs */
            continue;
        }

        fprintf(f, "    /* Inst %zu: opcode %d */\n", i, op);

        switch (op) {
            case DSPIR_NOP:
                fprintf(f, "    /* NOP */\n");
                break;

            case DSPIR_LOAD:
                /* Deinterleave on load: regs[inst->a0] = in[inst->a1] */
                fprintf(f, "    regs_re[%u] = in_f[%u];\n", inst->a0, inst->a1 * 2);
                fprintf(f, "    regs_im[%u] = in_f[%u];\n", inst->a0, inst->a1 * 2 + 1);
                break;

            case DSPIR_STORE:
                /* Reinterleave on store: out[inst->a0] = regs[inst->a1] */
                fprintf(f, "    out_f[%u] = regs_re[%u];\n", inst->a0 * 2, inst->a1);
                fprintf(f, "    out_f[%u] = regs_im[%u];\n", inst->a0 * 2 + 1, inst->a1);
                break;
                
            case DSPIR_LOAD_CONST: {
                union { uint32_t u; float f; } c = { .u = inst->a1 };
                fprintf(f, "    regs_re[%u] = %.9ef;\n", inst->a0, c.f);
                break;
            }
                
            case DSPIR_BFLY2:
                /* Always use split-plane butterfly */
                emit_split_bfly2(f, inst->a0, inst->a1);
                break;
                
            case DSPIR_BFLY4:
                fprintf(f, "    {\n");
                fprintf(f, "        float r0r = regs_re[%u], r0i = regs_im[%u];\n", inst->a0, inst->a0);
                fprintf(f, "        float r1r = regs_re[%u], r1i = regs_im[%u];\n", inst->a0 + 1, inst->a0 + 1);
                fprintf(f, "        float r2r = regs_re[%u], r2i = regs_im[%u];\n", inst->a0 + 2, inst->a0 + 2);
                fprintf(f, "        float r3r = regs_re[%u], r3i = regs_im[%u];\n", inst->a0 + 3, inst->a0 + 3);
                fprintf(f, "        float t0r = r0r + r2r, t0i = r0i + r2i;\n");
                fprintf(f, "        float t1r = r0r - r2r, t1i = r0i - r2i;\n");
                fprintf(f, "        float t2r = r1r + r3r, t2i = r1i + r3i;\n");
                fprintf(f, "        float t3r = r1r - r3r, t3i = r1i - r3i;\n");
                fprintf(f, "        regs_re[%u] = t0r + t2r; regs_im[%u] = t0i + t2i;\n", inst->a0, inst->a0);
                fprintf(f, "        regs_re[%u] = t0r - t2r; regs_im[%u] = t0i - t2i;\n", inst->a0 + 1, inst->a0 + 1);
                fprintf(f, "        regs_re[%u] = t1r + t3r; regs_im[%u] = t1i + t3i;\n", inst->a0 + 2, inst->a0 + 2);
                fprintf(f, "        regs_re[%u] = t1r - t3r; regs_im[%u] = t1i - t3i;\n", inst->a0 + 3, inst->a0 + 3);
                fprintf(f, "    }\n");
                break;
                
            case DSPIR_TWIDDLE_MUL:
                /* Always use split-plane twiddle multiply */
                fprintf(f, "    {\n");
                fprintf(f, "        float r = regs_re[%u], i = regs_im[%u];\n", inst->a0, inst->a0);
                fprintf(f, "        float wr = tw[%u*2], wi = tw[%u*2+1];\n", inst->a1, inst->a1);
                fprintf(f, "        regs_re[%u] = r*wr - i*wi;\n", inst->a0);
                fprintf(f, "        regs_im[%u] = r*wi + i*wr;\n", inst->a0);
                fprintf(f, "    }\n");
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
                /* Split-plane complex FFT stage with looped emission
                 * a0 = group_size (radix), a1 = stride, a2 = twiddle_step
                 * Pairs elements at (base+r) and (base+r+half), with
                 * twiddle at index r*tw_step in the interleaved twiddle array.
                 */
                fprintf(f, "    {\n");
                fprintf(f, "        const size_t radix = %u;\n", inst->a0);
                fprintf(f, "        const size_t stride = %u;\n", inst->a1);
                fprintf(f, "        const size_t tw_step = %u;\n", inst->a2);
                fprintf(f, "        const size_t half = radix / 2;\n");
                fprintf(f, "        const size_t ngroups = %zu / (radix * stride);\n", t->n);
                fprintf(f, "        for (size_t g = 0; g < ngroups; g++) {\n");
                fprintf(f, "            size_t base = g * radix * stride;\n");
                fprintf(f, "            for (size_t r = 0; r < half; r++) {\n");
                fprintf(f, "                size_t i1 = base + r * stride;\n");
                fprintf(f, "                size_t i2 = i1 + half * stride;\n");
                fprintf(f, "                float ar = regs_re[i1], ai = regs_im[i1];\n");
                fprintf(f, "                float br = regs_re[i2], bi = regs_im[i2];\n");
                fprintf(f, "                float wr = tw[(r * tw_step) * 2];\n");
                fprintf(f, "                float wi = tw[(r * tw_step) * 2 + 1];\n");
                fprintf(f, "                float tr = br*wr - bi*wi;\n");
                fprintf(f, "                float ti = br*wi + bi*wr;\n");
                fprintf(f, "                regs_re[i1] = ar + tr; regs_im[i1] = ai + ti;\n");
                fprintf(f, "                regs_re[i2] = ar - tr; regs_im[i2] = ai - ti;\n");
                fprintf(f, "            }\n");
                fprintf(f, "        }\n");
                fprintf(f, "    }\n");
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
    if (!use_stack) {
        fprintf(f, "    free(regs_re);\n");
        fprintf(f, "    free(regs_im);\n");
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
    
    ctx->from_cache = 0;
    
    /* Try to load from persistent cache first */
    if (try_load_cached_kernel(ctx, t, flags)) {
        ctx->flags = flags;
        return 0;
    }
    
    /* Store flags for this compilation */
    ctx->flags = flags;
    
    /* Detect architecture */
    dspir_arch_type arch = detect_arch();
    
    /* Generate unique temp file names (thread-safe) */
    static _Atomic int counter = 0;
    int id = atomic_fetch_add(&counter, 1);
    snprintf(ctx->c_path, sizeof(ctx->c_path), "/tmp/dspir_jit_%d_%d.c", getpid(), id);
    snprintf(ctx->so_path, sizeof(ctx->so_path), "/tmp/dspir_jit_%d_%d%s", getpid(), id, DSPIR_SO_EXT);
    
    /* Generate C source */
    dspir_jit_generate_c_ex(t, ctx->c_path, arch, flags);
    
    /* Get file size for debugging */
    struct stat st;
    if (stat(ctx->c_path, &st) == 0) {
        JIT_LOG("[JIT] Generated C source: %.1f KB\n", st.st_size / 1024.0);
    }
    
    /* Compile to shared library */
    JIT_LOG("[JIT] Compiling (this may take a moment)...\n");
    if (dspir_jit_compile_c(ctx, ctx->c_path, ctx->so_path, arch, flags) != 0) {
        fprintf(stderr, "[JIT] Compilation failed\n");
        return -1;
    }
    JIT_LOG("[JIT] Compilation successful\n");
    
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
    
    /* Save to persistent cache for future use */
    save_kernel_to_cache(ctx, t, flags);
    
    return 0;
}

/**
 * @brief Default JIT flags - enable split-plane and SIMD on supported platforms
 * 
 * Split-plane is now the default for maximum performance.
 * INPLACE should be explicitly requested when the caller knows input == output.
 */
static uint32_t get_default_jit_flags(void) {
    uint32_t flags = DSPIR_FLAG_JIT_SPLIT_PLANE;  /* Split-plane by default */
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
 * @brief Execute a transform via JIT with an in-memory kernel cache.
 *
 * First call: compiles (or loads from disk cache) and stores the handle +
 * function pointer in t->jit_cache.  Subsequent calls skip compilation and
 * invoke the cached function pointer directly.
 *
 * A compare-and-swap guards the cache write so concurrent first calls are
 * safe: at most one extra compilation is wasted and no handles are leaked.
 */
int dspir_execute_jit_cached(const dspir_transform *t, void *out, const void *in) {
    if (!t || !out || !in) return -1;

    /* Fast path: kernel already compiled and pinned */
    if (t->jit_cache && t->jit_cache->kernel_fn) {
        ((dspir_kernel_fn)t->jit_cache->kernel_fn)(out, in, t->n, t->twiddles[0]);
        return 0;
    }

    /* Slow path: compile (uses disk cache if available) */
    dspir_jit_ctx *ctx = dspir_jit_create();
    if (!ctx) return -1;

    uint32_t flags = get_default_jit_flags();
    if (dspir_jit_compile_ex(ctx, (dspir_transform *)t, flags) != 0) {
        dspir_jit_destroy(ctx);
        return -1;
    }

    dspir_kernel_fn fn = ctx->fn;
    if (!fn) {
        dspir_jit_destroy(ctx);
        return -1;
    }

    /* Build cache entry and pin it to the transform */
    dspir_jit_cache *cache = (dspir_jit_cache *)calloc(1, sizeof(dspir_jit_cache));
    if (cache) {
        cache->handle    = ctx->handle;
        cache->kernel_fn = (void *)fn;
        /* Steal the handle so dspir_jit_destroy doesn't dlclose it */
        ctx->handle = NULL;

        /* CAS: if another thread compiled concurrently, discard ours cleanly */
        dspir_jit_cache *expected = NULL;
        if (!__sync_bool_compare_and_swap(
                &((dspir_transform *)t)->jit_cache, expected, cache)) {
            /* Lost the race — return the handle to ctx so destroy frees it */
            ctx->handle = cache->handle;
            free(cache);
        }
    }

    fn(out, in, t->n, t->twiddles[0]);
    dspir_jit_destroy(ctx);  /* Cleans up temp files; handle already stolen (or returned) */
    return 0;
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
    
    /* Get library handle */
    void *handle = ctx->handle;
    if (!handle) {
        dspir_jit_destroy(ctx);
        return -1;
    }
    
    /* Get and execute the kernel - kernel handles deinterleave/reinterleave internally */
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
