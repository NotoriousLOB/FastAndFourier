/**
 * @file fastandfourier.h
 * @brief Main header for FastAndFourier DSP Library
 * 
 * FastAndFourier is a high-performance DSP library featuring:
 * - Intermediate Representation (IR) for transform description
 * - JIT compilation to native code
 * - Multi-architecture vectorization (x86_64, aarch64, CUDA)
 * - Multiple precision levels (FP8, FP16, FP32, FP64)
 * - Comprehensive transform support (FFT, DCT, DST, STFT, MDCT, Wavelets)
 * 
 * @version 1.0.0
 * @author FastAndFourier Team
 * @license MIT
 */

#ifndef FASTANDFOURIER_H
#define FASTANDFOURIER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>

/* C/C++ compatibility for restrict keyword */
#ifndef FAF_RESTRICT
    #if defined(__cplusplus)
        #if defined(__GNUC__) || defined(__clang__)
            #define FAF_RESTRICT __restrict__
        #else
            #define FAF_RESTRICT
        #endif
    #else
        #define FAF_RESTRICT restrict
    #endif
#endif

/* Version information */
#define FASTANDFOURIER_VERSION_MAJOR 1
#define FASTANDFOURIER_VERSION_MINOR 0
#define FASTANDFOURIER_VERSION_PATCH 0
#define FASTANDFOURIER_VERSION_STRING "1.0.0"

/* Public utility macros */
#define FAF_GET_OP(packed)      ((uint8_t)((packed) & 0xFF))
#define FAF_GET_FLAGS(packed)   ((uint8_t)(((packed) >> 8) & 0xFF))
#define FAF_GET_META(packed)    ((uint16_t)(((packed) >> 16) & 0xFFFF))

/* Minimum transform size (in points) at which JIT compilation is attempted
 * automatically.  Transforms smaller than this go straight to the VM
 * interpreter; larger ones try JIT first and fall back to VM on failure. */
#ifndef FAF_JIT_AUTO_THRESHOLD
#define FAF_JIT_AUTO_THRESHOLD 128
#endif

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define FAF_ARCH_X86_64
    #if defined(__AVX512F__)
        #define FAF_HAVE_AVX512
    #endif
    #if defined(__AVX2__)
        #define FAF_HAVE_AVX2
    #endif
    #if defined(__SSE4_2__)
        #define FAF_HAVE_SSE42
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define FAF_ARCH_AARCH64
    #define FAF_HAVE_NEON
    #if defined(__ARM_FEATURE_SVE)
        #define FAF_HAVE_SVE
    #endif
#endif

/* Precision level detection */
#ifdef FAF_ENABLE_FP8
    #define FAF_HAS_FP8 1
#endif
#ifdef FAF_ENABLE_FP16
    #define FAF_HAS_FP16 1
#endif
#define FAF_HAS_FP32 1
#define FAF_HAS_FP64 1

/* Alignment macros */
#define FAF_ALIGN 64
#define FAF_ALIGN_ATTR __attribute__((aligned(FAF_ALIGN)))
#define FAF_CACHE_LINE_SIZE 64

/* --- CORE TYPES --- */

/**
 * @brief Precision levels supported by the library
 */
typedef enum {
    FAF_PREC_FP8  = 0,   /**< 8-bit floating point (E4M3/E5M2) */
    FAF_PREC_FP16 = 1,   /**< 16-bit floating point (IEEE 754 half) */
    FAF_PREC_BF16 = 2,   /**< 16-bit brain floating point */
    FAF_PREC_FP32 = 3,   /**< 32-bit single precision */
    FAF_PREC_FP64 = 4,   /**< 64-bit double precision */
    FAF_PREC_INT8 = 5,   /**< 8-bit integer */
    FAF_PREC_INT16 = 6,  /**< 16-bit integer */
    FAF_PREC_INT32 = 7,  /**< 32-bit integer */
} faf_precision;

/**
 * @brief Transform types supported by the library
 */
typedef enum {
    FAF_TRANSFORM_FFT,       /**< Fast Fourier Transform */
    FAF_TRANSFORM_IFFT,      /**< Inverse FFT */
    FAF_TRANSFORM_RFFT,      /**< Real FFT */
    FAF_TRANSFORM_IRFFT,     /**< Inverse Real FFT */
    FAF_TRANSFORM_DCT_I,     /**< Discrete Cosine Transform Type I */
    FAF_TRANSFORM_DCT_II,    /**< Discrete Cosine Transform Type II */
    FAF_TRANSFORM_DCT_III,   /**< Discrete Cosine Transform Type III */
    FAF_TRANSFORM_DCT_IV,    /**< Discrete Cosine Transform Type IV */
    FAF_TRANSFORM_DST_I,     /**< Discrete Sine Transform Type I */
    FAF_TRANSFORM_DST_II,    /**< Discrete Sine Transform Type II */
    FAF_TRANSFORM_DST_III,   /**< Discrete Sine Transform Type III */
    FAF_TRANSFORM_DST_IV,    /**< Discrete Sine Transform Type IV */
    FAF_TRANSFORM_STFT,      /**< Short-Time Fourier Transform */
    FAF_TRANSFORM_MDCT,      /**< Modified Discrete Cosine Transform */
    FAF_TRANSFORM_IMDCT,     /**< Inverse MDCT */
    FAF_TRANSFORM_HAAR,      /**< Haar Wavelet Transform */
    FAF_TRANSFORM_DAUBECHIES4, /**< Daubechies-4 Wavelet Transform */
    FAF_TRANSFORM_CDF97,     /**< Cohen-Daubechies-Feauveau 9/7 Wavelet */
} faf_transform_type;

/**
 * @brief IR opcodes for the virtual machine
 */
typedef enum {
    FAF_NOP = 0,           /**< No operation */
    FAF_LOAD,              /**< Load from memory to register */
    FAF_STORE,             /**< Store from register to memory */
    FAF_LOAD_CONST,        /**< Load constant to register */
    FAF_BFLY2,             /**< Radix-2 butterfly */
    FAF_BFLY4,             /**< Radix-4 butterfly */
    FAF_BFLY8,             /**< Radix-8 butterfly */
    FAF_TWIDDLE_MUL,       /**< Complex multiplication by twiddle factor */
    FAF_TWIDDLE_MUL_CONJ,  /**< Complex multiplication by conjugate twiddle */
    FAF_FMA,               /**< Fused multiply-add */
    FAF_FMUL,              /**< Floating point multiply */
    FAF_FADD,              /**< Floating point add */
    FAF_FSUB,              /**< Floating point subtract */
    FAF_BARRIER,           /**< Memory barrier / prefetch hint */
    FAF_FFT_STAGE,         /**< Complete FFT stage */
    FAF_DCT_STAGE,         /**< Complete DCT stage */
    FAF_DST_STAGE,         /**< Complete DST stage */
    FAF_LIFT_PRED,         /**< Lifting scheme prediction step */
    FAF_LIFT_UPD,          /**< Lifting scheme update step */
    FAF_LIFT_SCALE,        /**< Lifting scheme scaling step */
    FAF_DOWN2,             /**< Downsampling by 2 */
    FAF_UP2,               /**< Upsampling by 2 */
    FAF_POLY_FIR,          /**< Polyphase FIR filter */
    FAF_POLY_IIR,          /**< Polyphase IIR filter */
    FAF_PERMUTE,           /**< Register permutation */
    FAF_SHUFFLE,           /**< Vector shuffle */
    FAF_BLEND,             /**< Vector blend */
    FAF_REDUCE_SUM,        /**< Horizontal sum reduction */
    FAF_REDUCE_MAX,        /**< Horizontal max reduction */
    FAF_REDUCE_MIN,        /**< Horizontal min reduction */
    FAF_CALL_BUILTIN,      /**< Call registered built-in function (Chirp DSL) */
    FAF_END = 255          /**< Program end - high value for extensibility */
} faf_opcode;

/**
 * @brief Single IR instruction
 * 
 * Packed format for efficient execution:
 * - packed: [opcode:8][flags:8][reserved:16]
 * - a0, a1, a2: Arguments (register indices, memory offsets, or immediates)
 */
typedef struct {
    uint32_t packed;
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
} faf_inst;

/**
 * @brief Cached JIT kernel for reuse
 */
typedef struct {
    void *handle;                /**< Dynamic library handle */
    void *kernel_fn;             /**< Compiled kernel function pointer */
    void *native_code;           /**< Directly generated machine code (if not using compiler) */
    char *so_path;               /**< Path to shared object */
    char *c_path;                /**< Path to C source */
} faf_jit_cache;

/**
 * @brief Complete transform program
 */
typedef struct {
    size_t n;                    /**< Transform size */
    size_t n_inst;               /**< Number of instructions */
    faf_inst *code;            /**< Instruction array */
    void *twiddles[8];           /**< Twiddle factor tables */
    size_t twiddle_sizes[8];     /**< Size of each twiddle table */
    faf_precision precision;   /**< Precision level */
    faf_transform_type type;   /**< Transform type */
    uint32_t flags;              /**< Transform flags */
    faf_jit_cache *jit_cache;  /**< Cached JIT compilation (NULL if not compiled) */
    size_t hop_length;           /**< STFT: hop between frames (0 if unused) */
    size_t win_length;           /**< STFT: window length (0 if unused) */
} faf_transform;

/**
 * @brief Execution context for the VM
 */
typedef struct {
    void *registers;             /**< Register file */
    size_t reg_count;            /**< Number of registers */
    size_t reg_size;             /**< Size of each register in bytes */
    void *scratch;               /**< Scratch memory */
    size_t scratch_size;         /**< Scratch memory size */
    uint32_t flags;              /**< Execution flags */
} faf_vm_ctx;

/**
 * @brief JIT compilation context
 */
typedef struct faf_jit_ctx faf_jit_ctx;

/**
 * @brief JIT-compiled kernel function type
 */
typedef void (*faf_kernel_fn)(void *FAF_RESTRICT out, 
                                 const void *FAF_RESTRICT in, 
                                 size_t n,
                                 const void *twiddles);

/* --- CORE API --- */

/**
 * @brief Initialize the library
 * @return 0 on success, non-zero on failure
 */
int faf_init(void);

/**
 * @brief Cleanup library resources
 */
void faf_cleanup(void);

/**
 * @brief Get library version string
 * @return Version string
 */
const char* faf_version(void);

/**
 * @brief Get architecture name
 * @return Architecture identifier string
 */
const char* faf_arch_name(void);

/**
 * @brief Allocate aligned memory for SIMD operations
 * @param size Size in bytes to allocate
 * @return Aligned pointer (64-byte aligned) or NULL on failure
 * @note Use faf_aligned_free() to free memory allocated by this function
 */
static inline void* faf_aligned_alloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#elif defined(__APPLE__) || defined(__FreeBSD__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
#else
    return aligned_alloc(64, size);
#endif
}

/**
 * @brief Free memory allocated by faf_aligned_alloc
 * @param ptr Pointer to free
 */
static inline void faf_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* --- TRANSFORM CREATION --- */

/**
 * @brief Create an FFT transform
 * @param n Transform size (must be power of 2)
 * @param inverse True for inverse FFT
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_fft(size_t n, 
                                   bool inverse, 
                                   faf_precision precision,
                                   uint32_t flags);

/**
 * @brief Create a DCT transform
 * @param n Transform size
 * @param type DCT type (I-IV)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_dct(size_t n,
                                   int type,
                                   faf_precision precision,
                                   uint32_t flags);

/**
 * @brief Create a DST transform
 * @param n Transform size
 * @param type DST type (I-IV)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_dst(size_t n,
                                   int type,
                                   faf_precision precision,
                                   uint32_t flags);

/**
 * @brief Create an STFT transform
 * @param n_fft FFT size
 * @param hop_length Hop between frames
 * @param win_length Window length
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_stft(size_t n_fft,
                                    size_t hop_length,
                                    size_t win_length,
                                    faf_precision precision,
                                    uint32_t flags);

/**
 * @brief Create an MDCT transform
 * @param n Transform size (must be even)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_mdct(size_t n,
                                    faf_precision precision,
                                    uint32_t flags);

/**
 * @brief Create a Haar wavelet transform
 * @param n Transform size (must be power of 2)
 * @param levels Decomposition levels
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_haar(size_t n,
                                    size_t levels,
                                    faf_precision precision,
                                    uint32_t flags);

/**
 * @brief Create a Daubechies-4 wavelet transform
 * @param n Transform size (must be power of 2)
 * @param levels Decomposition levels
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
faf_transform* faf_create_daubechies4(size_t n,
                                           size_t levels,
                                           faf_precision precision,
                                           uint32_t flags);

/**
 * @brief Destroy a transform and free resources
 * @param t Transform to destroy
 */
void faf_destroy_transform(faf_transform *t);

/* --- EXECUTION --- */

/**
 * @brief Execute transform using interpreted VM (fallback)
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int faf_execute_vm(const faf_transform *t, 
                      void *FAF_RESTRICT out, 
                      const void *FAF_RESTRICT in);

/**
 * @brief Execute transform using JIT-compiled kernel
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int faf_execute_jit(const faf_transform *t,
                       void *FAF_RESTRICT out,
                       const void *FAF_RESTRICT in);

/**
 * @brief Execute a transform via JIT, caching the compiled kernel on the
 *        transform struct so that subsequent calls pay no compile overhead.
 *
 * On the first call the kernel is compiled (or loaded from the on-disk cache)
 * and pinned to t->jit_cache.  Every later call takes the fast path and
 * invokes the cached function pointer directly.  Falls back to the VM if JIT
 * is unavailable.  Thread-safe: uses a compare-and-swap so concurrent first
 * calls waste one compilation at most, with no handle leak.
 *
 * @param t   Transform to execute (logically mutable via jit_cache)
 * @param out Output buffer
 * @param in  Input buffer
 * @return 0 on success, non-zero on error
 */
int faf_execute_jit_cached(const faf_transform *t,
                              void *FAF_RESTRICT out,
                              const void *FAF_RESTRICT in);

/**
 * @brief Execute transform with automatic backend selection
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int faf_execute(const faf_transform *t,
                  void *FAF_RESTRICT out,
                  const void *FAF_RESTRICT in);

/* Type-specific execution functions */
int faf_execute_f32(const faf_transform *t, 
                       float *FAF_RESTRICT out, 
                       const float *FAF_RESTRICT in);
int faf_execute_f64(const faf_transform *t,
                       double *FAF_RESTRICT out,
                       const double *FAF_RESTRICT in);

/* Split-plane execution (separate real/imag arrays for better SIMD) */
int faf_execute_split_f32(const faf_transform *t,
                             float *FAF_RESTRICT out_re,
                             float *FAF_RESTRICT out_im,
                             const float *FAF_RESTRICT in_re,
                             const float *FAF_RESTRICT in_im);
int faf_execute_split_f64(const faf_transform *t,
                             double *FAF_RESTRICT out_re,
                             double *FAF_RESTRICT out_im,
                             const double *FAF_RESTRICT in_re,
                             const double *FAF_RESTRICT in_im);
#ifdef FAF_HAS_FP16
    /* _Float16 is a C extension; skip for C++ */
    #ifndef __cplusplus
        typedef _Float16 faf_f16_t;
        int faf_execute_f16(const faf_transform *t,
                               faf_f16_t *FAF_RESTRICT out,
                               const faf_f16_t *FAF_RESTRICT in);
    #endif
#endif
#ifdef FAF_HAS_FP8
/* faf_fp8_t is a plain struct, safe in both C and C++ (unlike _Float16) */
typedef struct { uint8_t data; } faf_fp8_t;
int faf_execute_fp8(const faf_transform *t,
                       faf_fp8_t *FAF_RESTRICT out,
                       const faf_fp8_t *FAF_RESTRICT in);
#endif

/* --- JIT COMPILATION --- */

/* JIT compilation flags */
#define FAF_FLAG_JIT_INPLACE     0x0100  /**< JIT: Work in-place without register file */
#define FAF_FLAG_JIT_SIMD        0x0200  /**< JIT: Use SIMD intrinsics */
#define FAF_FLAG_JIT_SPLIT_PLANE 0x0400  /**< JIT: Use split real/imag planes for better vectorization */

/**
 * @brief Create a JIT compilation context
 * @return JIT context or NULL on error
 */
faf_jit_ctx* faf_jit_create(void);

/**
 * @brief Compile a transform to native code
 * @param ctx JIT context
 * @param t Transform to compile
 * @return 0 on success, non-zero on error
 */
int faf_jit_compile(faf_jit_ctx *ctx, faf_transform *t);

/**
 * @brief Compile a transform to native code with extended options
 * @param ctx JIT context
 * @param t Transform to compile
 * @param flags JIT flags (FAF_FLAG_JIT_*)
 * @return 0 on success, non-zero on error
 */
int faf_jit_compile_ex(faf_jit_ctx *ctx, faf_transform *t, uint32_t flags);

/**
 * @brief Get the compiled kernel function
 * @param ctx JIT context
 * @return Kernel function pointer or NULL
 */
faf_kernel_fn faf_jit_get_kernel(faf_jit_ctx *ctx);

/**
 * @brief Destroy JIT context
 * @param ctx JIT context to destroy
 */
void faf_jit_destroy(faf_jit_ctx *ctx);

/**
 * @brief Set JIT logging verbosity
 * @param level 0 = silent (default), 1 = verbose
 */
void faf_jit_set_verbose(int level);

/**
 * @brief Execute transform using JIT with extended options
 * @param t Transform to execute
 * @param out Output buffer (must be aligned)
 * @param in Input buffer (must be aligned)
 * @param flags JIT flags (FAF_FLAG_JIT_*)
 * @return 0 on success, non-zero on error
 */
int faf_execute_jit_ex(const faf_transform *t, void *out, const void *in, uint32_t flags);

/* --- UTILITY FUNCTIONS --- */

/**
 * @brief Initialize twiddle factors for given size
 * @param tw Buffer to fill (size 2*n for complex)
 * @param n Transform size
 * @param inverse True for inverse transform
 */
void faf_init_twiddles(void *tw, size_t n, bool inverse);

/**
 * @brief Get required buffer alignment
 * @return Alignment in bytes
 */
size_t faf_get_alignment(void);

/**
 * @brief Get transform size requirements
 * @param type Transform type
 * @param min_size Minimum size
 * @return Recommended transform size
 */
size_t faf_get_recommended_size(faf_transform_type type, size_t min_size);

/**
 * @brief Check if size is supported for transform type
 * @param type Transform type
 * @param n Size to check
 * @return True if supported
 */
bool faf_is_size_supported(faf_transform_type type, size_t n);

/**
 * @brief Get size of precision type in bytes
 * @param prec Precision level
 * @return Size in bytes
 */
size_t faf_precision_size(faf_precision prec);

/**
 * @brief Get name of precision level
 * @param prec Precision level
 * @return Name string
 */
const char* faf_precision_name(faf_precision prec);

/**
 * @brief Get name of transform type
 * @param type Transform type
 * @return Name string
 */
const char* faf_transform_name(faf_transform_type type);

/* --- ERROR HANDLING --- */

/**
 * @brief Get last error message
 * @return Error string
 */
const char* faf_get_error(void);

/**
 * @brief Clear error state
 */
void faf_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTANDFOURIER_H */
