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
#ifndef DSPIR_RESTRICT
    #if defined(__cplusplus)
        #if defined(__GNUC__) || defined(__clang__)
            #define DSPIR_RESTRICT __restrict__
        #else
            #define DSPIR_RESTRICT
        #endif
    #else
        #define DSPIR_RESTRICT restrict
    #endif
#endif

/* Version information */
#define FASTANDFOURIER_VERSION_MAJOR 1
#define FASTANDFOURIER_VERSION_MINOR 0
#define FASTANDFOURIER_VERSION_PATCH 0
#define FASTANDFOURIER_VERSION_STRING "1.0.0"

/* Public utility macros */
#define DSPIR_GET_OP(packed)      ((uint8_t)((packed) & 0xFF))
#define DSPIR_GET_FLAGS(packed)   ((uint8_t)(((packed) >> 8) & 0xFF))
#define DSPIR_GET_META(packed)    ((uint16_t)(((packed) >> 16) & 0xFFFF))

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define DSPIR_ARCH_X86_64
    #if defined(__AVX512F__)
        #define DSPIR_HAVE_AVX512
    #endif
    #if defined(__AVX2__)
        #define DSPIR_HAVE_AVX2
    #endif
    #if defined(__SSE4_2__)
        #define DSPIR_HAVE_SSE42
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPIR_ARCH_AARCH64
    #define DSPIR_HAVE_NEON
    #if defined(__ARM_FEATURE_SVE)
        #define DSPIR_HAVE_SVE
    #endif
#endif

/* Precision level detection */
#ifdef DSPIR_ENABLE_FP8
    #define DSPIR_HAS_FP8 1
#endif
#ifdef DSPIR_ENABLE_FP16
    #define DSPIR_HAS_FP16 1
#endif
#define DSPIR_HAS_FP32 1
#define DSPIR_HAS_FP64 1

/* Alignment macros */
#define DSPIR_ALIGN 64
#define DSPIR_ALIGN_ATTR __attribute__((aligned(DSPIR_ALIGN)))
#define DSPIR_CACHE_LINE_SIZE 64

/* --- CORE TYPES --- */

/**
 * @brief Precision levels supported by the library
 */
typedef enum {
    DSPIR_PREC_FP8  = 0,   /**< 8-bit floating point (E4M3/E5M2) */
    DSPIR_PREC_FP16 = 1,   /**< 16-bit floating point (IEEE 754 half) */
    DSPIR_PREC_BF16 = 2,   /**< 16-bit brain floating point */
    DSPIR_PREC_FP32 = 3,   /**< 32-bit single precision */
    DSPIR_PREC_FP64 = 4,   /**< 64-bit double precision */
    DSPIR_PREC_INT8 = 5,   /**< 8-bit integer */
    DSPIR_PREC_INT16 = 6,  /**< 16-bit integer */
    DSPIR_PREC_INT32 = 7,  /**< 32-bit integer */
} dspir_precision;

/**
 * @brief Transform types supported by the library
 */
typedef enum {
    DSPIR_TRANSFORM_FFT,       /**< Fast Fourier Transform */
    DSPIR_TRANSFORM_IFFT,      /**< Inverse FFT */
    DSPIR_TRANSFORM_RFFT,      /**< Real FFT */
    DSPIR_TRANSFORM_IRFFT,     /**< Inverse Real FFT */
    DSPIR_TRANSFORM_DCT_I,     /**< Discrete Cosine Transform Type I */
    DSPIR_TRANSFORM_DCT_II,    /**< Discrete Cosine Transform Type II */
    DSPIR_TRANSFORM_DCT_III,   /**< Discrete Cosine Transform Type III */
    DSPIR_TRANSFORM_DCT_IV,    /**< Discrete Cosine Transform Type IV */
    DSPIR_TRANSFORM_DST_I,     /**< Discrete Sine Transform Type I */
    DSPIR_TRANSFORM_DST_II,    /**< Discrete Sine Transform Type II */
    DSPIR_TRANSFORM_DST_III,   /**< Discrete Sine Transform Type III */
    DSPIR_TRANSFORM_DST_IV,    /**< Discrete Sine Transform Type IV */
    DSPIR_TRANSFORM_STFT,      /**< Short-Time Fourier Transform */
    DSPIR_TRANSFORM_MDCT,      /**< Modified Discrete Cosine Transform */
    DSPIR_TRANSFORM_IMDCT,     /**< Inverse MDCT */
    DSPIR_TRANSFORM_HAAR,      /**< Haar Wavelet Transform */
    DSPIR_TRANSFORM_DAUBECHIES4, /**< Daubechies-4 Wavelet Transform */
    DSPIR_TRANSFORM_CDF97,     /**< Cohen-Daubechies-Feauveau 9/7 Wavelet */
} dspir_transform_type;

/**
 * @brief IR opcodes for the virtual machine
 */
typedef enum {
    DSPIR_NOP = 0,           /**< No operation */
    DSPIR_LOAD,              /**< Load from memory to register */
    DSPIR_STORE,             /**< Store from register to memory */
    DSPIR_LOAD_CONST,        /**< Load constant to register */
    DSPIR_BFLY2,             /**< Radix-2 butterfly */
    DSPIR_BFLY4,             /**< Radix-4 butterfly */
    DSPIR_BFLY8,             /**< Radix-8 butterfly */
    DSPIR_TWIDDLE_MUL,       /**< Complex multiplication by twiddle factor */
    DSPIR_TWIDDLE_MUL_CONJ,  /**< Complex multiplication by conjugate twiddle */
    DSPIR_FMA,               /**< Fused multiply-add */
    DSPIR_FMUL,              /**< Floating point multiply */
    DSPIR_FADD,              /**< Floating point add */
    DSPIR_FSUB,              /**< Floating point subtract */
    DSPIR_BARRIER,           /**< Memory barrier / prefetch hint */
    DSPIR_FFT_STAGE,         /**< Complete FFT stage */
    DSPIR_DCT_STAGE,         /**< Complete DCT stage */
    DSPIR_DST_STAGE,         /**< Complete DST stage */
    DSPIR_LIFT_PRED,         /**< Lifting scheme prediction step */
    DSPIR_LIFT_UPD,          /**< Lifting scheme update step */
    DSPIR_LIFT_SCALE,        /**< Lifting scheme scaling step */
    DSPIR_DOWN2,             /**< Downsampling by 2 */
    DSPIR_UP2,               /**< Upsampling by 2 */
    DSPIR_POLY_FIR,          /**< Polyphase FIR filter */
    DSPIR_POLY_IIR,          /**< Polyphase IIR filter */
    DSPIR_PERMUTE,           /**< Register permutation */
    DSPIR_SHUFFLE,           /**< Vector shuffle */
    DSPIR_BLEND,             /**< Vector blend */
    DSPIR_REDUCE_SUM,        /**< Horizontal sum reduction */
    DSPIR_REDUCE_MAX,        /**< Horizontal max reduction */
    DSPIR_REDUCE_MIN,        /**< Horizontal min reduction */
    DSPIR_END = 255          /**< Program end - high value for extensibility */
} dspir_opcode;

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
} dspir_inst;

/**
 * @brief Cached JIT kernel for reuse
 */
typedef struct {
    void *handle;                /**< Dynamic library handle */
    void *kernel_fn;             /**< Compiled kernel function pointer */
    void *native_code;           /**< Directly generated machine code (if not using compiler) */
    char *so_path;               /**< Path to shared object */
    char *c_path;                /**< Path to C source */
} dspir_jit_cache;

/**
 * @brief Complete transform program
 */
typedef struct {
    size_t n;                    /**< Transform size */
    size_t n_inst;               /**< Number of instructions */
    dspir_inst *code;            /**< Instruction array */
    void *twiddles[8];           /**< Twiddle factor tables */
    size_t twiddle_sizes[8];     /**< Size of each twiddle table */
    dspir_precision precision;   /**< Precision level */
    dspir_transform_type type;   /**< Transform type */
    uint32_t flags;              /**< Transform flags */
    dspir_jit_cache *jit_cache;  /**< Cached JIT compilation (NULL if not compiled) */
} dspir_transform;

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
} dspir_vm_ctx;

/**
 * @brief JIT compilation context
 */
typedef struct dspir_jit_ctx dspir_jit_ctx;

/**
 * @brief JIT-compiled kernel function type
 */
typedef void (*dspir_kernel_fn)(void *DSPIR_RESTRICT out, 
                                 const void *DSPIR_RESTRICT in, 
                                 size_t n,
                                 const void *twiddles);

/* --- CORE API --- */

/**
 * @brief Initialize the library
 * @return 0 on success, non-zero on failure
 */
int dspir_init(void);

/**
 * @brief Cleanup library resources
 */
void dspir_cleanup(void);

/**
 * @brief Get library version string
 * @return Version string
 */
const char* dspir_version(void);

/**
 * @brief Get architecture name
 * @return Architecture identifier string
 */
const char* dspir_arch_name(void);

/**
 * @brief Allocate aligned memory for SIMD operations
 * @param size Size in bytes to allocate
 * @return Aligned pointer (64-byte aligned) or NULL on failure
 * @note Use dspir_aligned_free() to free memory allocated by this function
 */
static inline void* dspir_aligned_alloc(size_t size) {
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
 * @brief Free memory allocated by dspir_aligned_alloc
 * @param ptr Pointer to free
 */
static inline void dspir_aligned_free(void* ptr) {
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
dspir_transform* dspir_create_fft(size_t n, 
                                   bool inverse, 
                                   dspir_precision precision,
                                   uint32_t flags);

/**
 * @brief Create a DCT transform
 * @param n Transform size
 * @param type DCT type (I-IV)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
dspir_transform* dspir_create_dct(size_t n,
                                   int type,
                                   dspir_precision precision,
                                   uint32_t flags);

/**
 * @brief Create a DST transform
 * @param n Transform size
 * @param type DST type (I-IV)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
dspir_transform* dspir_create_dst(size_t n,
                                   int type,
                                   dspir_precision precision,
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
dspir_transform* dspir_create_stft(size_t n_fft,
                                    size_t hop_length,
                                    size_t win_length,
                                    dspir_precision precision,
                                    uint32_t flags);

/**
 * @brief Create an MDCT transform
 * @param n Transform size (must be even)
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
dspir_transform* dspir_create_mdct(size_t n,
                                    dspir_precision precision,
                                    uint32_t flags);

/**
 * @brief Create a Haar wavelet transform
 * @param n Transform size (must be power of 2)
 * @param levels Decomposition levels
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
dspir_transform* dspir_create_haar(size_t n,
                                    size_t levels,
                                    dspir_precision precision,
                                    uint32_t flags);

/**
 * @brief Create a Daubechies-4 wavelet transform
 * @param n Transform size (must be power of 2)
 * @param levels Decomposition levels
 * @param precision Precision level
 * @param flags Additional flags
 * @return Transform object or NULL on error
 */
dspir_transform* dspir_create_daubechies4(size_t n,
                                           size_t levels,
                                           dspir_precision precision,
                                           uint32_t flags);

/**
 * @brief Destroy a transform and free resources
 * @param t Transform to destroy
 */
void dspir_destroy_transform(dspir_transform *t);

/* --- EXECUTION --- */

/**
 * @brief Execute transform using interpreted VM (fallback)
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int dspir_execute_vm(const dspir_transform *t, 
                      void *DSPIR_RESTRICT out, 
                      const void *DSPIR_RESTRICT in);

/**
 * @brief Execute transform using JIT-compiled kernel
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int dspir_execute_jit(const dspir_transform *t,
                       void *DSPIR_RESTRICT out,
                       const void *DSPIR_RESTRICT in);

/**
 * @brief Execute transform with automatic backend selection
 * @param t Transform to execute
 * @param out Output buffer
 * @param in Input buffer
 * @return 0 on success, non-zero on error
 */
int dspir_execute(const dspir_transform *t,
                  void *DSPIR_RESTRICT out,
                  const void *DSPIR_RESTRICT in);

/* Type-specific execution functions */
int dspir_execute_f32(const dspir_transform *t, 
                       float *DSPIR_RESTRICT out, 
                       const float *DSPIR_RESTRICT in);
int dspir_execute_f64(const dspir_transform *t,
                       double *DSPIR_RESTRICT out,
                       const double *DSPIR_RESTRICT in);

/* Split-plane execution (separate real/imag arrays for better SIMD) */
int dspir_execute_split_f32(const dspir_transform *t,
                             float *DSPIR_RESTRICT out_re,
                             float *DSPIR_RESTRICT out_im,
                             const float *DSPIR_RESTRICT in_re,
                             const float *DSPIR_RESTRICT in_im);
#ifdef DSPIR_HAS_FP16
    /* _Float16 is a C extension; skip for C++ */
    #ifndef __cplusplus
        typedef _Float16 dspir_f16_t;
        int dspir_execute_f16(const dspir_transform *t,
                               dspir_f16_t *DSPIR_RESTRICT out,
                               const dspir_f16_t *DSPIR_RESTRICT in);
    #endif
#endif
#ifdef DSPIR_HAS_FP8
typedef struct { uint8_t data; } dspir_fp8_t;
int dspir_execute_fp8(const dspir_transform *t,
                       dspir_fp8_t *DSPIR_RESTRICT out,
                       const dspir_fp8_t *DSPIR_RESTRICT in);
#endif

/* --- JIT COMPILATION --- */

/* JIT compilation flags */
#define DSPIR_FLAG_JIT_INPLACE     0x0100  /**< JIT: Work in-place without register file */
#define DSPIR_FLAG_JIT_SIMD        0x0200  /**< JIT: Use SIMD intrinsics */
#define DSPIR_FLAG_JIT_SPLIT_PLANE 0x0400  /**< JIT: Use split real/imag planes for better vectorization */

/**
 * @brief Create a JIT compilation context
 * @return JIT context or NULL on error
 */
dspir_jit_ctx* dspir_jit_create(void);

/**
 * @brief Compile a transform to native code
 * @param ctx JIT context
 * @param t Transform to compile
 * @return 0 on success, non-zero on error
 */
int dspir_jit_compile(dspir_jit_ctx *ctx, dspir_transform *t);

/**
 * @brief Compile a transform to native code with extended options
 * @param ctx JIT context
 * @param t Transform to compile
 * @param flags JIT flags (DSPIR_FLAG_JIT_*)
 * @return 0 on success, non-zero on error
 */
int dspir_jit_compile_ex(dspir_jit_ctx *ctx, dspir_transform *t, uint32_t flags);

/**
 * @brief Get the compiled kernel function
 * @param ctx JIT context
 * @return Kernel function pointer or NULL
 */
dspir_kernel_fn dspir_jit_get_kernel(dspir_jit_ctx *ctx);

/**
 * @brief Destroy JIT context
 * @param ctx JIT context to destroy
 */
void dspir_jit_destroy(dspir_jit_ctx *ctx);

/**
 * @brief Execute transform using JIT with extended options
 * @param t Transform to execute
 * @param out Output buffer (must be aligned)
 * @param in Input buffer (must be aligned)
 * @param flags JIT flags (DSPIR_FLAG_JIT_*)
 * @return 0 on success, non-zero on error
 */
int dspir_execute_jit_ex(const dspir_transform *t, void *out, const void *in, uint32_t flags);

/* --- UTILITY FUNCTIONS --- */

/**
 * @brief Initialize twiddle factors for given size
 * @param tw Buffer to fill (size 2*n for complex)
 * @param n Transform size
 * @param inverse True for inverse transform
 */
void dspir_init_twiddles(void *tw, size_t n, bool inverse);

/**
 * @brief Get required buffer alignment
 * @return Alignment in bytes
 */
size_t dspir_get_alignment(void);

/**
 * @brief Get transform size requirements
 * @param type Transform type
 * @param min_size Minimum size
 * @return Recommended transform size
 */
size_t dspir_get_recommended_size(dspir_transform_type type, size_t min_size);

/**
 * @brief Check if size is supported for transform type
 * @param type Transform type
 * @param n Size to check
 * @return True if supported
 */
bool dspir_is_size_supported(dspir_transform_type type, size_t n);

/**
 * @brief Get size of precision type in bytes
 * @param prec Precision level
 * @return Size in bytes
 */
size_t dspir_precision_size(dspir_precision prec);

/**
 * @brief Get name of precision level
 * @param prec Precision level
 * @return Name string
 */
const char* dspir_precision_name(dspir_precision prec);

/**
 * @brief Get name of transform type
 * @param type Transform type
 * @return Name string
 */
const char* dspir_transform_name(dspir_transform_type type);

/* --- ERROR HANDLING --- */

/**
 * @brief Get last error message
 * @return Error string
 */
const char* dspir_get_error(void);

/**
 * @brief Clear error state
 */
void dspir_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTANDFOURIER_H */
