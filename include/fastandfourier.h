/**
 * @file fastandfourier.h
 * @brief Main header for FastAndFourier DSP Library
 * 
 * FastAndFourier is a high-performance DSP library featuring:
 * - Intermediate Representation (IR) for transform description
 * - JIT compilation to native code
 * - Multi-architecture vectorization (x86_64, aarch64, CUDA)
 * - Multiple precision levels (FP8, FP16, FP32, FP64)
 * - Comprehensive transform support (FFT, DCT, DST, STFT, MDCT, DWT, CWT)
 * 
 * @version 1.1.0
 * @author adri4n <yo@adri4n.net>
 * @license MIT
 */

#ifndef FASTANDFOURIER_H
#define FASTANDFOURIER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define FASTANDFOURIER_VERSION_MINOR 1
#define FASTANDFOURIER_VERSION_PATCH 0
#define FASTANDFOURIER_VERSION_STRING "1.1.0"

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
    #ifndef FAF_ARCH_X86_64
        #define FAF_ARCH_X86_64
    #endif
    #if defined(__AVX512F__)
        #ifndef FAF_HAVE_AVX512
            #define FAF_HAVE_AVX512
        #endif
    #endif
    #if defined(__AVX2__)
        #ifndef FAF_HAVE_AVX2
            #define FAF_HAVE_AVX2
        #endif
    #endif
    #if defined(__SSE4_2__)
        #ifndef FAF_HAVE_SSE42
            #define FAF_HAVE_SSE42
        #endif
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #ifndef FAF_ARCH_AARCH64
        #define FAF_ARCH_AARCH64
    #endif
    #ifndef FAF_HAVE_NEON
        #define FAF_HAVE_NEON
    #endif
    #if defined(__ARM_FEATURE_SVE)
        #ifndef FAF_HAVE_SVE
            #define FAF_HAVE_SVE
        #endif
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
    FAF_TRANSFORM_DAUBECHIES4, /**< Daubechies-4 (D4 / db2) Wavelet Transform */
    FAF_TRANSFORM_CDF53,     /**< CDF 5/3 (LeGall) Wavelet, JPEG 2000 lossless */
    FAF_TRANSFORM_CDF97,     /**< Cohen-Daubechies-Feauveau 9/7 Wavelet */
    FAF_TRANSFORM_SYM4,      /**< Symlet-4 Wavelet */
    FAF_TRANSFORM_PIPELINE,  /**< Chirp fused pipeline (R2C → C → C2R) */
    FAF_TRANSFORM_CWT,      /**< Continuous Wavelet Transform (filter bank) */
    FAF_TRANSFORM_ICWT       /**< Inverse CWT (dual-frame or L1) */
} faf_transform_type;

/**
 * @brief Named discrete wavelet families (lifting or orthonormal filter bank)
 */
typedef enum {
    FAF_WAVELET_HAAR = 0,    /**< Haar / db1, orthonormal */
    FAF_WAVELET_D4,          /**< Daubechies D4 / db2, 4-tap orthonormal */
    FAF_WAVELET_CDF53,       /**< LeGall 5/3, JPEG 2000 lossless */
    FAF_WAVELET_CDF97,       /**< CDF 9/7, JPEG 2000 lossy */
    FAF_WAVELET_SYM4,        /**< Symlet 4, 8-tap orthonormal */
    FAF_WAVELET_COUNT
} faf_wavelet_family;

/** Packed into FAF_DWT_STAGE.a2 */
#define FAF_DWT_FLAG_INVERSE  0x1u

/** FAF_THRESHOLD.a0 */
#define FAF_THRESH_HARD  0u
#define FAF_THRESH_SOFT  1u

/* --- CWT FILTER BANK TYPES --- */

/**
 * @brief Analytic wavelet prototype for the CWT filter bank
 *
 * All families are strictly analytic (zero for ω ≤ 0), zero-mean, and
 * peak-normalized in the frequency domain. Morse with (β, γ) = (20, 3)
 * is the default: MATLAB `cwtfilterbank` TimeBandwidth=60, P² = βγ.
 */
typedef enum {
    FAF_CWT_WAVELET_MORLET  = 0, /**< Analytic Morlet, zero-mean correction */
    FAF_CWT_WAVELET_MORSE   = 1, /**< Generalized Morse (default) */
    FAF_CWT_WAVELET_BUMP    = 2, /**< Compact-support bump */
    FAF_CWT_WAVELET_SHANNON = 3, /**< Ideal one-octave bandpass */
    FAF_CWT_WAVELET_MEYER   = 4  /**< Zero-phase analytic Meyer */
} faf_cwt_wavelet;

/**
 * @brief Per-scale filter normalization (not cosmetic)
 *
 * L1 dilation (ψ̂_s(ξ) = ψ̂(sξ)) preserves sinusoid amplitude across scales
 * and is the default for scalograms and the one-integral inverse.
 * L2 energy-corrects so the Littlewood-Paley sum is ≈ 1 (Plancherel).
 * Bandpass sets each peak to `bandpass_peak` (MATLAB-style display gain).
 * Dual-frame inversion uses the live analysis filters, so it is valid
 * for every kind.
 */
typedef enum {
    FAF_CWT_NORM_L1       = 0, /**< Amplitude-preserving L1 dilation (default) */
    FAF_CWT_NORM_L2       = 1, /**< Energy-corrected; LP sum ≈ 1 */
    FAF_CWT_NORM_BANDPASS = 2  /**< Peak = bandpass_peak (default 2) */
} faf_cwt_norm_kind;

typedef enum {
    FAF_CWT_SCALE_GEOMETRIC = 0, /**< Constant-Q, 2^{1/voices} (default) */
    FAF_CWT_SCALE_LINEAR    = 1  /**< Linear Hz; does not tile, needs ALLOW_UNTILED */
} faf_cwt_scale_kind;

typedef enum {
    FAF_CWT_CENTER_PEAK   = 0, /**< Scale from prototype peak frequency */
    FAF_CWT_CENTER_ENERGY = 1  /**< Scale from energy centroid of |ψ̂|² */
} faf_cwt_center_kind;

typedef enum {
    FAF_CWT_INV_DUAL = 0, /**< Dual-frame (exact to FP noise when LP-certified) */
    FAF_CWT_INV_L1   = 1  /**< Calderón one-integral; L1 banks only, approximate */
} faf_cwt_inverse_kind;

#define FAF_CWT_FLAG_INCLUDE_LOWPASS  0x0001u /**< Residual φ fills the DC hole (default) */
#define FAF_CWT_FLAG_ALLOW_UNTILED    0x0002u /**< Create even if LP certification fails */
#define FAF_CWT_FLAG_VALIDATE_STRICT  0x0020u /**< Fail create on LP/wrap failure (default) */
/* Reserved; ignored in 1.1.0 (SSQ / alternative output layouts). */
#define FAF_CWT_FLAG_FIXED_LOWPASS    0x0004u
#define FAF_CWT_FLAG_KEEP_SPECTRUM    0x0008u
#define FAF_CWT_FLAG_REAL_OUTPUT      0x0010u

/**
 * @brief Buffer / spectrum layout
 *
 * FAF_LAYOUT_DEFAULT lets each create function pick a type-specific default:
 * split for C2C FFT, real for DWT/DCT, hermitian for R2C.
 */
typedef enum {
    FAF_LAYOUT_DEFAULT = 0,  /**< Type-specific default */
    FAF_LAYOUT_SPLIT,        /**< re[n], im[n] — native for C2C */
    FAF_LAYOUT_HERMITIAN,    /**< packed R2C, n/2+1 split planes */
    FAF_LAYOUT_REAL,         /**< float[n] — native for DWT / DCT */
    FAF_LAYOUT_INTERLEAVED   /**< complex float[2n] — opt-in convenience */
} faf_layout;

/**
 * @brief Scaling convention (FFT and wavelets)
 */
typedef enum {
    FAF_NORM_DEFAULT = 0,    /**< Type-specific default */
    FAF_NORM_NONE,           /**< Forward unscaled; inverse 1/n (NumPy) */
    FAF_NORM_ORTHO,          /**< 1/sqrt(n) both ways; Haar/D4/Sym4 default */
    FAF_NORM_FORWARD,        /**< 1/n on forward, unscaled inverse */
    FAF_NORM_LAZY,           /**< Haar split only, no scale */
    FAF_NORM_JPEG2000        /**< CDF 5/3 / 9/7 JPEG 2000 convention */
} faf_norm;

typedef enum {
    FAF_DIR_FORWARD = 0,
    FAF_DIR_INVERSE
} faf_direction;

typedef enum {
    FAF_BACKEND_AUTO = 0,    /**< JIT above threshold, else VM */
    FAF_BACKEND_VM,
    FAF_BACKEND_JIT
} faf_backend;

/**
 * @brief Create-time configuration for `faf_create_cwt` / `faf_create_icwt`
 *
 * Use `faf_cwt_config_init(n)` rather than memset: Morse (β, γ) = (20, 3),
 * L1 norm, geometric scales, 10 voices, residual lowpass, strict LP checks.
 * `n` must be even and 5-smooth. `fmin`/`fmax` ≤ 0 select automatic bounds
 * that stay off Nyquist and cap the coarsest scale.
 */
typedef struct faf_cwt_config {
    size_t              n;             /**< Signal length (even, 5-smooth) */
    double              fs;            /**< Sample rate (default 1) */
    faf_precision       precision;     /**< FP32 or FP64 */
    faf_backend         backend;       /**< Inner RFFT backend */
    faf_cwt_wavelet     wavelet;
    faf_cwt_norm_kind   norm;
    faf_cwt_scale_kind  scale_kind;
    faf_cwt_center_kind center_kind;
    unsigned            voices;        /**< Voices per octave (geometric) */
    double              fmin;          /**< Lowest center frequency; 0 = auto */
    double              fmax;          /**< Highest center frequency; 0 = auto */
    double              morlet_mu;     /**< Morlet carrier (default 6) */
    double              morse_gamma;   /**< Morse γ (default 3) */
    double              morse_beta;    /**< Morse β (default 20, so P² = 60) */
    double              bump_center;
    double              bump_width;
    double              bandpass_peak; /**< Peak gain for BANDPASS (default 2) */
    double              lp_alpha;      /**< Max |A−1| allowed on the certified band */
    double              lp_beta;       /**< Mean |A−1| allowed on the certified band */
    double              lp_floor;      /**< Relative floor used to find the certified band */
    uint32_t            flags;         /**< FAF_CWT_FLAG_* */
} faf_cwt_config;

/**
 * @brief Littlewood-Paley report, filled at bank creation
 *
 * `A[k] = |φ[k]|² + Σ_j |Ψ_L2[j,k]|²` on the packed Hermitian grid.
 * Tightness is `frame_cond = max_A / min_A` on `[k_lo, k_hi]`.
 * The `A` pointer is owned by the transform; it is valid until destroy.
 */
typedef struct faf_cwt_lp_report {
    size_t  n;
    size_t  n_bins;
    size_t  n_scales;
    int     has_lowpass;
    int     passed;
    double  alpha, beta;
    double  max_abs_dev;
    double  mean_abs_dev;
    double  min_A, max_A;
    double  frame_cond;      /**< max_A / min_A on the certified band */
    size_t  k_lo, k_hi;
    size_t  hole_bins_dc;
    size_t  hole_bins_nyq;
    double  max_dc_wavelet;
    double  max_wrap_energy; /**< Energy near t = n/2 of the coarsest kernel */
    double  admissibility_C; /**< ∫ ψ̂(ω)/ω dω, for the L1 one-integral inverse */
    const double *A;
} faf_cwt_lp_report;

/**
 * @brief Shared create-time configuration
 *
 * Pass the same struct to every faf_create_* . Zero / DEFAULT fields take
 * type-specific defaults. Use faf_config_init() rather than memset so
 * precision starts at FP32 (FAF_PREC_FP8 is zero).
 */
typedef struct faf_config {
    size_t            n;          /**< Required: transform length */
    faf_precision     precision;  /**< Default FP32 */
    faf_layout        layout;     /**< Default: type-specific */
    faf_norm          norm;       /**< Default: NONE (FFT) / ORTHO (DWT) */
    faf_direction     dir;        /**< Default FORWARD */
    faf_backend       backend;    /**< Default AUTO */
    uint32_t          flags;      /**< FAF_FLAG_INPLACE, … */

    faf_wavelet_family family;    /**< DWT; ignored otherwise */
    size_t            levels;     /**< DWT; 0 = full log2(n) */
    int               dct_type;   /**< DCT/DST type 1–4; 0 = 2 */
    size_t            hop_length; /**< STFT */
    size_t            win_length; /**< STFT */
} faf_config;

/**
 * @brief Execute-time buffer descriptor
 *
 * For C2C/DWT, in/out.layout must match t->cfg.layout.
 * For R2C forward: in is REAL of length n, out is HERMITIAN (or
 * INTERLEAVED packed) of length n/2+1. Inverse swaps those.
 * im is NULL for REAL. n is the logical length of this buffer
 * (n, or n/2+1 for HERMITIAN / packed interleaved).
 */
typedef struct faf_buffer {
    void      *re;
    void      *im;
    size_t     n;
    faf_layout layout;
} faf_buffer;

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
    FAF_BITREV,            /**< Bit-reversal permutation of n complex samples */
    FAF_DWT_STAGE,         /**< One Mallat DWT/IDWT level (family, length, flags) */
    FAF_THRESHOLD,         /**< Hard/soft threshold of detail coefficients */
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
typedef struct faf_transform {
    size_t n;                    /**< Transform size (real length for R2C) */
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
    size_t levels;               /**< DWT: number of Mallat decomposition levels */
    faf_wavelet_family family; /**< DWT: wavelet family */
    faf_config cfg;            /**< Resolved create-time configuration */
    void *scratch;             /**< Aligned workspace (DWT, packing) */
    size_t scratch_size;       /**< Bytes allocated at scratch */
    struct faf_transform *inner; /**< Nested C2C for R2C (n/2), NULL otherwise */
    struct faf_transform *inner_inv; /**< Nested inverse (fused R2C pipeline) */
    void *user_aux;            /**< Bound spectrum re[] (mul-spectrum) */
    void *user_aux_im;         /**< Bound spectrum im[] */
    size_t user_aux_n;         /**< Bound spectrum length (bins) */
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
    if (size == 0) return NULL;
    /* C11 aligned_alloc requires size to be a multiple of the alignment. */
    size_t aligned_size = (size + (size_t)63) & ~(size_t)63;
#if defined(_WIN32)
    return _aligned_malloc(aligned_size, 64);
#elif defined(__APPLE__) || defined(__FreeBSD__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, aligned_size) != 0) return NULL;
    return ptr;
#else
    return aligned_alloc(64, aligned_size);
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

/* --- CONFIG HELPERS --- */

/**
 * @brief Zero a config and set n, precision=FP32, other fields DEFAULT/FORWARD/AUTO
 */
static inline faf_config faf_config_init(size_t n) {
    faf_config c;
    memset(&c, 0, sizeof(c));
    c.n = n;
    c.precision = FAF_PREC_FP32;
    return c;
}

/**
 * @brief Snapshot the resolved config stored on a transform
 */
static inline faf_config faf_config_from(const faf_transform *t) {
    faf_config c;
    memset(&c, 0, sizeof(c));
    return t ? t->cfg : c;
}

/**
 * @brief Flip dir, keep every other knob
 */
static inline faf_config faf_config_inverse(faf_config cfg) {
    cfg.dir = (cfg.dir == FAF_DIR_INVERSE) ? FAF_DIR_FORWARD : FAF_DIR_INVERSE;
    return cfg;
}

static inline faf_buffer faf_buffer_real(void *data, size_t n) {
    faf_buffer b;
    memset(&b, 0, sizeof(b));
    b.re = data;
    b.n = n;
    b.layout = FAF_LAYOUT_REAL;
    return b;
}

static inline faf_buffer faf_buffer_split(void *re, void *im, size_t n) {
    faf_buffer b;
    memset(&b, 0, sizeof(b));
    b.re = re;
    b.im = im;
    b.n = n;
    b.layout = FAF_LAYOUT_SPLIT;
    return b;
}

static inline faf_buffer faf_buffer_interleaved(void *data, size_t n) {
    faf_buffer b;
    memset(&b, 0, sizeof(b));
    b.re = data;
    b.n = n;
    b.layout = FAF_LAYOUT_INTERLEAVED;
    return b;
}

static inline faf_buffer faf_buffer_hermitian(void *re, void *im, size_t n_bins) {
    faf_buffer b;
    memset(&b, 0, sizeof(b));
    b.re = re;
    b.im = im;
    b.n = n_bins;
    b.layout = FAF_LAYOUT_HERMITIAN;
    return b;
}

const char* faf_layout_name(faf_layout layout);
const char* faf_norm_name(faf_norm norm);

/**
 * @brief Logical spectrum length: n/2+1 for R2C, n otherwise
 */
size_t faf_spectrum_len(const faf_transform *t);

/* --- TRANSFORM CREATION --- */

/**
 * @brief Create a transform of any supported type from one config
 */
faf_transform* faf_create(faf_transform_type type, const faf_config *cfg);

/**
 * @brief Create an FFT (C2C). n must be 5-smooth (2^a 3^b 5^c).
 *
 * Power-of-2 sizes use the existing radix-2 kernel. Other 5-smooth
 * sizes use mixed-radix 2/3/4/5. Default layout SPLIT, default norm NONE.
 */
faf_transform* faf_create_fft(const faf_config *cfg);

/**
 * @brief Create a real FFT (R2C) or its inverse (C2R).
 *
 * n must be even and 5-smooth, >= 2. Default layout HERMITIAN (packed
 * split-plane spectrum of n/2+1 bins); INTERLEAVED packed is opt-in.
 * Default norm NONE (unscaled forward, 1/n on the real inverse).
 * Set cfg->dir = FAF_DIR_INVERSE, or use faf_create_inverse(), for C2R.
 *
 * Forward: REAL x[n] -> HERMITIAN (re[n/2+1], im[n/2+1]).
 * Inverse: HERMITIAN -> REAL y[n].
 */
faf_transform* faf_create_rfft(const faf_config *cfg);

/**
 * @brief Create a DCT. cfg->dct_type is 1–4 (0 means type II).
 */
faf_transform* faf_create_dct(const faf_config *cfg);

/**
 * @brief Create a DST. cfg->dct_type is 1–4 (0 means type II).
 */
faf_transform* faf_create_dst(const faf_config *cfg);

/**
 * @brief Create an STFT. hop_length / win_length of 0 pick defaults.
 */
faf_transform* faf_create_stft(const faf_config *cfg);

/**
 * @brief Create an MDCT. n must be even.
 */
faf_transform* faf_create_mdct(const faf_config *cfg);

/**
 * @brief Create a Haar DWT (family forced to HAAR)
 */
faf_transform* faf_create_haar(const faf_config *cfg);

/**
 * @brief Create a Daubechies-4 / db2 DWT
 */
faf_transform* faf_create_daubechies4(const faf_config *cfg);

/**
 * @brief Create a DWT / IDWT. family and levels come from cfg.
 */
faf_transform* faf_create_dwt(const faf_config *cfg);

faf_transform* faf_create_cdf53(const faf_config *cfg);
faf_transform* faf_create_cdf97(const faf_config *cfg);
faf_transform* faf_create_sym4(const faf_config *cfg);

/**
 * @brief Recreate the inverse of fwd, inheriting every config knob
 */
faf_transform* faf_create_inverse(const faf_transform *fwd);

/**
 * @brief Wavelet family name (e.g. "haar", "cdf97")
 */
const char* faf_wavelet_name(faf_wavelet_family family);

/**
 * @brief Analysis-filter tap count for a family (0 if lifting-only)
 */
int faf_wavelet_taps(faf_wavelet_family family);

/**
 * @brief Parse a family name (haar, d4, db2, daubechies4, cdf53, legall, cdf97, sym4)
 * @return 0 on success
 */
int faf_wavelet_from_name(const char *name, faf_wavelet_family *out);

/* --- CWT FILTER BANK --- */

/**
 * @brief Zero a CWT config and fill Morse / L1 / geometric defaults
 */
faf_cwt_config faf_cwt_config_init(size_t n);

/**
 * @brief Create a CWT analysis transform
 *
 * Forward execute: REAL x[n] → REAL W[n_rows · n], row-major, row 0 is
 * the residual lowpass (if enabled), then coarse → fine wavelet scales.
 * @return Transform or NULL (see `faf_get_error`)
 */
faf_transform* faf_create_cwt(const faf_cwt_config *cfg);

/**
 * @brief Create a CWT inverse transform from a config
 *
 * Dual-frame inverse is valid for every analysis norm. L1 one-integral
 * inverse requires `cfg->norm == FAF_CWT_NORM_L1`. Prefer
 * `faf_create_inverse(fwd)` to inherit the forward bank's resolved config.
 */
faf_transform* faf_create_icwt(const faf_cwt_config *cfg, faf_cwt_inverse_kind kind);

const faf_cwt_lp_report* faf_cwt_bank_report(const faf_transform *t);
size_t faf_cwt_n_scales(const faf_transform *t);
size_t faf_cwt_n_rows(const faf_transform *t);
size_t faf_cwt_n_bins(const faf_transform *t); /**< Packed Hermitian length n/2+1 */
int faf_cwt_has_lowpass(const faf_transform *t);

/**
 * @brief Copy up to `cap` center frequencies (Hz) or scales
 * @return Number of values written, or -1 on error
 */
int faf_cwt_freqs(const faf_transform *t, double *hz, size_t cap);
int faf_cwt_scales(const faf_transform *t, double *s, size_t cap);
void faf_cwt_report_fprint(FILE *fp, const faf_cwt_lp_report *r);

/**
 * @brief Frequency-domain wavelet row (packed Hermitian, length `n/2+1`)
 *
 * `scale_index` is 0 = coarsest wavelet, not counting the lowpass row.
 * The pointer is valid until `faf_destroy_transform`. Do not stride
 * between scales with `n/2+1`; rows are internally padded.
 */
const float  *faf_cwt_psi_f32(const faf_transform *t, size_t scale_index);
const double *faf_cwt_psi_f64(const faf_transform *t, size_t scale_index);
const float  *faf_cwt_phi_f32(const faf_transform *t);
const double *faf_cwt_phi_f64(const faf_transform *t);

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
 * @brief Execute transform.
 *
 * C2C/DWT: in/out.layout must match t->cfg.layout.
 * R2C: forward is REAL -> HERMITIAN (or INTERLEAVED packed);
 * inverse is the reverse. No silent layout conversion.
 */
int faf_execute(const faf_transform *t,
                faf_buffer *out,
                const faf_buffer *in);

/**
 * @brief Explicit interleaved ↔ split converters (edge of the garage only).
 *
 * faf_execute never interleaves or deinterleaves for you. Convert here,
 * then pass buffers whose layout matches t->cfg.layout.
 *
 * interleaved is [re0, im0, re1, im1, ...]; split is re[n], im[n].
 * n is the number of complex samples (or packed Hermitian bins).
 */
int faf_deinterleave_f32(float *FAF_RESTRICT re, float *FAF_RESTRICT im,
                         const float *FAF_RESTRICT interleaved, size_t n);
int faf_interleave_f32(float *FAF_RESTRICT interleaved,
                       const float *FAF_RESTRICT re, const float *FAF_RESTRICT im,
                       size_t n);
int faf_deinterleave_f64(double *FAF_RESTRICT re, double *FAF_RESTRICT im,
                         const double *FAF_RESTRICT interleaved, size_t n);
int faf_interleave_f64(double *FAF_RESTRICT interleaved,
                       const double *FAF_RESTRICT re, const double *FAF_RESTRICT im,
                       size_t n);

/* Interleaved convenience wrappers. Prefer faf_execute + faf_buffer. */
int faf_execute_f32(const faf_transform *t, 
                       float *FAF_RESTRICT out, 
                       const float *FAF_RESTRICT in);
int faf_execute_f64(const faf_transform *t,
                       double *FAF_RESTRICT out,
                       const double *FAF_RESTRICT in);
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

/**
 * @brief Set the thread-local error message (printf-style)
 */
void faf_set_error(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* FASTANDFOURIER_H */
