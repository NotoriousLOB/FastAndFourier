/**
 * @file dspir.h
 * @brief Internal IR definitions for FastAndFourier
 */

#ifndef DSPIR_H
#define DSPIR_H

#include "fastandfourier.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture type enum for JIT (values to avoid conflict with macros) */
typedef enum {
    DSPIR_ARCH_TYPE_GENERIC = 0,
    DSPIR_ARCH_TYPE_X86_64 = 1,
    DSPIR_ARCH_TYPE_AARCH64 = 2,
    DSPIR_ARCH_TYPE_CUDA = 3
} dspir_arch_type;

/* --- IR PACKING MACROS --- */

#define DSPIR_PACK_OP(op) ((uint32_t)(op) & 0xFF)
#define DSPIR_PACK_FLAGS(flags) (((uint32_t)(flags) & 0xFF) << 8)
#define DSPIR_PACK_META(meta) (((uint32_t)(meta) & 0xFFFF) << 16)
#define DSPIR_PACK_INST(op, flags, meta) \
    (DSPIR_PACK_OP(op) | DSPIR_PACK_FLAGS(flags) | DSPIR_PACK_META(meta))

#define DSPIR_GET_OP(packed) ((packed) & 0xFF)
#define DSPIR_GET_FLAGS(packed) (((packed) >> 8) & 0xFF)
#define DSPIR_GET_META(packed) (((packed) >> 16) & 0xFFFF)

/* --- INSTRUCTION FLAGS --- */

#define DSPIR_FLAG_VEC_MASK  0x07  /**< Vector width: 0=scalar, 1=4, 2=8, 3=16, 4=32 */
#define DSPIR_FLAG_PREC_MASK 0x38  /**< Precision override */
#define DSPIR_FLAG_PREDICATED 0x40 /**< Predicated execution */
#define DSPIR_FLAG_VOLATILE  0x80  /**< Volatile memory access */

/* Vector width encoding */
#define DSPIR_VEC_SCALAR 0
#define DSPIR_VEC_4      1  /**< 4 elements (SSE) */
#define DSPIR_VEC_8      2  /**< 8 elements (AVX) */
#define DSPIR_VEC_16     3  /**< 16 elements (AVX-512) */
#define DSPIR_VEC_SVE    4  /**< SVE variable width */

/* --- TRANSFORM FLAGS --- */

#define DSPIR_FLAG_INVERSE     0x0001  /**< Inverse transform */
#define DSPIR_FLAG_NORMALIZE   0x0002  /**< Normalize output */
#define DSPIR_FLAG_INPLACE     0x0004  /**< In-place transform */
#define DSPIR_FLAG_REAL        0x0008  /**< Real input/output */
#define DSPIR_FLAG_PARALLEL    0x0010  /**< Enable parallel execution */
#define DSPIR_FLAG_STREAMING   0x0020  /**< Streaming mode */
#define DSPIR_FLAG_CACHE_OPT   0x0040  /**< Cache-optimized */
#define DSPIR_FLAG_JIT_INPLACE 0x0100  /**< JIT: Work in-place without register file */
#define DSPIR_FLAG_JIT_SIMD    0x0200  /**< JIT: Use SIMD intrinsics */

/* --- VM INTERNALS --- */

/**
 * @brief Direct-threaded dispatch labels (computed goto)
 */
#ifdef DSPIR_USE_DIRECT_THREADED
    #define DSPIR_DISPATCH_TABLE \
        &&label_NOP, &&label_LOAD, &&label_STORE, &&label_LOAD_CONST, \
        &&label_BFLY2, &&label_BFLY4, &&label_BFLY8, \
        &&label_TWIDDLE_MUL, &&label_TWIDDLE_MUL_CONJ, \
        &&label_FMA, &&label_FMUL, &&label_FADD, &&label_FSUB, \
        &&label_BARRIER, &&label_FFT_STAGE, &&label_DCT_STAGE, \
        &&label_DST_STAGE, &&label_LIFT_PRED, &&label_LIFT_UPD, \
        &&label_LIFT_SCALE, &&label_DOWN2, &&label_UP2, \
        &&label_POLY_FIR, &&label_POLY_IIR, &&label_PERMUTE, \
        &&label_SHUFFLE, &&label_BLEND, &&label_REDUCE_SUM, \
        &&label_REDUCE_MAX, &&label_REDUCE_MIN, &&label_END
    
    #define DSPIR_DISPATCH(op) goto *dispatch_table[op]
#else
    #define DSPIR_DISPATCH(op) switch(op)
#endif

/* --- REGISTER FILE --- */

/* Maximum registers - supports up to 64K complex points (128K registers interleaved)
 * This allows transforms up to 2^16 = 65536 points
 */
#define DSPIR_MAX_REGISTERS 131072
#define DSPIR_REG_SIZE_AVX512 64  /**< 16 floats */
#define DSPIR_REG_SIZE_AVX2   32  /**< 8 floats */
#define DSPIR_REG_SIZE_SSE    16  /**< 4 floats */
#define DSPIR_REG_SIZE_NEON   16  /**< 4 floats */

/**
 * @brief Register file for different precisions
 */
typedef union {
    uint8_t  fp8[DSPIR_MAX_REGISTERS * 64];
    uint16_t fp16[DSPIR_MAX_REGISTERS * 32];
    float    fp32[DSPIR_MAX_REGISTERS * 16];
    double   fp64[DSPIR_MAX_REGISTERS * 8];
    int8_t   i8[DSPIR_MAX_REGISTERS * 64];
    int16_t  i16[DSPIR_MAX_REGISTERS * 32];
    int32_t  i32[DSPIR_MAX_REGISTERS * 16];
} dspir_regfile;

/* --- ARCHITECTURE-SPECIFIC CONTEXTS --- */

#ifdef DSPIR_ARCH_X86_64
/**
 * @brief x86_64-specific execution context
 */
typedef struct {
    void *avx512_ctx;
    void *avx2_ctx;
    void *sse_ctx;
    int preferred_width;  /**< Preferred vector width in floats */
} dspir_x86_ctx;
#endif

#ifdef DSPIR_ARCH_AARCH64
/**
 * @brief AArch64-specific execution context
 */
typedef struct {
    void *sve_ctx;
    void *neon_ctx;
    int preferred_width;
    uint64_t sve_vl;  /**< SVE vector length */
} dspir_arm_ctx;
#endif

#ifdef DSPIR_HAVE_CUDA
/**
 * @brief CUDA-specific execution context
 */
typedef struct {
    void *cuda_ctx;
    int device_id;
    int max_threads;
    size_t shared_mem_size;
} dspir_cuda_ctx;
#endif

/* --- INTERNAL FUNCTION DECLARATIONS --- */

/* VM execution */
int dspir_vm_execute_f32(const dspir_transform *t,
                          float *DSPIR_RESTRICT out,
                          const float *DSPIR_RESTRICT in);
int dspir_vm_execute_f64(const dspir_transform *t,
                          double *DSPIR_RESTRICT out,
                          const double *DSPIR_RESTRICT in);

/* x86 kernels */
#ifdef DSPIR_ARCH_X86_64
void dspir_x86_sse_execute_f32(const dspir_transform *t,
                                float *DSPIR_RESTRICT out,
                                const float *DSPIR_RESTRICT in);
void dspir_x86_avx2_execute_f32(const dspir_transform *t,
                                 float *DSPIR_RESTRICT out,
                                 const float *DSPIR_RESTRICT in);
void dspir_x86_avx512_execute_f32(const dspir_transform *t,
                                   float *DSPIR_RESTRICT out,
                                   const float *DSPIR_RESTRICT in);
#endif

/* ARM kernels */
#ifdef DSPIR_ARCH_AARCH64
void dspir_arm_neon_execute_f32(const dspir_transform *t,
                                 float *DSPIR_RESTRICT out,
                                 const float *DSPIR_RESTRICT in);
#ifdef DSPIR_HAVE_SVE
void dspir_arm_sve_execute_f32(const dspir_transform *t,
                                float *DSPIR_RESTRICT out,
                                const float *DSPIR_RESTRICT in);
#endif
#endif

/* CUDA kernels */
#ifdef DSPIR_HAVE_CUDA
void dspir_cuda_execute_f32(const dspir_transform *t,
                             float *DSPIR_RESTRICT out,
                             const float *DSPIR_RESTRICT in);
#endif

/* Bytecode generators */
void dspir_gen_fft_radix2(dspir_transform *t, size_t n, bool inverse);
void dspir_gen_fft_radix4(dspir_transform *t, size_t n, bool inverse);
void dspir_gen_fft_mixed(dspir_transform *t, size_t n, bool inverse);
void dspir_gen_dct_ii(dspir_transform *t, size_t n);
void dspir_gen_dct_iv(dspir_transform *t, size_t n);
void dspir_gen_dst_ii(dspir_transform *t, size_t n);
void dspir_gen_mdct(dspir_transform *t, size_t n);
void dspir_gen_haar(dspir_transform *t, size_t n, size_t levels);
void dspir_gen_daubechies4(dspir_transform *t, size_t n, size_t levels);

/* Twiddle generation */
void dspir_gen_twiddles_f32(float *tw, size_t n, bool inverse);
void dspir_gen_twiddles_f64(double *tw, size_t n, bool inverse);
void dspir_gen_dct_twiddles_f32(float *tw, size_t n, int type);
void dspir_gen_dst_twiddles_f32(float *tw, size_t n, int type);
void dspir_gen_mdct_twiddles_f32(float *tw, size_t n);
void dspir_gen_stft_twiddles_f32(float *tw, size_t n_fft, size_t hop_length, size_t win_length);

/* Window functions */
void dspir_gen_hann_window_f32(float *win, size_t n);
void dspir_gen_hamming_window_f32(float *win, size_t n);
void dspir_gen_blackman_window_f32(float *win, size_t n);
void dspir_gen_kaiser_window_f32(float *win, size_t n, float beta);

/* Wavelet coefficients */
void dspir_gen_haar_coeffs_f32(float *lo, float *hi);
void dspir_gen_daubechies4_coeffs_f32(float *lo, float *hi);
void dspir_gen_cdf97_coeffs_f32(float *lo, float *hi, float *lo_r, float *hi_r);

/* JIT compilers */
#ifdef DSPIR_ARCH_X86_64
int dspir_jit_compile_x86(dspir_jit_ctx *ctx, const dspir_transform *t);
#endif
#ifdef DSPIR_ARCH_AARCH64
int dspir_jit_compile_arm(dspir_jit_ctx *ctx, const dspir_transform *t);
#endif

/* Utility */
size_t dsir_next_power_of_2(size_t n);
int dspir_is_power_of_2(size_t n);
void dspir_bit_reverse_permute_f32(float *data, size_t n);
void dspir_bit_reverse_permute_f64(double *data, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* DSPIR_H */
