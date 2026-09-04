/**
 * @file faf.h
 * @brief Internal IR definitions for FastAndFourier
 */

#ifndef FAF_H
#define FAF_H

#include "fastandfourier.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture type enum for JIT (values to avoid conflict with macros) */
typedef enum {
    FAF_ARCH_TYPE_GENERIC = 0,
    FAF_ARCH_TYPE_X86_64 = 1,
    FAF_ARCH_TYPE_AARCH64 = 2,
    FAF_ARCH_TYPE_CUDA = 3
} faf_arch_type;

/* --- IR PACKING MACROS --- */

#define FAF_PACK_OP(op) ((uint32_t)(op) & 0xFF)
#define FAF_PACK_FLAGS(flags) (((uint32_t)(flags) & 0xFF) << 8)
#define FAF_PACK_META(meta) (((uint32_t)(meta) & 0xFFFF) << 16)
#define FAF_PACK_INST(op, flags, meta) \
    (FAF_PACK_OP(op) | FAF_PACK_FLAGS(flags) | FAF_PACK_META(meta))

/* fastandfourier.h already provides these; keep local definitions only when
 * the public header has not been included first. */
#ifndef FAF_GET_OP
#define FAF_GET_OP(packed) ((packed) & 0xFF)
#endif
#ifndef FAF_GET_FLAGS
#define FAF_GET_FLAGS(packed) (((packed) >> 8) & 0xFF)
#endif
#ifndef FAF_GET_META
#define FAF_GET_META(packed) (((packed) >> 16) & 0xFFFF)
#endif

/* --- INSTRUCTION FLAGS --- */

#define FAF_FLAG_VEC_MASK  0x07  /**< Vector width: 0=scalar, 1=4, 2=8, 3=16, 4=32 */
#define FAF_FLAG_PREC_MASK 0x38  /**< Precision override */
#define FAF_FLAG_PREDICATED 0x40 /**< Predicated execution */
#define FAF_FLAG_VOLATILE  0x80  /**< Volatile memory access */

/* Vector width encoding */
#define FAF_VEC_SCALAR 0
#define FAF_VEC_4      1  /**< 4 elements (SSE) */
#define FAF_VEC_8      2  /**< 8 elements (AVX) */
#define FAF_VEC_16     3  /**< 16 elements (AVX-512) */
#define FAF_VEC_SVE    4  /**< SVE variable width */

/* --- TRANSFORM FLAGS --- */

#define FAF_FLAG_INVERSE     0x0001  /**< Inverse transform */
#define FAF_FLAG_NORMALIZE   0x0002  /**< Normalize output */
#define FAF_FLAG_INPLACE     0x0004  /**< In-place transform */
#define FAF_FLAG_REAL        0x0008  /**< Real input/output */
#define FAF_FLAG_PARALLEL    0x0010  /**< Enable parallel execution */
#define FAF_FLAG_STREAMING   0x0020  /**< Streaming mode */
#define FAF_FLAG_CACHE_OPT   0x0040  /**< Cache-optimized */
#define FAF_FLAG_JIT_INPLACE 0x0100  /**< JIT: Work in-place without register file */
#define FAF_FLAG_JIT_SIMD    0x0200  /**< JIT: Use SIMD intrinsics */

/* --- VM INTERNALS --- */

/**
 * @brief Direct-threaded dispatch labels (computed goto)
 */
#ifdef FAF_USE_DIRECT_THREADED
    #define FAF_DISPATCH_TABLE \
        &&label_NOP, &&label_LOAD, &&label_STORE, &&label_LOAD_CONST, \
        &&label_BFLY2, &&label_BFLY4, &&label_BFLY8, \
        &&label_TWIDDLE_MUL, &&label_TWIDDLE_MUL_CONJ, \
        &&label_FMA, &&label_FMUL, &&label_FADD, &&label_FSUB, \
        &&label_BARRIER, &&label_FFT_STAGE, &&label_DCT_STAGE, \
        &&label_DST_STAGE, &&label_LIFT_PRED, &&label_LIFT_UPD, \
        &&label_LIFT_SCALE, &&label_DOWN2, &&label_UP2, \
        &&label_POLY_FIR, &&label_POLY_IIR, &&label_PERMUTE, \
        &&label_SHUFFLE, &&label_BLEND, &&label_REDUCE_SUM, \
        &&label_REDUCE_MAX, &&label_REDUCE_MIN, &&label_CALL_BUILTIN, \
        &&label_BITREV, &&label_DWT_STAGE, &&label_THRESHOLD
    
    #define FAF_DISPATCH(op) goto *dispatch_table[op]
#else
    #define FAF_DISPATCH(op) switch(op)
#endif

/* --- REGISTER FILE --- */

/* Maximum registers - supports up to 64K complex points (128K registers interleaved)
 * This allows transforms up to 2^16 = 65536 points
 */
#define FAF_MAX_REGISTERS 131072
#define FAF_REG_SIZE_AVX512 64  /**< 16 floats */
#define FAF_REG_SIZE_AVX2   32  /**< 8 floats */
#define FAF_REG_SIZE_SSE    16  /**< 4 floats */
#define FAF_REG_SIZE_NEON   16  /**< 4 floats */

/**
 * @brief Register file for different precisions
 */
typedef union {
    uint8_t  fp8[FAF_MAX_REGISTERS * 64];
    uint16_t fp16[FAF_MAX_REGISTERS * 32];
    float    fp32[FAF_MAX_REGISTERS * 16];
    double   fp64[FAF_MAX_REGISTERS * 8];
    int8_t   i8[FAF_MAX_REGISTERS * 64];
    int16_t  i16[FAF_MAX_REGISTERS * 32];
    int32_t  i32[FAF_MAX_REGISTERS * 16];
} faf_regfile;

/* --- ARCHITECTURE-SPECIFIC CONTEXTS --- */

#ifdef FAF_ARCH_X86_64
/**
 * @brief x86_64-specific execution context
 */
typedef struct {
    void *avx512_ctx;
    void *avx2_ctx;
    void *sse_ctx;
    int preferred_width;  /**< Preferred vector width in floats */
} faf_x86_ctx;
#endif

#ifdef FAF_ARCH_AARCH64
/**
 * @brief AArch64-specific execution context
 */
typedef struct {
    void *sve_ctx;
    void *neon_ctx;
    int preferred_width;
    uint64_t sve_vl;  /**< SVE vector length */
} faf_arm_ctx;
#endif

#ifdef FAF_HAVE_CUDA
/**
 * @brief CUDA-specific execution context
 */
typedef struct {
    void *cuda_ctx;
    int device_id;
    int max_threads;
    size_t shared_mem_size;
} faf_cuda_ctx;
#endif

/* --- INTERNAL FUNCTION DECLARATIONS --- */

/* VM execution */
int faf_vm_execute_f32(const faf_transform *t,
                          float *FAF_RESTRICT out,
                          const float *FAF_RESTRICT in);
int faf_vm_execute_f64(const faf_transform *t,
                          double *FAF_RESTRICT out,
                          const double *FAF_RESTRICT in);

/* x86 kernels */
#ifdef FAF_ARCH_X86_64
void faf_x86_sse_execute_f32(const faf_transform *t,
                                float *FAF_RESTRICT out,
                                const float *FAF_RESTRICT in);
void faf_x86_avx2_execute_f32(const faf_transform *t,
                                 float *FAF_RESTRICT out,
                                 const float *FAF_RESTRICT in);
void faf_x86_avx512_execute_f32(const faf_transform *t,
                                   float *FAF_RESTRICT out,
                                   const float *FAF_RESTRICT in);
#endif

/* ARM kernels */
#ifdef FAF_ARCH_AARCH64
void faf_radix2_split_neon_f32(float *re, float *im, size_t n,
                                size_t group, size_t stride, size_t tw_step,
                                const float *tw, size_t ntw);
void faf_radix2_split_neon_f64(double *re, double *im, size_t n,
                                size_t group, size_t stride, size_t tw_step,
                                const double *tw, size_t ntw);
#ifdef FAF_HAVE_SVE
void faf_arm_sve_execute_f32(const faf_transform *t,
                                float *FAF_RESTRICT out,
                                const float *FAF_RESTRICT in);
#endif
#endif

/* CUDA kernels */
#ifdef FAF_HAVE_CUDA
void faf_cuda_execute_f32(const faf_transform *t,
                             float *FAF_RESTRICT out,
                             const float *FAF_RESTRICT in);
#endif

/* Small-N FFT kernels (split-plane, in-place) */
void faf_fft_kernel_2_f32(float *re, float *im, int inverse);
void faf_fft_kernel_3_f32(float *re, float *im, int inverse);
void faf_fft_kernel_4_f32(float *re, float *im, int inverse);
void faf_fft_kernel_5_f32(float *re, float *im, int inverse);
void faf_fft_kernel_6_f32(float *re, float *im, int inverse);
void faf_fft_kernel_7_f32(float *re, float *im, int inverse);
void faf_fft_kernel_8_f32(float *re, float *im, int inverse);
void faf_fft_kernel_9_f32(float *re, float *im, int inverse);
void faf_fft_kernel_10_f32(float *re, float *im, int inverse);
void faf_fft_kernel_12_f32(float *re, float *im, int inverse);
void faf_fft_kernel_2_f64(double *re, double *im, int inverse);
void faf_fft_kernel_3_f64(double *re, double *im, int inverse);
void faf_fft_kernel_4_f64(double *re, double *im, int inverse);
void faf_fft_kernel_5_f64(double *re, double *im, int inverse);
void faf_fft_kernel_6_f64(double *re, double *im, int inverse);
void faf_fft_kernel_7_f64(double *re, double *im, int inverse);
void faf_fft_kernel_8_f64(double *re, double *im, int inverse);
void faf_fft_kernel_9_f64(double *re, double *im, int inverse);
void faf_fft_kernel_10_f64(double *re, double *im, int inverse);
void faf_fft_kernel_12_f64(double *re, double *im, int inverse);
int faf_fft_is_codelet_size(size_t n);
int faf_fft_kernel_execute(const faf_transform *t,
                           void *out_re, void *out_im,
                           const void *in_re, const void *in_im);

/* Split-radix DIF (pot2 N>=16, split-plane) */
size_t faf_sr_twiddle_count(size_t n);
void faf_gen_sr_twiddles_f32(float *tw, size_t n, int inverse);
void faf_gen_sr_twiddles_f64(double *tw, size_t n, int inverse);
int faf_fft_sr_dif_execute(const faf_transform *t,
                           void *out_re, void *out_im,
                           const void *in_re, const void *in_im);
int faf_fft_dit_execute(const faf_transform *t,
                        void *out_re, void *out_im,
                        const void *in_re, const void *in_im);

/* Rader prime FFT */
int faf_is_7_smooth(size_t n);
int faf_is_prime(size_t n);
int faf_rader_eligible(size_t n);
size_t faf_primitive_root(size_t p);
int faf_factor_7smooth(size_t n, int *factors, int *n_factors);
int faf_fft_init_rader(faf_transform *t);
int faf_fft_rader_execute(const faf_transform *t,
                          void *out_re, void *out_im,
                          const void *in_re, const void *in_im);
faf_transform *faf_create_fft_ex(const faf_config *cfg, int allow_7smooth);

/* Bytecode generators */
void faf_gen_fft_radix2(faf_transform *t, size_t n, bool inverse);
void faf_gen_fft_radix4(faf_transform *t, size_t n, bool inverse);
void faf_gen_fft_mixed(faf_transform *t, size_t n, bool inverse);
void faf_gen_dct_ii(faf_transform *t, size_t n);
void faf_gen_dct_iv(faf_transform *t, size_t n);
void faf_gen_dst_ii(faf_transform *t, size_t n);
void faf_gen_mdct(faf_transform *t, size_t n);
void faf_gen_haar(faf_transform *t, size_t n, size_t levels);
void faf_gen_daubechies4(faf_transform *t, size_t n, size_t levels);
void faf_gen_dwt(faf_transform *t, faf_wavelet_family family, size_t n,
                 size_t levels, bool inverse);

/* One Mallat level, in-place on a packed real array of length n (n even). */
void faf_dwt_level_f32(float *x, size_t n, faf_wavelet_family family, int inverse);
void faf_dwt_level_f64(double *x, size_t n, faf_wavelet_family family, int inverse);
void faf_dwt_level_conv_f32(float *x, size_t n, faf_wavelet_family family,
                            faf_wavelet_convention conv, int inverse);
void faf_dwt_level_conv_f64(double *x, size_t n, faf_wavelet_family family,
                            faf_wavelet_convention conv, int inverse);

/* Convention helpers */
faf_wavelet_convention faf_convention_default(faf_wavelet_family family);
int faf_validate_convention(faf_wavelet_family family, faf_wavelet_convention conv);

/* Polyphase FIR DWT backend */
/* work is n samples (create-time scratch). NULL → heap fallback (tests). */
void faf_dwt_polyfir_fwd_f32(float *x, size_t n,
                              const float *h, const float *g, int len,
                              float *work);
void faf_dwt_polyfir_inv_f32(float *x, size_t n,
                              const float *ht, const float *gt, int len,
                              float *work);
void faf_dwt_polyfir_fwd_f64(double *x, size_t n,
                              const double *h, const double *g, int len,
                              double *work);
void faf_dwt_polyfir_inv_f64(double *x, size_t n,
                              const double *ht, const double *gt, int len,
                              double *work);
void faf_dwt_ref_conv_decim_f32(float *x, size_t n,
                                 const float *h, const float *g, int len);
void faf_dwt_ref_conv_decim_f64(double *x, size_t n,
                                 const double *h, const double *g, int len);
int faf_dwt_builtin_taps_f32(faf_wavelet_family fam, faf_wavelet_convention conv,
                             float *h, float *g, int *len_hg,
                             float *ht, float *gt, int *len_syn);
int faf_dwt_builtin_taps_f64(faf_wavelet_family fam, faf_wavelet_convention conv,
                             double *h, double *g, int *len_hg,
                             double *ht, double *gt, int *len_syn);
void faf_dwt_apply_level_f32(float *x, size_t n, const faf_transform *t,
                             faf_wavelet_family fam,
                             faf_wavelet_convention conv, int inverse);
void faf_dwt_apply_level_f64(double *x, size_t n, const faf_transform *t,
                             faf_wavelet_family fam,
                             faf_wavelet_convention conv, int inverse);
int faf_dwt_resolve_backend(faf_wavelet_family family,
                            faf_wavelet_convention conv,
                            faf_dwt_backend requested,
                            faf_dwt_backend *out);
void faf_threshold_band_f32(float *x, size_t start, size_t end, int mode, float lambda);
void faf_threshold_band_f64(double *x, size_t start, size_t end, int mode, double lambda);

/* Number of FFT_STAGE ops (and optional BITREV) Chirp should emit for size n. */
size_t faf_fft_stage_count(size_t n);
void faf_emit_fft_stages(faf_transform *t, size_t n, size_t *idx);

/* Twiddle generation */
void faf_gen_twiddles_f32(float *tw, size_t n, bool inverse);
void faf_gen_twiddles_f64(double *tw, size_t n, bool inverse);
void faf_gen_dct_twiddles_f32(float *tw, size_t n, int type);
void faf_gen_dst_twiddles_f32(float *tw, size_t n, int type);
void faf_gen_mdct_twiddles_f32(float *tw, size_t n);
void faf_gen_stft_twiddles_f32(float *tw, size_t n_fft, size_t hop_length, size_t win_length);

/* Window functions */
void faf_gen_hann_window_f32(float *win, size_t n);
void faf_gen_hamming_window_f32(float *win, size_t n);
void faf_gen_blackman_window_f32(float *win, size_t n);
void faf_gen_kaiser_window_f32(float *win, size_t n, float beta);

/* Wavelet coefficients */
void faf_gen_haar_coeffs_f32(float *lo, float *hi);
void faf_gen_daubechies4_coeffs_f32(float *lo, float *hi);
void faf_gen_cdf53_coeffs_f32(float *lo, float *hi);
void faf_gen_cdf97_coeffs_f32(float *lo, float *hi, float *lo_r, float *hi_r);
void faf_gen_sym4_coeffs_f32(float *lo, float *hi);

/* JIT compilers */
#ifdef FAF_ARCH_X86_64
int faf_jit_compile_x86(faf_jit_ctx *ctx, const faf_transform *t);
#endif
#ifdef FAF_ARCH_AARCH64
int faf_jit_compile_arm(faf_jit_ctx *ctx, const faf_transform *t);
#endif

/* Real FFT pack / unpack execute (called from faf_execute) */
int faf_rfft_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in);

/* Fused Chirp R2C → spectral C → C2R */
int faf_pipeline_execute(const faf_transform *t, faf_buffer *out,
                         const faf_buffer *in);

/* Bluestein (chirp-z) FFT — create-time scratch, no BLUESTEIN opcode. */
int faf_fft_init_bluestein(faf_transform *t);
int faf_bluestein_execute(const faf_transform *t, faf_buffer *out,
                          const faf_buffer *in);

/* Internal split-plane VM. Public execute is faf_execute + faf_buffer. */
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

/* Utility */
size_t dsir_next_power_of_2(size_t n);
int faf_is_power_of_2(size_t n);
int faf_is_5_smooth(size_t n);
size_t faf_next_5_smooth(size_t min);
int faf_factor_5smooth(size_t n, int *factors, int *n_factors);
size_t faf_digit_reverse(size_t i, const int *factors, int n_factors);
void faf_digitrev_split_f32(float *re, float *im, size_t n);
void faf_digitrev_split_f64(double *re, double *im, size_t n);
void faf_gen_twiddles_full_f32(float *tw, size_t n, bool inverse);
void faf_gen_twiddles_full_f64(double *tw, size_t n, bool inverse);

/* Mixed-radix FFT stage (a1 = 3/4/5; a1==1 is the legacy radix-2 path) */
void faf_fft_stage_split_f32(float *re, float *im, size_t n,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             const float *tw, size_t ntw, int inverse);
void faf_fft_stage_split_f64(double *re, double *im, size_t n,
                             uint32_t a0, uint32_t a1, uint32_t a2,
                             const double *tw, size_t ntw, int inverse);
void faf_fft_stage_interleaved_f32(float *regs, size_t n,
                                   uint32_t a0, uint32_t a1, uint32_t a2,
                                   const float *tw, size_t ntw, int inverse);
void faf_fft_stage_interleaved_f64(double *regs, size_t n,
                                   uint32_t a0, uint32_t a1, uint32_t a2,
                                   const double *tw, size_t ntw, int inverse);

void faf_bit_reverse_permute_f32(float *data, size_t n);
void faf_bit_reverse_permute_f64(double *data, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* FAF_H */
