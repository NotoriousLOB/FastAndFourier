/**
 * @file faf_core.c
 * @brief Core library implementation
 */

#include "faf.h"
#include "faf_cwt.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <dlfcn.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Thread-local error buffer */
_Thread_local static char g_error_buf[256] = {0};

/* Architecture info */
static const char* g_arch_name = NULL;

const char* faf_version(void) {
    return FASTANDFOURIER_VERSION_STRING;
}

const char* faf_arch_name(void) {
    if (g_arch_name == NULL) {
        #if defined(FAF_ARCH_X86_64)
            #if defined(FAF_HAVE_AVX512)
                g_arch_name = "x86_64-avx512";
            #elif defined(FAF_HAVE_AVX2)
                g_arch_name = "x86_64-avx2";
            #elif defined(FAF_HAVE_SSE42)
                g_arch_name = "x86_64-sse4.2";
            #else
                g_arch_name = "x86_64-generic";
            #endif
        #elif defined(FAF_ARCH_AARCH64)
            #if defined(FAF_HAVE_SVE)
                g_arch_name = "aarch64-sve";
            #elif defined(FAF_HAVE_NEON)
                g_arch_name = "aarch64-neon";
            #else
                g_arch_name = "aarch64-generic";
            #endif
        #else
            g_arch_name = "unknown";
        #endif
    }
    return g_arch_name;
}

int faf_init(void) {
    /* Initialize any global state */
    g_arch_name = NULL;
    faf_clear_error();
    return 0;
}

void faf_cleanup(void) {
    /* Cleanup global state */
}

const char* faf_get_error(void) {
    return g_error_buf;
}

void faf_clear_error(void) {
    g_error_buf[0] = '\0';
}

void faf_set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, args);
    va_end(args);
}

static void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, args);
    va_end(args);
}

size_t faf_get_alignment(void) {
    return FAF_ALIGN;
}

int faf_deinterleave_f32(float *FAF_RESTRICT re, float *FAF_RESTRICT im,
                         const float *FAF_RESTRICT interleaved, size_t n) {
    if (!re || !im || !interleaved) {
        set_error("deinterleave: null buffer");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        re[i] = interleaved[2 * i];
        im[i] = interleaved[2 * i + 1];
    }
    return 0;
}

int faf_interleave_f32(float *FAF_RESTRICT interleaved,
                       const float *FAF_RESTRICT re, const float *FAF_RESTRICT im,
                       size_t n) {
    if (!interleaved || !re || !im) {
        set_error("interleave: null buffer");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        interleaved[2 * i]     = re[i];
        interleaved[2 * i + 1] = im[i];
    }
    return 0;
}

int faf_deinterleave_f64(double *FAF_RESTRICT re, double *FAF_RESTRICT im,
                         const double *FAF_RESTRICT interleaved, size_t n) {
    if (!re || !im || !interleaved) {
        set_error("deinterleave: null buffer");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        re[i] = interleaved[2 * i];
        im[i] = interleaved[2 * i + 1];
    }
    return 0;
}

int faf_interleave_f64(double *FAF_RESTRICT interleaved,
                       const double *FAF_RESTRICT re, const double *FAF_RESTRICT im,
                       size_t n) {
    if (!interleaved || !re || !im) {
        set_error("interleave: null buffer");
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        interleaved[2 * i]     = re[i];
        interleaved[2 * i + 1] = im[i];
    }
    return 0;
}

void faf_herm_mul_f32(float *yr, float *yi,
                      const float *xr, const float *xi,
                      const float *hr, const float *hi, size_t n_bins) {
    if (!yr || !yi || !xr || !xi || !hr || !hi || n_bins == 0) return;
    yr[0] = xr[0] * hr[0];
    yi[0] = 0.0f;
    for (size_t k = 1; k + 1 < n_bins; k++) {
        float ar = xr[k], ai = xi[k];
        float br = hr[k], bi = hi[k];
        yr[k] = ar * br - ai * bi;
        yi[k] = ar * bi + ai * br;
    }
    if (n_bins > 1) {
        yr[n_bins - 1] = xr[n_bins - 1] * hr[n_bins - 1];
        yi[n_bins - 1] = 0.0f;
    }
}

void faf_herm_mul_f64(double *yr, double *yi,
                      const double *xr, const double *xi,
                      const double *hr, const double *hi, size_t n_bins) {
    if (!yr || !yi || !xr || !xi || !hr || !hi || n_bins == 0) return;
    yr[0] = xr[0] * hr[0];
    yi[0] = 0.0;
    for (size_t k = 1; k + 1 < n_bins; k++) {
        double ar = xr[k], ai = xi[k];
        double br = hr[k], bi = hi[k];
        yr[k] = ar * br - ai * bi;
        yi[k] = ar * bi + ai * br;
    }
    if (n_bins > 1) {
        yr[n_bins - 1] = xr[n_bins - 1] * hr[n_bins - 1];
        yi[n_bins - 1] = 0.0;
    }
}

void faf_herm_mul_conj_f32(float *yr, float *yi,
                           const float *xr, const float *xi,
                           const float *hr, const float *hi, size_t n_bins) {
    if (!yr || !yi || !xr || !xi || !hr || !hi || n_bins == 0) return;
    yr[0] = xr[0] * hr[0];
    yi[0] = 0.0f;
    for (size_t k = 1; k + 1 < n_bins; k++) {
        float ar = xr[k], ai = xi[k];
        float br = hr[k], bi = hi[k];
        yr[k] = ar * br + ai * bi;
        yi[k] = ai * br - ar * bi;
    }
    if (n_bins > 1) {
        yr[n_bins - 1] = xr[n_bins - 1] * hr[n_bins - 1];
        yi[n_bins - 1] = 0.0f;
    }
}

void faf_herm_mul_conj_f64(double *yr, double *yi,
                           const double *xr, const double *xi,
                           const double *hr, const double *hi, size_t n_bins) {
    if (!yr || !yi || !xr || !xi || !hr || !hi || n_bins == 0) return;
    yr[0] = xr[0] * hr[0];
    yi[0] = 0.0;
    for (size_t k = 1; k + 1 < n_bins; k++) {
        double ar = xr[k], ai = xi[k];
        double br = hr[k], bi = hi[k];
        yr[k] = ar * br + ai * bi;
        yi[k] = ai * br - ar * bi;
    }
    if (n_bins > 1) {
        yr[n_bins - 1] = xr[n_bins - 1] * hr[n_bins - 1];
        yi[n_bins - 1] = 0.0;
    }
}

size_t faf_precision_size(faf_precision prec) {
    switch (prec) {
        case FAF_PREC_FP8:  return 1;
        case FAF_PREC_FP16:
        case FAF_PREC_BF16: return 2;
        case FAF_PREC_FP32: return 4;
        case FAF_PREC_FP64: return 8;
        case FAF_PREC_INT8: return 1;
        case FAF_PREC_INT16: return 2;
        case FAF_PREC_INT32: return 4;
        default: return 4;
    }
}

const char* faf_precision_name(faf_precision prec) {
    switch (prec) {
        case FAF_PREC_FP8:  return "fp8";
        case FAF_PREC_FP16: return "fp16";
        case FAF_PREC_BF16: return "bf16";
        case FAF_PREC_FP32: return "fp32";
        case FAF_PREC_FP64: return "fp64";
        case FAF_PREC_INT8: return "int8";
        case FAF_PREC_INT16: return "int16";
        case FAF_PREC_INT32: return "int32";
        default: return "unknown";
    }
}

const char* faf_layout_name(faf_layout layout) {
    switch (layout) {
        case FAF_LAYOUT_DEFAULT:     return "default";
        case FAF_LAYOUT_SPLIT:       return "split";
        case FAF_LAYOUT_HERMITIAN:   return "hermitian";
        case FAF_LAYOUT_REAL:        return "real";
        case FAF_LAYOUT_INTERLEAVED: return "interleaved";
        default: return "unknown";
    }
}

const char* faf_norm_name(faf_norm norm) {
    switch (norm) {
        case FAF_NORM_DEFAULT:  return "default";
        case FAF_NORM_NONE:     return "none";
        case FAF_NORM_ORTHO:    return "ortho";
        case FAF_NORM_FORWARD:  return "forward";
        case FAF_NORM_LAZY:     return "lazy";
        case FAF_NORM_JPEG2000: return "jpeg2000";
        default: return "unknown";
    }
}

size_t faf_spectrum_len(const faf_transform *t) {
    if (!t) return 0;
    if (t->type == FAF_TRANSFORM_RFFT || t->type == FAF_TRANSFORM_IRFFT ||
        t->type == FAF_TRANSFORM_PIPELINE)
        return t->n / 2 + 1;
    return t->n;
}

const char* faf_transform_name(faf_transform_type type) {
    switch (type) {
        case FAF_TRANSFORM_FFT:          return "fft";
        case FAF_TRANSFORM_IFFT:         return "ifft";
        case FAF_TRANSFORM_RFFT:         return "rfft";
        case FAF_TRANSFORM_IRFFT:        return "irfft";
        case FAF_TRANSFORM_DCT_I:        return "dct_i";
        case FAF_TRANSFORM_DCT_II:       return "dct_ii";
        case FAF_TRANSFORM_DCT_III:      return "dct_iii";
        case FAF_TRANSFORM_DCT_IV:       return "dct_iv";
        case FAF_TRANSFORM_DST_I:        return "dst_i";
        case FAF_TRANSFORM_DST_II:       return "dst_ii";
        case FAF_TRANSFORM_DST_III:      return "dst_iii";
        case FAF_TRANSFORM_DST_IV:       return "dst_iv";
        case FAF_TRANSFORM_STFT:         return "stft";
        case FAF_TRANSFORM_MDCT:         return "mdct";
        case FAF_TRANSFORM_IMDCT:        return "imdct";
        case FAF_TRANSFORM_HAAR:         return "haar";
        case FAF_TRANSFORM_DAUBECHIES4:  return "daubechies4";
        case FAF_TRANSFORM_CDF53:        return "cdf53";
        case FAF_TRANSFORM_CDF97:        return "cdf97";
        case FAF_TRANSFORM_SYM4:         return "sym4";
        case FAF_TRANSFORM_PIPELINE:     return "pipeline";
        case FAF_TRANSFORM_CWT:          return "cwt";
        case FAF_TRANSFORM_ICWT:         return "icwt";
        default: return "unknown";
    }
}

size_t dsir_next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    #if SIZE_MAX > 0xFFFFFFFF
    n |= n >> 32;
    #endif
    return n + 1;
}

int faf_is_power_of_2(size_t n) {
    return n && ((n & (n - 1)) == 0);
}

void faf_bit_reverse_permute_f32(float *data, size_t n) {
    size_t bits = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 1;
        bits++;
    }
    
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        size_t x = i;
        for (size_t k = 0; k < bits; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            float tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

void faf_bit_reverse_permute_f64(double *data, size_t n) {
    size_t bits = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 1;
        bits++;
    }
    
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        size_t x = i;
        for (size_t k = 0; k < bits; k++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            double tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

bool faf_is_size_supported(faf_transform_type type, size_t n) {
    if (n == 0) return false;
    
    switch (type) {
        case FAF_TRANSFORM_FFT:
        case FAF_TRANSFORM_IFFT:
            return faf_is_5_smooth(n);
        case FAF_TRANSFORM_HAAR:
        case FAF_TRANSFORM_DAUBECHIES4:
        case FAF_TRANSFORM_CDF53:
        case FAF_TRANSFORM_CDF97:
        case FAF_TRANSFORM_SYM4:
            return faf_is_power_of_2(n);
        case FAF_TRANSFORM_RFFT:
        case FAF_TRANSFORM_IRFFT:
        case FAF_TRANSFORM_CWT:
        case FAF_TRANSFORM_ICWT:
            return faf_is_5_smooth(n) && (n % 2u) == 0 && n >= 2;
        case FAF_TRANSFORM_DCT_I:
            return n >= 2;
        case FAF_TRANSFORM_DCT_II:
        case FAF_TRANSFORM_DCT_III:
        case FAF_TRANSFORM_DCT_IV:
        case FAF_TRANSFORM_DST_I:
        case FAF_TRANSFORM_DST_II:
        case FAF_TRANSFORM_DST_III:
        case FAF_TRANSFORM_DST_IV:
            return n > 0;
        case FAF_TRANSFORM_MDCT:
        case FAF_TRANSFORM_IMDCT:
            return (n % 2) == 0;
        default:
            return true;
    }
}

size_t faf_get_recommended_size(faf_transform_type type, size_t min_size) {
    if (type == FAF_TRANSFORM_FFT || type == FAF_TRANSFORM_IFFT ||
        type == FAF_TRANSFORM_RFFT || type == FAF_TRANSFORM_IRFFT ||
        type == FAF_TRANSFORM_CWT || type == FAF_TRANSFORM_ICWT) {
        size_t n = faf_next_5_smooth(min_size);
        if (type == FAF_TRANSFORM_RFFT || type == FAF_TRANSFORM_IRFFT ||
            type == FAF_TRANSFORM_CWT || type == FAF_TRANSFORM_ICWT) {
            if (n < 2) n = 2;
            if ((n % 2u) != 0) n = faf_next_5_smooth(n + 1);
        }
        return n;
    }
    return dsir_next_power_of_2(min_size);
}

static int alloc_scratch(faf_transform *t, size_t bytes) {
    if (bytes == 0) return 0;
    bytes = (bytes + 63u) & ~(size_t)63u;
    t->scratch = aligned_alloc(64, bytes);
    if (!t->scratch) {
        set_error("Failed to allocate transform scratch");
        return -1;
    }
    t->scratch_size = bytes;
    memset(t->scratch, 0, bytes);
    return 0;
}

static void apply_resolved_config(faf_transform *t, faf_config c) {
    t->cfg = c;
    t->n = c.n;
    t->precision = c.precision;
    t->flags = c.flags;
    if (c.dir == FAF_DIR_INVERSE)
        t->flags |= FAF_FLAG_INVERSE;
    t->family = c.family;
    t->levels = c.levels;
    t->hop_length = c.hop_length;
    t->win_length = c.win_length;
}

int faf_dwt_resolve_backend(faf_wavelet_family family,
                            faf_wavelet_convention conv,
                            faf_dwt_backend requested,
                            faf_dwt_backend *out) {
    faf_dwt_backend b = requested;
    if (b == FAF_DWT_BACKEND_AUTO) {
        if (conv == FAF_CONV_CUSTOM_PR || conv == FAF_CONV_ANALYSIS_ONLY)
            b = FAF_DWT_BACKEND_FIR;
        else if (conv == FAF_CONV_CDF53_INT || conv == FAF_CONV_CDF97_JPEG)
            b = FAF_DWT_BACKEND_LIFT;
        else if (family < FAF_WAVELET_COUNT && faf_wavelet_taps(family) > 0 &&
                 faf_wavelet_taps(family) <= 8)
            b = FAF_DWT_BACKEND_LIFT;
        else if (family == FAF_WAVELET_CDF97)
            b = FAF_DWT_BACKEND_LIFT; /* 9-tap JPEG pair stays lifting */
        else
            b = FAF_DWT_BACKEND_FIR;
    }
    if (b == FAF_DWT_BACKEND_FIR &&
        (conv == FAF_CONV_CDF53_INT || conv == FAF_CONV_CDF97_JPEG)) {
        set_error("DWT backend FIR incompatible with lifting convention '%s'",
                  faf_convention_name(conv));
        return -1;
    }
    if (b == FAF_DWT_BACKEND_LIFT &&
        (conv == FAF_CONV_CUSTOM_PR || conv == FAF_CONV_ANALYSIS_ONLY)) {
        set_error("DWT backend LIFT incompatible with convention '%s' (no factorization)",
                  faf_convention_name(conv));
        return -1;
    }
    if (out) *out = b;
    return 0;
}

static faf_transform* alloc_transform(void) {
    faf_transform *t = calloc(1, sizeof(faf_transform));
    if (!t)
        set_error("Failed to allocate transform");
    return t;
}

/* Transform creation functions */
faf_transform* faf_create_fft(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_fft: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    int use_bluestein = 0;
    if (!faf_is_5_smooth(c.n)) {
        if (c.flags & FAF_FLAG_BLUESTEIN) {
            use_bluestein = 1;
        } else {
            set_error("FFT size must be 5-smooth (2^a 3^b 5^c), got %zu; "
                      "nearest is %zu (or set FAF_FLAG_BLUESTEIN)", c.n,
                      faf_get_recommended_size(FAF_TRANSFORM_FFT, c.n));
            return NULL;
        }
    }
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_SPLIT;
    if (c.norm == FAF_NORM_DEFAULT)
        c.norm = FAF_NORM_NONE;
    if (c.layout == FAF_LAYOUT_REAL || c.layout == FAF_LAYOUT_HERMITIAN) {
        set_error("FFT C2C does not support layout '%s'", faf_layout_name(c.layout));
        return NULL;
    }
    if (c.norm == FAF_NORM_LAZY || c.norm == FAF_NORM_JPEG2000) {
        set_error("FFT does not support norm '%s'", faf_norm_name(c.norm));
        return NULL;
    }
    if (use_bluestein &&
        c.precision != FAF_PREC_FP32 && c.precision != FAF_PREC_FP64) {
        set_error("Bluestein supports FP32 and FP64 only");
        return NULL;
    }

    faf_transform *t = alloc_transform();
    if (!t) return NULL;

    bool inverse = (c.dir == FAF_DIR_INVERSE);
    t->type = inverse ? FAF_TRANSFORM_IFFT : FAF_TRANSFORM_FFT;
    apply_resolved_config(t, c);

    if (use_bluestein) {
        t->flags |= FAF_FLAG_BLUESTEIN;
        t->cfg.flags |= FAF_FLAG_BLUESTEIN;
        if (faf_fft_init_bluestein(t) != 0) {
            faf_destroy_transform(t);
            return NULL;
        }
        return t;
    }

    if (faf_is_power_of_2(c.n)) {
        /* Pow2 path is frozen: existing radix-2 stage emitter. */
        faf_gen_fft_radix2(t, c.n, inverse);
    } else {
        faf_gen_fft_mixed(t, c.n, inverse);
    }
    if (!t->code) {
        set_error("Failed to generate FFT bytecode");
        faf_destroy_transform(t);
        return NULL;
    }
    return t;
}

faf_transform* faf_create_dct(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_dct: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.n == 0) {
        set_error("DCT size must be > 0");
        return NULL;
    }
    int type = c.dct_type ? c.dct_type : 2;
    if (type < 1 || type > 4) {
        set_error("DCT type must be 1-4, got %d", type);
        return NULL;
    }
    c.dct_type = type;
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_REAL;
    if (c.norm == FAF_NORM_DEFAULT)
        c.norm = FAF_NORM_NONE;

    faf_transform *t = alloc_transform();
    if (!t) return NULL;

    switch (type) {
        case 1: t->type = FAF_TRANSFORM_DCT_I; break;
        case 2: t->type = FAF_TRANSFORM_DCT_II; break;
        case 3: t->type = FAF_TRANSFORM_DCT_III; break;
        case 4: t->type = FAF_TRANSFORM_DCT_IV; break;
    }
    apply_resolved_config(t, c);

    switch (type) {
        case 2:
            faf_gen_dct_ii(t, c.n);
            break;
        case 4:
            faf_gen_dct_iv(t, c.n);
            break;
        default:
            faf_gen_dct_ii(t, c.n);
            break;
    }
    return t;
}

faf_transform* faf_create_dst(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_dst: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.n == 0) {
        set_error("DST size must be > 0");
        return NULL;
    }
    int type = c.dct_type ? c.dct_type : 2;
    if (type < 1 || type > 4) {
        set_error("DST type must be 1-4, got %d", type);
        return NULL;
    }
    c.dct_type = type;
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_REAL;
    if (c.norm == FAF_NORM_DEFAULT)
        c.norm = FAF_NORM_NONE;

    faf_transform *t = alloc_transform();
    if (!t) return NULL;

    switch (type) {
        case 1: t->type = FAF_TRANSFORM_DST_I; break;
        case 2: t->type = FAF_TRANSFORM_DST_II; break;
        case 3: t->type = FAF_TRANSFORM_DST_III; break;
        case 4: t->type = FAF_TRANSFORM_DST_IV; break;
    }
    apply_resolved_config(t, c);
    faf_gen_dst_ii(t, c.n);
    return t;
}

faf_transform* faf_create_mdct(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_mdct: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.n % 2 != 0) {
        set_error("MDCT size must be even, got %zu", c.n);
        return NULL;
    }
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_REAL;
    if (c.norm == FAF_NORM_DEFAULT)
        c.norm = FAF_NORM_NONE;

    faf_transform *t = alloc_transform();
    if (!t) return NULL;

    t->type = (c.dir == FAF_DIR_INVERSE) ? FAF_TRANSFORM_IMDCT : FAF_TRANSFORM_MDCT;
    apply_resolved_config(t, c);
    faf_gen_mdct(t, c.n);
    return t;
}

static faf_transform_type family_to_type(faf_wavelet_family family) {
    switch (family) {
        case FAF_WAVELET_HAAR:  return FAF_TRANSFORM_HAAR;
        case FAF_WAVELET_D4:    return FAF_TRANSFORM_DAUBECHIES4;
        case FAF_WAVELET_CDF53: return FAF_TRANSFORM_CDF53;
        case FAF_WAVELET_CDF97: return FAF_TRANSFORM_CDF97;
        case FAF_WAVELET_SYM4:  return FAF_TRANSFORM_SYM4;
        default:                return FAF_TRANSFORM_HAAR;
    }
}

static size_t log2_size(size_t n) {
    size_t bits = 0;
    while (n > 1) { n >>= 1; bits++; }
    return bits;
}

faf_transform* faf_create_dwt(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_dwt: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.flags & FAF_FLAG_BLUESTEIN) {
        set_error("DWT does not support Bluestein");
        return NULL;
    }
    if (!faf_is_power_of_2(c.n) || c.n < 2) {
        set_error("DWT size must be a power of 2 >= 2, got %zu", c.n);
        return NULL;
    }
    if (c.family >= FAF_WAVELET_COUNT) {
        set_error("Unknown wavelet family %d", (int)c.family);
        return NULL;
    }
    if (c.layout == FAF_LAYOUT_DEFAULT)
        c.layout = FAF_LAYOUT_REAL;
    if (c.norm == FAF_NORM_DEFAULT) {
        if (c.family == FAF_WAVELET_CDF53 || c.family == FAF_WAVELET_CDF97)
            c.norm = FAF_NORM_JPEG2000;
        else
            c.norm = FAF_NORM_ORTHO;
    }

    if (c.conv == FAF_CONV_UNSPEC)
        c.conv = faf_convention_default(c.family);
    if (faf_validate_convention(c.family, c.conv) != 0) {
        set_error("DWT convention '%s' incompatible with family '%s'",
                  faf_convention_name(c.conv), faf_wavelet_name(c.family));
        return NULL;
    }
    if (c.norm == FAF_NORM_ORTHO &&
        (c.conv == FAF_CONV_CDF53_INT || c.conv == FAF_CONV_CDF97_JPEG)) {
        set_error("DWT norm ORTHO incompatible with lifting convention '%s'",
                  faf_convention_name(c.conv));
        return NULL;
    }

    if (faf_dwt_resolve_backend(c.family, c.conv, c.dwt_backend, &c.dwt_backend) != 0)
        return NULL;

    size_t max_levels = log2_size(c.n);
    if (c.levels == 0) c.levels = max_levels;
    if (c.levels > max_levels) {
        set_error("DWT levels %zu exceed log2(%zu)=%zu", c.levels, c.n, max_levels);
        return NULL;
    }

    faf_transform *t = alloc_transform();
    if (!t) return NULL;

    t->type = family_to_type(c.family);
    apply_resolved_config(t, c);

    size_t elem = (c.precision == FAF_PREC_FP64) ? sizeof(double) : sizeof(float);
    if (alloc_scratch(t, c.n * elem) != 0) {
        faf_destroy_transform(t);
        return NULL;
    }

    bool inverse = (c.dir == FAF_DIR_INVERSE);
    faf_gen_dwt(t, c.family, c.n, c.levels, inverse);
    if (!t->code) {
        set_error("Failed to generate DWT bytecode");
        faf_destroy_transform(t);
        return NULL;
    }
    return t;
}

faf_transform* faf_create_haar(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_haar: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    c.family = FAF_WAVELET_HAAR;
    return faf_create_dwt(&c);
}

faf_transform* faf_create_daubechies4(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_daubechies4: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    c.family = FAF_WAVELET_D4;
    return faf_create_dwt(&c);
}

faf_transform* faf_create_cdf53(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_cdf53: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    c.family = FAF_WAVELET_CDF53;
    return faf_create_dwt(&c);
}

faf_transform* faf_create_cdf97(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_cdf97: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    c.family = FAF_WAVELET_CDF97;
    return faf_create_dwt(&c);
}

faf_transform* faf_create_sym4(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_sym4: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    c.family = FAF_WAVELET_SYM4;
    return faf_create_dwt(&c);
}

int faf_dwt_set_taps(faf_transform *t,
                     const float *h, const float *g, int len_hg,
                     const float *ht, const float *gt, int len_syn) {
    if (!t || !h || !g || len_hg < 1) {
        set_error("faf_dwt_set_taps: t, h, g required and len_hg >= 1");
        return -1;
    }
    if (t->cfg.conv != FAF_CONV_CUSTOM_PR && t->cfg.conv != FAF_CONV_ANALYSIS_ONLY) {
        set_error("faf_dwt_set_taps: transform convention is '%s', need CUSTOM_PR or ANALYSIS_ONLY",
                  faf_convention_name(t->cfg.conv));
        return -1;
    }
    t->custom_h = h;
    t->custom_g = g;
    t->custom_len_hg = len_hg;
    if (ht && gt) {
        t->custom_ht = ht;
        t->custom_gt = gt;
        t->custom_len_syn = len_syn;
    } else {
        t->custom_ht = NULL;
        t->custom_gt = NULL;
        t->custom_len_syn = 0;
        t->cfg.conv = FAF_CONV_ANALYSIS_ONLY;
    }
    return 0;
}

faf_transform* faf_create_stft(const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create_stft: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    if (c.win_length == 0) c.win_length = c.n;
    if (c.hop_length == 0) c.hop_length = c.win_length / 4;
    if (c.win_length > c.n) {
        set_error("STFT window length %zu exceeds FFT size %zu", c.win_length, c.n);
        return NULL;
    }

    faf_config fft_cfg = c;
    fft_cfg.dir = FAF_DIR_FORWARD;
    faf_transform *t = faf_create_fft(&fft_cfg);
    if (!t) return NULL;

    t->type = FAF_TRANSFORM_STFT;
    t->hop_length = c.hop_length;
    t->win_length = c.win_length;
    t->cfg.hop_length = c.hop_length;
    t->cfg.win_length = c.win_length;

    /* Store Hann window in twiddles[1] */
    float *win = (float*)malloc(c.n * sizeof(float));
    if (!win) {
        faf_destroy_transform(t);
        return NULL;
    }
    faf_gen_hann_window_f32(win, c.win_length);
    for (size_t i = c.win_length; i < c.n; i++) {
        win[i] = 0.0f;
    }
    t->twiddles[1] = win;
    t->twiddle_sizes[1] = c.n;

    return t;
}

faf_transform* faf_create(faf_transform_type type, const faf_config *cfg) {
    if (!cfg) {
        set_error("faf_create: cfg is NULL");
        return NULL;
    }
    faf_config c = *cfg;
    switch (type) {
        case FAF_TRANSFORM_IFFT:
            c.dir = FAF_DIR_INVERSE;
            return faf_create_fft(&c);
        case FAF_TRANSFORM_FFT:
            return faf_create_fft(&c);
        case FAF_TRANSFORM_DCT_I:
            c.dct_type = 1; return faf_create_dct(&c);
        case FAF_TRANSFORM_DCT_II:
            c.dct_type = 2; return faf_create_dct(&c);
        case FAF_TRANSFORM_DCT_III:
            c.dct_type = 3; return faf_create_dct(&c);
        case FAF_TRANSFORM_DCT_IV:
            c.dct_type = 4; return faf_create_dct(&c);
        case FAF_TRANSFORM_DST_I:
            c.dct_type = 1; return faf_create_dst(&c);
        case FAF_TRANSFORM_DST_II:
            c.dct_type = 2; return faf_create_dst(&c);
        case FAF_TRANSFORM_DST_III:
            c.dct_type = 3; return faf_create_dst(&c);
        case FAF_TRANSFORM_DST_IV:
            c.dct_type = 4; return faf_create_dst(&c);
        case FAF_TRANSFORM_STFT:
            return faf_create_stft(&c);
        case FAF_TRANSFORM_IMDCT:
            c.dir = FAF_DIR_INVERSE;
            /* fall through */
        case FAF_TRANSFORM_MDCT:
            return faf_create_mdct(&c);
        case FAF_TRANSFORM_HAAR:
            c.family = FAF_WAVELET_HAAR; return faf_create_dwt(&c);
        case FAF_TRANSFORM_DAUBECHIES4:
            c.family = FAF_WAVELET_D4; return faf_create_dwt(&c);
        case FAF_TRANSFORM_CDF53:
            c.family = FAF_WAVELET_CDF53; return faf_create_dwt(&c);
        case FAF_TRANSFORM_CDF97:
            c.family = FAF_WAVELET_CDF97; return faf_create_dwt(&c);
        case FAF_TRANSFORM_SYM4:
            c.family = FAF_WAVELET_SYM4; return faf_create_dwt(&c);
        case FAF_TRANSFORM_IRFFT:
            c.dir = FAF_DIR_INVERSE;
            /* fall through */
        case FAF_TRANSFORM_RFFT:
            return faf_create_rfft(&c);
        case FAF_TRANSFORM_CWT:
        case FAF_TRANSFORM_ICWT:
            set_error("CWT: use faf_create_cwt / faf_create_icwt");
            return NULL;
        default:
            set_error("Unknown transform type %d", (int)type);
            return NULL;
    }
}

faf_transform* faf_create_inverse(const faf_transform *fwd) {
    if (!fwd) {
        set_error("faf_create_inverse: transform is NULL");
        return NULL;
    }
    faf_config c = faf_config_inverse(fwd->cfg);
    switch (fwd->type) {
        case FAF_TRANSFORM_FFT:
        case FAF_TRANSFORM_IFFT:
            return faf_create_fft(&c);
        case FAF_TRANSFORM_DCT_I:
        case FAF_TRANSFORM_DCT_II:
        case FAF_TRANSFORM_DCT_III:
        case FAF_TRANSFORM_DCT_IV:
            /* II <-> III; I and IV are involutions up to scale */
            if (fwd->type == FAF_TRANSFORM_DCT_II) c.dct_type = 3;
            else if (fwd->type == FAF_TRANSFORM_DCT_III) c.dct_type = 2;
            return faf_create_dct(&c);
        case FAF_TRANSFORM_DST_I:
        case FAF_TRANSFORM_DST_II:
        case FAF_TRANSFORM_DST_III:
        case FAF_TRANSFORM_DST_IV:
            if (fwd->type == FAF_TRANSFORM_DST_II) c.dct_type = 3;
            else if (fwd->type == FAF_TRANSFORM_DST_III) c.dct_type = 2;
            return faf_create_dst(&c);
        case FAF_TRANSFORM_MDCT:
        case FAF_TRANSFORM_IMDCT:
            return faf_create_mdct(&c);
        case FAF_TRANSFORM_STFT:
            return faf_create_stft(&c);
        case FAF_TRANSFORM_HAAR:
        case FAF_TRANSFORM_DAUBECHIES4:
        case FAF_TRANSFORM_CDF53:
        case FAF_TRANSFORM_CDF97:
        case FAF_TRANSFORM_SYM4:
            if (c.conv == FAF_CONV_ANALYSIS_ONLY) {
                set_error("faf_create_inverse: analysis-only bank has no inverse");
                return NULL;
            }
            {
                faf_transform *inv = faf_create_dwt(&c);
                if (inv && fwd->custom_h) {
                    inv->custom_h = fwd->custom_h;
                    inv->custom_g = fwd->custom_g;
                    inv->custom_ht = fwd->custom_ht;
                    inv->custom_gt = fwd->custom_gt;
                    inv->custom_len_hg = fwd->custom_len_hg;
                    inv->custom_len_syn = fwd->custom_len_syn;
                }
                return inv;
            }
        case FAF_TRANSFORM_RFFT:
        case FAF_TRANSFORM_IRFFT:
            return faf_create_rfft(&c);
        case FAF_TRANSFORM_CWT: {
            const faf_cwt_bank *bank = (const faf_cwt_bank *)fwd->user_aux;
            if (!bank) { set_error("faf_create_inverse: CWT has no bank"); return NULL; }
            return faf_create_icwt(&bank->cfg, FAF_CWT_INV_DUAL);
        }
        default:
            set_error("faf_create_inverse: unsupported type %s",
                      faf_transform_name(fwd->type));
            return NULL;
    }
}

void faf_destroy_transform(faf_transform *t) {
    if (!t) return;

    if ((t->type == FAF_TRANSFORM_CWT || t->type == FAF_TRANSFORM_ICWT) && t->user_aux) {
        faf_cwt_bank *bank = (faf_cwt_bank *)t->user_aux;
        if (bank->bank_owned) {
            t->inner = NULL;
            t->inner_inv = NULL;
            faf_cwt_bank_destroy(bank);
        }
        t->user_aux = NULL;
    }

    if (t->inner) {
        faf_destroy_transform(t->inner);
        t->inner = NULL;
    }
    if (t->inner_inv) {
        faf_destroy_transform(t->inner_inv);
        t->inner_inv = NULL;
    }
    
    /* Clean up JIT cache if present */
    if (t->jit_cache) {
        if (t->jit_cache->handle) {
            dlclose(t->jit_cache->handle);
        }
        if (t->jit_cache->so_path) {
            unlink(t->jit_cache->so_path);
            free(t->jit_cache->so_path);
        }
        if (t->jit_cache->c_path) {
            unlink(t->jit_cache->c_path);
            free(t->jit_cache->c_path);
        }
        if (t->jit_cache->native_code) {
            free(t->jit_cache->native_code);
        }
        free(t->jit_cache);
    }
    
    if (t->code) {
        free(t->code);
    }
    
    for (int i = 0; i < 8; i++) {
        if (t->twiddles[i]) {
            free(t->twiddles[i]);
        }
    }

    if (t->scratch) {
        free(t->scratch);
        t->scratch = NULL;
    }

    free(t);
}

static int scale_buffer_f32(faf_buffer *out, float s) {
    if (!out || !out->re || s == 1.0f) return 0;
    size_t n = out->n;
    float *re = (float *)out->re;
    if (out->layout == FAF_LAYOUT_INTERLEAVED) {
        for (size_t i = 0; i < 2 * n; i++) re[i] *= s;
        return 0;
    }
    for (size_t i = 0; i < n; i++) re[i] *= s;
    if (out->im) {
        float *im = (float *)out->im;
        for (size_t i = 0; i < n; i++) im[i] *= s;
    }
    return 0;
}

static int scale_buffer_f64(faf_buffer *out, double s) {
    if (!out || !out->re || s == 1.0) return 0;
    size_t n = out->n;
    double *re = (double *)out->re;
    if (out->layout == FAF_LAYOUT_INTERLEAVED) {
        for (size_t i = 0; i < 2 * n; i++) re[i] *= s;
        return 0;
    }
    for (size_t i = 0; i < n; i++) re[i] *= s;
    if (out->im) {
        double *im = (double *)out->im;
        for (size_t i = 0; i < n; i++) im[i] *= s;
    }
    return 0;
}

static int apply_fft_norm(const faf_transform *t, faf_buffer *out) {
    if (!t || !out) return 0;
    if (t->type != FAF_TRANSFORM_FFT && t->type != FAF_TRANSFORM_IFFT &&
        t->type != FAF_TRANSFORM_RFFT && t->type != FAF_TRANSFORM_IRFFT)
        return 0;
    if (t->n == 0) return 0;
    int inverse = (t->cfg.dir == FAF_DIR_INVERSE) ||
                  (t->type == FAF_TRANSFORM_IFFT) ||
                  (t->type == FAF_TRANSFORM_IRFFT);
    double scale = 1.0;
    switch (t->cfg.norm) {
        case FAF_NORM_NONE:
            if (!inverse) return 0;
            scale = 1.0 / (double)t->n;
            break;
        case FAF_NORM_ORTHO:
            scale = 1.0 / sqrt((double)t->n);
            break;
        case FAF_NORM_FORWARD:
            if (inverse) return 0;
            scale = 1.0 / (double)t->n;
            break;
        default:
            return 0;
    }
    if (t->precision == FAF_PREC_FP64)
        return scale_buffer_f64(out, scale);
    return scale_buffer_f32(out, (float)scale);
}

static int execute_untyped(const faf_transform *t, void *out, const void *in) {
    if (t->cfg.backend != FAF_BACKEND_VM &&
        t->precision == FAF_PREC_FP32 && t->n >= FAF_JIT_AUTO_THRESHOLD) {
        int ret = faf_execute_jit_cached(t, out, in);
        if (ret == 0) return 0;
    }
    if (t->cfg.backend == FAF_BACKEND_JIT) {
        int ret = faf_execute_jit_cached(t, out, in);
        if (ret == 0) return 0;
        if (t->cfg.backend == FAF_BACKEND_JIT && t->cfg.backend != FAF_BACKEND_AUTO) {
            /* fall through to VM if JIT failed */
        }
    }
    return faf_execute_vm(t, out, in);
}

int faf_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in) {
    if (!t || !out || !in) return -1;

    if (t->type == FAF_TRANSFORM_PIPELINE)
        return faf_pipeline_execute(t, out, in);

    if (t->type == FAF_TRANSFORM_RFFT || t->type == FAF_TRANSFORM_IRFFT) {
        int rret = faf_rfft_execute(t, out, in);
        if (rret != 0) return rret;
        return apply_fft_norm(t, out);
    }

    if ((t->flags & FAF_FLAG_BLUESTEIN) && t->inner) {
        int bret = faf_bluestein_execute(t, out, in);
        if (bret != 0) return bret;
        return apply_fft_norm(t, out);
    }

    if (t->type == FAF_TRANSFORM_CWT)
        return faf_cwt_execute(t, out, in);
    if (t->type == FAF_TRANSFORM_ICWT)
        return faf_icwt_execute(t, out, in);

    if (in->layout != t->cfg.layout || out->layout != t->cfg.layout) {
        set_error("buffer layout (%s/%s) does not match transform (%s)",
                  faf_layout_name(in->layout), faf_layout_name(out->layout),
                  faf_layout_name(t->cfg.layout));
        return -1;
    }
    if (!in->re || !out->re) {
        set_error("execute buffers must provide re");
        return -1;
    }

    int ret = -1;
    switch (t->cfg.layout) {
        case FAF_LAYOUT_SPLIT:
        case FAF_LAYOUT_HERMITIAN:
            if (!in->im || !out->im) {
                set_error("split/hermitian layout requires im planes");
                return -1;
            }
            if (t->precision == FAF_PREC_FP64) {
                ret = faf_execute_split_f64(t,
                    (double *)out->re, (double *)out->im,
                    (const double *)in->re, (const double *)in->im);
            } else {
                ret = faf_execute_split_f32(t,
                    (float *)out->re, (float *)out->im,
                    (const float *)in->re, (const float *)in->im);
            }
            break;
        case FAF_LAYOUT_INTERLEAVED:
            ret = execute_untyped(t, out->re, in->re);
            break;
        case FAF_LAYOUT_REAL:
            if (t->precision == FAF_PREC_FP64) {
                ret = faf_execute_f64(t, (double *)out->re, (const double *)in->re);
            } else {
                ret = faf_execute_f32(t, (float *)out->re, (const float *)in->re);
            }
            break;
        default:
            set_error("unsupported layout");
            return -1;
    }
    if (ret != 0) return ret;
    return apply_fft_norm(t, out);
}

int faf_execute_vm(const faf_transform *t, void *restrict out, const void *restrict in) {
    /* STFT: apply window to input before FFT */
    if (t->type == FAF_TRANSFORM_STFT && t->twiddles[1] && t->precision == FAF_PREC_FP32) {
        const float *win = (const float *)t->twiddles[1];
        const float *in_f = (const float *)in;
        float *windowed = (float *)aligned_alloc(64, t->n * 2 * sizeof(float));
        if (!windowed) return -1;
        for (size_t i = 0; i < t->n; i++) {
            windowed[i * 2]     = in_f[i * 2]     * win[i];
            windowed[i * 2 + 1] = in_f[i * 2 + 1] * win[i];
        }
        int ret = faf_execute_f32(t, out, windowed);
        free(windowed);
        return ret;
    }

    switch (t->precision) {
        case FAF_PREC_FP32:
            return faf_execute_f32(t, out, in);
        case FAF_PREC_FP64:
            return faf_execute_f64(t, out, in);
        default:
            set_error("Unsupported precision for VM execution");
            return -1;
    }
}
