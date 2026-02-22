/**
 * @file dspir_core.c
 * @brief Core library implementation
 */

#include "dspir.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Thread-local error buffer */
_Thread_local static char g_error_buf[256] = {0};

/* Architecture info */
static const char* g_arch_name = NULL;

const char* dspir_version(void) {
    return FASTANDFOURIER_VERSION_STRING;
}

const char* dspir_arch_name(void) {
    if (g_arch_name == NULL) {
        #if defined(DSPIR_ARCH_X86_64)
            #if defined(DSPIR_HAVE_AVX512)
                g_arch_name = "x86_64-avx512";
            #elif defined(DSPIR_HAVE_AVX2)
                g_arch_name = "x86_64-avx2";
            #elif defined(DSPIR_HAVE_SSE42)
                g_arch_name = "x86_64-sse4.2";
            #else
                g_arch_name = "x86_64-generic";
            #endif
        #elif defined(DSPIR_ARCH_AARCH64)
            #if defined(DSPIR_HAVE_SVE)
                g_arch_name = "aarch64-sve";
            #elif defined(DSPIR_HAVE_NEON)
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

int dspir_init(void) {
    /* Initialize any global state */
    g_arch_name = NULL;
    dspir_clear_error();
    return 0;
}

void dspir_cleanup(void) {
    /* Cleanup global state */
}

const char* dspir_get_error(void) {
    return g_error_buf;
}

void dspir_clear_error(void) {
    g_error_buf[0] = '\0';
}

static void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buf, sizeof(g_error_buf), fmt, args);
    va_end(args);
}

size_t dspir_get_alignment(void) {
    return DSPIR_ALIGN;
}

size_t dspir_precision_size(dspir_precision prec) {
    switch (prec) {
        case DSPIR_PREC_FP8:  return 1;
        case DSPIR_PREC_FP16:
        case DSPIR_PREC_BF16: return 2;
        case DSPIR_PREC_FP32: return 4;
        case DSPIR_PREC_FP64: return 8;
        case DSPIR_PREC_INT8: return 1;
        case DSPIR_PREC_INT16: return 2;
        case DSPIR_PREC_INT32: return 4;
        default: return 4;
    }
}

const char* dspir_precision_name(dspir_precision prec) {
    switch (prec) {
        case DSPIR_PREC_FP8:  return "fp8";
        case DSPIR_PREC_FP16: return "fp16";
        case DSPIR_PREC_BF16: return "bf16";
        case DSPIR_PREC_FP32: return "fp32";
        case DSPIR_PREC_FP64: return "fp64";
        case DSPIR_PREC_INT8: return "int8";
        case DSPIR_PREC_INT16: return "int16";
        case DSPIR_PREC_INT32: return "int32";
        default: return "unknown";
    }
}

const char* dspir_transform_name(dspir_transform_type type) {
    switch (type) {
        case DSPIR_TRANSFORM_FFT:          return "fft";
        case DSPIR_TRANSFORM_IFFT:         return "ifft";
        case DSPIR_TRANSFORM_RFFT:         return "rfft";
        case DSPIR_TRANSFORM_IRFFT:        return "irfft";
        case DSPIR_TRANSFORM_DCT_I:        return "dct_i";
        case DSPIR_TRANSFORM_DCT_II:       return "dct_ii";
        case DSPIR_TRANSFORM_DCT_III:      return "dct_iii";
        case DSPIR_TRANSFORM_DCT_IV:       return "dct_iv";
        case DSPIR_TRANSFORM_DST_I:        return "dst_i";
        case DSPIR_TRANSFORM_DST_II:       return "dst_ii";
        case DSPIR_TRANSFORM_DST_III:      return "dst_iii";
        case DSPIR_TRANSFORM_DST_IV:       return "dst_iv";
        case DSPIR_TRANSFORM_STFT:         return "stft";
        case DSPIR_TRANSFORM_MDCT:         return "mdct";
        case DSPIR_TRANSFORM_IMDCT:        return "imdct";
        case DSPIR_TRANSFORM_HAAR:         return "haar";
        case DSPIR_TRANSFORM_DAUBECHIES4:  return "daubechies4";
        case DSPIR_TRANSFORM_CDF97:        return "cdf97";
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

int dspir_is_power_of_2(size_t n) {
    return n && ((n & (n - 1)) == 0);
}

void dspir_bit_reverse_permute_f32(float *data, size_t n) {
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

void dspir_bit_reverse_permute_f64(double *data, size_t n) {
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

bool dspir_is_size_supported(dspir_transform_type type, size_t n) {
    if (n == 0) return false;
    
    switch (type) {
        case DSPIR_TRANSFORM_FFT:
        case DSPIR_TRANSFORM_IFFT:
        case DSPIR_TRANSFORM_HAAR:
        case DSPIR_TRANSFORM_DAUBECHIES4:
            return dspir_is_power_of_2(n);
        case DSPIR_TRANSFORM_RFFT:
        case DSPIR_TRANSFORM_IRFFT:
            return dspir_is_power_of_2(n) && n >= 2;
        case DSPIR_TRANSFORM_DCT_I:
            return n >= 2;
        case DSPIR_TRANSFORM_DCT_II:
        case DSPIR_TRANSFORM_DCT_III:
        case DSPIR_TRANSFORM_DCT_IV:
        case DSPIR_TRANSFORM_DST_I:
        case DSPIR_TRANSFORM_DST_II:
        case DSPIR_TRANSFORM_DST_III:
        case DSPIR_TRANSFORM_DST_IV:
            return n > 0;
        case DSPIR_TRANSFORM_MDCT:
        case DSPIR_TRANSFORM_IMDCT:
            return (n % 2) == 0;
        default:
            return true;
    }
}

size_t dspir_get_recommended_size(dspir_transform_type type, size_t min_size) {
    size_t n = dsir_next_power_of_2(min_size);
    
    /* For FFT, prefer sizes with small prime factors for Bluestein */
    if (type == DSPIR_TRANSFORM_FFT || type == DSPIR_TRANSFORM_IFFT) {
        /* Already power of 2, good for Cooley-Tukey */
        return n;
    }
    
    return n;
}

/* Transform creation functions */
dspir_transform* dspir_create_fft(size_t n, bool inverse, dspir_precision precision, uint32_t flags) {
    if (!dspir_is_power_of_2(n)) {
        set_error("FFT size must be power of 2, got %zu", n);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    t->type = inverse ? DSPIR_TRANSFORM_IFFT : DSPIR_TRANSFORM_FFT;
    t->flags = flags | (inverse ? DSPIR_FLAG_INVERSE : 0);
    
    /* Choose optimal radix based on size */
    if (n <= 16) {
        dspir_gen_fft_radix2(t, n, inverse);
    } else if ((n & (n - 1)) == 0 && n >= 64) {
        dspir_gen_fft_radix4(t, n, inverse);
    } else {
        dspir_gen_fft_mixed(t, n, inverse);
    }
    
    return t;
}

dspir_transform* dspir_create_dct(size_t n, int type, dspir_precision precision, uint32_t flags) {
    if (n == 0) {
        set_error("DCT size must be > 0");
        return NULL;
    }
    if (type < 1 || type > 4) {
        set_error("DCT type must be 1-4, got %d", type);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    switch (type) {
        case 1: t->type = DSPIR_TRANSFORM_DCT_I; break;
        case 2: t->type = DSPIR_TRANSFORM_DCT_II; break;
        case 3: t->type = DSPIR_TRANSFORM_DCT_III; break;
        case 4: t->type = DSPIR_TRANSFORM_DCT_IV; break;
    }
    t->flags = flags;
    
    /* Generate appropriate DCT bytecode */
    switch (type) {
        case 2:
            dspir_gen_dct_ii(t, n);
            break;
        case 4:
            dspir_gen_dct_iv(t, n);
            break;
        default:
            /* Fall back to general DCT via FFT for other types */
            dspir_gen_dct_ii(t, n);
            break;
    }
    
    return t;
}

dspir_transform* dspir_create_dst(size_t n, int type, dspir_precision precision, uint32_t flags) {
    if (n == 0) {
        set_error("DST size must be > 0");
        return NULL;
    }
    if (type < 1 || type > 4) {
        set_error("DST type must be 1-4, got %d", type);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    switch (type) {
        case 1: t->type = DSPIR_TRANSFORM_DST_I; break;
        case 2: t->type = DSPIR_TRANSFORM_DST_II; break;
        case 3: t->type = DSPIR_TRANSFORM_DST_III; break;
        case 4: t->type = DSPIR_TRANSFORM_DST_IV; break;
    }
    t->flags = flags;
    
    dspir_gen_dst_ii(t, n);
    
    return t;
}

dspir_transform* dspir_create_mdct(size_t n, dspir_precision precision, uint32_t flags) {
    if (n % 2 != 0) {
        set_error("MDCT size must be even, got %zu", n);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    t->type = DSPIR_TRANSFORM_MDCT;
    t->flags = flags;
    
    dspir_gen_mdct(t, n);
    
    return t;
}

dspir_transform* dspir_create_haar(size_t n, size_t levels, dspir_precision precision, uint32_t flags) {
    if (!dspir_is_power_of_2(n)) {
        set_error("Haar transform size must be power of 2, got %zu", n);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    t->type = DSPIR_TRANSFORM_HAAR;
    t->flags = flags;
    
    dspir_gen_haar(t, n, levels);
    
    return t;
}

dspir_transform* dspir_create_daubechies4(size_t n, size_t levels, dspir_precision precision, uint32_t flags) {
    if (!dspir_is_power_of_2(n)) {
        set_error("Daubechies-4 transform size must be power of 2, got %zu", n);
        return NULL;
    }
    
    dspir_transform *t = calloc(1, sizeof(dspir_transform));
    if (!t) {
        set_error("Failed to allocate transform");
        return NULL;
    }
    
    t->n = n;
    t->precision = precision;
    t->type = DSPIR_TRANSFORM_DAUBECHIES4;
    t->flags = flags;
    
    dspir_gen_daubechies4(t, n, levels);
    
    return t;
}

dspir_transform* dspir_create_stft(size_t n_fft, size_t hop_length, size_t win_length,
                                    dspir_precision precision, uint32_t flags) {
    if (win_length == 0) win_length = n_fft;
    if (hop_length == 0) hop_length = win_length / 4;
    if (win_length > n_fft) {
        set_error("STFT window length %zu exceeds FFT size %zu", win_length, n_fft);
        return NULL;
    }

    /* Create the underlying FFT transform */
    dspir_transform *t = dspir_create_fft(n_fft, false, precision, flags);
    if (!t) return NULL;

    t->type = DSPIR_TRANSFORM_STFT;
    t->hop_length = hop_length;
    t->win_length = win_length;

    /* Store Hann window in twiddles[1] */
    float *win = (float*)malloc(n_fft * sizeof(float));
    if (!win) {
        dspir_destroy_transform(t);
        return NULL;
    }
    dspir_gen_hann_window_f32(win, win_length);
    /* Zero-pad if win_length < n_fft */
    for (size_t i = win_length; i < n_fft; i++) {
        win[i] = 0.0f;
    }
    t->twiddles[1] = win;
    t->twiddle_sizes[1] = n_fft;

    return t;
}

void dspir_destroy_transform(dspir_transform *t) {
    if (!t) return;
    
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
    
    free(t);
}

/* Execution dispatch */
int dspir_execute(const dspir_transform *t, void *restrict out, const void *restrict in) {
    if (!t || !out || !in) return -1;

    /* Auto-select: try JIT for FP32 transforms with size >= 64 */
    if (t->precision == DSPIR_PREC_FP32 && t->n >= 64) {
        int ret = dspir_execute_jit(t, (void*)out, (const void*)in);
        if (ret == 0) return 0;
        /* JIT failed (no compiler, etc.) — fall back to VM */
    }

    return dspir_execute_vm(t, out, in);
}

int dspir_execute_vm(const dspir_transform *t, void *restrict out, const void *restrict in) {
    /* STFT: apply window to input before FFT */
    if (t->type == DSPIR_TRANSFORM_STFT && t->twiddles[1] && t->precision == DSPIR_PREC_FP32) {
        const float *win = (const float *)t->twiddles[1];
        const float *in_f = (const float *)in;
        float *windowed = (float *)aligned_alloc(64, t->n * 2 * sizeof(float));
        if (!windowed) return -1;
        for (size_t i = 0; i < t->n; i++) {
            windowed[i * 2]     = in_f[i * 2]     * win[i];
            windowed[i * 2 + 1] = in_f[i * 2 + 1] * win[i];
        }
        int ret = dspir_execute_f32(t, out, windowed);
        free(windowed);
        return ret;
    }

    switch (t->precision) {
        case DSPIR_PREC_FP32:
            return dspir_execute_f32(t, out, in);
        case DSPIR_PREC_FP64:
            return dspir_execute_f64(t, out, in);
        default:
            set_error("Unsupported precision for VM execution");
            return -1;
    }
}
