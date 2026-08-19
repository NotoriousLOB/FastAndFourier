/**
 * @file faf_test_util.h
 * @brief Shared helpers for tests during the faf_config API.
 *
 * FFT/DWT tests still feed interleaved float[2n] buffers into faf_execute_f32,
 * so helpers pin layout to INTERLEAVED. Default create layout is SPLIT/REAL.
 */
#ifndef FAF_TEST_UTIL_H
#define FAF_TEST_UTIL_H

#include "fastandfourier.h"

static inline faf_transform *test_fft(size_t n, bool inverse, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.precision = prec;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_fft(&c);
}

static inline faf_transform *test_fft_n(size_t n) {
    return test_fft(n, false, FAF_PREC_FP32);
}

static inline faf_transform *test_dct(size_t n, int type) {
    faf_config c = faf_config_init(n);
    c.dct_type = type;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dct(&c);
}

static inline faf_transform *test_dst(size_t n, int type) {
    faf_config c = faf_config_init(n);
    c.dct_type = type;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dst(&c);
}

static inline faf_transform *test_mdct(size_t n) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_mdct(&c);
}

static inline faf_transform *test_dwt(faf_wavelet_family family, size_t n,
                                      size_t levels, bool inverse,
                                      faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.family = family;
    c.levels = levels;
    c.precision = prec;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dwt(&c);
}

static inline faf_transform *test_haar(size_t n, size_t levels) {
    return test_dwt(FAF_WAVELET_HAAR, n, levels, false, FAF_PREC_FP32);
}

static inline faf_transform *test_d4(size_t n, size_t levels) {
    return test_dwt(FAF_WAVELET_D4, n, levels, false, FAF_PREC_FP32);
}

#endif
