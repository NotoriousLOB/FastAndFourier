#ifndef FAF_BENCH_UTIL_H
#define FAF_BENCH_UTIL_H

#include "fastandfourier.h"

static inline faf_transform *bench_fft(size_t n, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.precision = prec;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_fft(&c);
}

static inline faf_transform *bench_fft_split(size_t n, faf_precision prec) {
    faf_config c = faf_config_init(n);
    c.precision = prec;
    c.layout = FAF_LAYOUT_SPLIT;
    return faf_create_fft(&c);
}

static inline faf_transform *bench_dct(size_t n, int type) {
    faf_config c = faf_config_init(n);
    c.dct_type = type;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dct(&c);
}

static inline faf_transform *bench_dst(size_t n, int type) {
    faf_config c = faf_config_init(n);
    c.dct_type = type;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dst(&c);
}

static inline faf_transform *bench_haar(size_t n, size_t levels) {
    faf_config c = faf_config_init(n);
    c.levels = levels;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_haar(&c);
}

static inline faf_transform *bench_d4(size_t n, size_t levels) {
    faf_config c = faf_config_init(n);
    c.levels = levels;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_daubechies4(&c);
}

static inline faf_transform *bench_mdct(size_t n) {
    faf_config c = faf_config_init(n);
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_mdct(&c);
}

static inline faf_transform *bench_dwt(faf_wavelet_family fam, size_t n,
                                       size_t levels, bool inverse) {
    faf_config c = faf_config_init(n);
    c.family = fam;
    c.levels = levels;
    c.dir = inverse ? FAF_DIR_INVERSE : FAF_DIR_FORWARD;
    c.layout = FAF_LAYOUT_INTERLEAVED;
    return faf_create_dwt(&c);
}

#endif
