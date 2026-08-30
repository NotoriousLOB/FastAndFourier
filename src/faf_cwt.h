/**
 * @file faf_cwt.h
 * @brief Internal CWT filter bank definitions (not installed)
 */

#ifndef FAF_CWT_H
#define FAF_CWT_H

#include "faf.h"

#define FAF_CWT_MAX_SCALES 512

typedef struct faf_cwt_bank {
    faf_cwt_config      cfg;
    size_t              n;
    size_t              n_bins;
    size_t              n_bins_padded;
    size_t              n_scales;
    size_t              n_rows;
    int                 has_lowpass;

    double             *scales;
    double             *freqs_hz;

    void               *Psi;          /* float or double, n_rows * n_bins_padded */
    double             *Psi_l2;       /* n_rows * n_bins_padded, always double */
    double             *A;            /* n_bins, LP sum */
    double             *Psi_dual;     /* n_rows * n_bins_padded */
    float              *Psi_dual_f32; /* precomputed f32 copy (NULL if f64) */

    faf_cwt_lp_report   report;

    faf_transform      *rfft;
    faf_transform      *irfft;

    void               *X_re, *X_im;  /* Hermitian scratch, n_bins */
    void               *Y_re, *Y_im;
    size_t              scratch_bytes;

    double             *xi;           /* freq bins for SSQ hooks, n_bins */

    faf_cwt_inverse_kind inv_kind;
    int                 bank_owned;
} faf_cwt_bank;

/* Internal execute functions called from faf_execute dispatch */
int faf_cwt_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in);
int faf_icwt_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in);

/* Bank lifetime */
faf_cwt_bank *faf_cwt_bank_create(const faf_cwt_config *cfg);
void faf_cwt_bank_destroy(faf_cwt_bank *bank);

/* Kernel dispatch */
void faf_cwt_mul_hermitian_f32(float *FAF_RESTRICT y_re, float *FAF_RESTRICT y_im,
                               const float *FAF_RESTRICT x_re, const float *FAF_RESTRICT x_im,
                               const float *FAF_RESTRICT psi, size_t n_bins);
void faf_cwt_mul_hermitian_f64(double *FAF_RESTRICT y_re, double *FAF_RESTRICT y_im,
                               const double *FAF_RESTRICT x_re, const double *FAF_RESTRICT x_im,
                               const double *FAF_RESTRICT psi, size_t n_bins);
void faf_cwt_mulacc_hermitian_f32(float *FAF_RESTRICT acc_re, float *FAF_RESTRICT acc_im,
                                  const float *FAF_RESTRICT y_re, const float *FAF_RESTRICT y_im,
                                  const float *FAF_RESTRICT psi, size_t n_bins);
void faf_cwt_mulacc_hermitian_f64(double *FAF_RESTRICT acc_re, double *FAF_RESTRICT acc_im,
                                  const double *FAF_RESTRICT y_re, const double *FAF_RESTRICT y_im,
                                  const double *FAF_RESTRICT psi, size_t n_bins);

/* Wavelet prototypes (create-time, double precision) */
double faf_cwt_proto_morlet(double omega, double mu);
double faf_cwt_proto_morse(double omega, double gamma, double beta);
double faf_cwt_proto_bump(double omega, double center, double width);
double faf_cwt_proto_shannon(double omega, double lo, double hi);
double faf_cwt_proto_meyer(double omega);
double faf_cwt_proto_peak_omega(faf_cwt_wavelet kind, const faf_cwt_config *cfg);

#endif /* FAF_CWT_H */
