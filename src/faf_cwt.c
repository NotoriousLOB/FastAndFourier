/**
 * @file faf_cwt.c
 * @brief CWT filter bank: create, execute, inverse, LP report
 */

#include "faf_cwt.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif
#ifndef INFINITY
#define INFINITY HUGE_VAL
#endif

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */

faf_cwt_config faf_cwt_config_init(size_t n) {
    faf_cwt_config c;
    memset(&c, 0, sizeof(c));
    c.n           = n;
    c.fs          = 1.0;
    c.precision   = FAF_PREC_FP32;
    c.backend     = FAF_BACKEND_AUTO;
    c.wavelet     = FAF_CWT_WAVELET_MORSE;
    c.norm        = FAF_CWT_NORM_L1;
    c.scale_kind  = FAF_CWT_SCALE_GEOMETRIC;
    c.center_kind = FAF_CWT_CENTER_PEAK;
    c.voices      = 10;
    c.morlet_mu   = 6.0;
    c.morse_gamma = 3.0;
    /* MATLAB cwtfilterbank default: γ=3, TimeBandwidth P²=βγ=60 → β=20. */
    c.morse_beta  = 20.0;
    c.bump_center = 5.0;
    c.bump_width  = 1.0;
    c.bandpass_peak = 2.0;
    c.lp_alpha    = 0.05;
    c.lp_beta     = 0.05;
    c.lp_floor    = 1e-4;
    c.flags       = FAF_CWT_FLAG_INCLUDE_LOWPASS | FAF_CWT_FLAG_VALIDATE_STRICT;
    return c;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static double cwt_eval_proto(double omega, const faf_cwt_config *cfg) {
    switch (cfg->wavelet) {
        case FAF_CWT_WAVELET_MORLET:
            return faf_cwt_proto_morlet(omega, cfg->morlet_mu);
        case FAF_CWT_WAVELET_MORSE:
            return faf_cwt_proto_morse(omega, cfg->morse_gamma, cfg->morse_beta);
        case FAF_CWT_WAVELET_BUMP:
            return faf_cwt_proto_bump(omega, cfg->bump_center, cfg->bump_width);
        case FAF_CWT_WAVELET_SHANNON:
            return faf_cwt_proto_shannon(omega, M_PI / 2.0, M_PI);
        case FAF_CWT_WAVELET_MEYER:
            return faf_cwt_proto_meyer(omega);
        default:
            return 0.0;
    }
}

static double cwt_proto_energy_omega(const faf_cwt_config *cfg) {
    double dw = 1e-4;
    double wmax = 80.0;
    double num = 0.0, den = 0.0;
    for (double w = dw; w <= wmax; w += dw) {
        double p = cwt_eval_proto(w, cfg);
        double e = p * p;
        num += w * e;
        den += e;
    }
    if (den <= 0.0)
        return faf_cwt_proto_peak_omega(cfg->wavelet, cfg);
    return num / den;
}

static double cwt_admissibility(const faf_cwt_config *cfg, double xi0) {
    double dw = 1e-4;
    double omega_max = xi0 * 20.0;
    double sum = 0.0;
    for (double w = dw; w <= omega_max; w += dw) {
        double psi = cwt_eval_proto(w, cfg);
        sum += psi / w;
    }
    return sum * dw;
}

/* ------------------------------------------------------------------ */
/* Bank create                                                         */
/* ------------------------------------------------------------------ */

faf_cwt_bank *faf_cwt_bank_create(const faf_cwt_config *cfg) {
    if (!cfg) {
        faf_set_error("cwt: null argument");
        return NULL;
    }

    faf_cwt_config c = *cfg;

    /* Validate n */
    if (c.n < 2 || (c.n & 1) != 0) {
        faf_set_error("cwt: n=%zu not even 5-smooth", c.n);
        return NULL;
    }
    if (!faf_is_5_smooth(c.n)) {
        faf_set_error("cwt: n=%zu not even 5-smooth", c.n);
        return NULL;
    }

    /* Validate precision */
    if (c.precision != FAF_PREC_FP32 && c.precision != FAF_PREC_FP64) {
        faf_set_error("cwt: precision %d unsupported", (int)c.precision);
        return NULL;
    }

    /* Validate voices */
    if (c.voices == 0) {
        faf_set_error("cwt: voices must be >= 1");
        return NULL;
    }

    /* Validate fs */
    if (c.fs <= 0.0) c.fs = 1.0;

    /* Validate wavelet params */
    if (c.wavelet == FAF_CWT_WAVELET_MORSE) {
        if (c.morse_gamma <= 0.0 || c.morse_beta <= 0.0) {
            faf_set_error("cwt: morse gamma/beta must be > 0");
            return NULL;
        }
    }
    if (c.wavelet == FAF_CWT_WAVELET_MORLET && c.morlet_mu <= 0.0) {
        faf_set_error("cwt: morlet mu must be > 0");
        return NULL;
    }
    if (c.wavelet == FAF_CWT_WAVELET_BUMP && c.bump_width <= 0.0) {
        faf_set_error("cwt: bump width must be > 0");
        return NULL;
    }
    if (c.flags & FAF_CWT_FLAG_FIXED_LOWPASS) {
        faf_set_error("cwt: FAF_CWT_FLAG_FIXED_LOWPASS is reserved");
        return NULL;
    }

    double xi0 = (c.center_kind == FAF_CWT_CENTER_ENERGY)
        ? cwt_proto_energy_omega(&c)
        : faf_cwt_proto_peak_omega(c.wavelet, &c);
    if (!(xi0 > 0.0) || xi0 != xi0) {
        faf_set_error("cwt: prototype center frequency is not finite");
        return NULL;
    }
    size_t n = c.n;
    size_t n_bins = n / 2 + 1;
    size_t n_bins_padded = (n_bins + 15) & ~((size_t)15);

    /* Auto fmin/fmax. Cap the coarsest centre at 16 cycles in the record
     * so the kernel does not wrap the length-n torus; the residual
     * lowpass absorbs everything below that (paper §7.4). Stay off
     * Nyquist so the finest scale is not an aliased stub. */
    double nyq = c.fs / 2.0;
    if (c.fmax <= 0.0)
        c.fmax = nyq * 0.45;
    if (c.fmin <= 0.0)
        c.fmin = 16.0 * c.fs / (double)n;

    if (c.fmin >= c.fmax) {
        faf_set_error("cwt: fmin >= fmax");
        return NULL;
    }
    if (c.fmax >= nyq) {
        faf_set_error("cwt: fmax at or above Nyquist");
        return NULL;
    }

    /* Count scales */
    size_t J = 0;
    if (c.scale_kind == FAF_CWT_SCALE_GEOMETRIC) {
        double octaves = log(c.fmax / c.fmin) / M_LN2;
        if (!(octaves >= 0.0) || octaves != octaves) {
            faf_set_error("cwt: fmin/fmax do not form a valid octave span");
            return NULL;
        }
        J = (size_t)floor((double)c.voices * octaves + 1.0 + 1e-12);
        if (J < 1) J = 1;
    } else {
        J = c.voices < 2 ? 2 : (size_t)c.voices;
    }

    if (J == 0) {
        faf_set_error("cwt: zero scales computed");
        return NULL;
    }
    if (J > FAF_CWT_MAX_SCALES) {
        faf_set_error("cwt: too many scales (%zu > %d)", J, FAF_CWT_MAX_SCALES);
        return NULL;
    }

    int has_lp = (c.flags & FAF_CWT_FLAG_INCLUDE_LOWPASS) ? 1 : 0;
    size_t n_rows = J + (size_t)has_lp;

    /* Allocate bank */
    faf_cwt_bank *bank = calloc(1, sizeof(faf_cwt_bank));
    if (!bank) {
        faf_set_error("cwt: alloc failed");
        return NULL;
    }
    bank->cfg = c;
    bank->n = n;
    bank->n_bins = n_bins;
    bank->n_bins_padded = n_bins_padded;
    bank->n_scales = J;
    bank->has_lowpass = has_lp;
    bank->n_rows = n_rows;

    /* Allocate arrays */
    bank->scales   = faf_aligned_alloc(J * sizeof(double));
    bank->freqs_hz = faf_aligned_alloc(J * sizeof(double));
    bank->Psi_l2   = faf_aligned_alloc(n_rows * n_bins_padded * sizeof(double));
    bank->A        = faf_aligned_alloc(n_bins * sizeof(double));
    bank->Psi_dual = faf_aligned_alloc(n_rows * n_bins_padded * sizeof(double));
    bank->xi       = faf_aligned_alloc(n_bins * sizeof(double));

    size_t elem = (c.precision == FAF_PREC_FP64) ? sizeof(double) : sizeof(float);
    bank->Psi = faf_aligned_alloc(n_rows * n_bins_padded * elem);

    if (!bank->scales || !bank->freqs_hz || !bank->Psi_l2 || !bank->A ||
        !bank->Psi_dual || !bank->xi || !bank->Psi) {
        faf_set_error("cwt: alloc failed");
        faf_cwt_bank_destroy(bank);
        return NULL;
    }

    memset(bank->Psi_l2, 0, n_rows * n_bins_padded * sizeof(double));
    memset(bank->Psi_dual, 0, n_rows * n_bins_padded * sizeof(double));
    memset(bank->Psi, 0, n_rows * n_bins_padded * elem);

    /* Fill frequency bins */
    for (size_t k = 0; k < n_bins; k++)
        bank->xi[k] = 2.0 * M_PI * (double)k / (double)n;

    /* Fill scales and frequencies */
    if (c.scale_kind == FAF_CWT_SCALE_GEOMETRIC) {
        for (size_t j = 0; j < J; j++) {
            double f = c.fmin * pow(2.0, (double)j / (double)c.voices);
            bank->freqs_hz[j] = f;
            double omega_j = 2.0 * M_PI * f / c.fs;
            bank->scales[j] = xi0 / omega_j;
        }
    } else {
        double df = (c.fmax - c.fmin) / (double)(J > 1 ? J - 1 : 1);
        for (size_t j = 0; j < J; j++) {
            double f = c.fmin + (double)j * df;
            bank->freqs_hz[j] = f;
            double omega_j = 2.0 * M_PI * f / c.fs;
            bank->scales[j] = xi0 / omega_j;
        }
    }

    /* Evaluate prototype with L2 dilation: Psi_l2[j,k] = proto(s_j * xi_k).
     * The CQT geometric spacing ensures sum_j |proto(s_j * xi_k)|^2 ≈ const
     * across the tiled band, making A[k] ≈ 1.
     * Row order: row 0 = phi if lowpass, then wavelet scales coarse→fine
     * (largest s / lowest f first). */
    size_t wav_offset = has_lp ? 1 : 0;
    for (size_t j = 0; j < J; j++) {
        size_t row = wav_offset + j;
        double *psi_row = bank->Psi_l2 + row * n_bins_padded;
        double s = bank->scales[j];
        for (size_t k = 0; k < n_bins; k++) {
            double omega = s * bank->xi[k];
            psi_row[k] = cwt_eval_proto(omega, &c);
        }
        psi_row[0] = 0.0; /* zero mean */
        if (n_bins > 1) psi_row[n_bins - 1] = 0.0; /* analytic: Nyquist unused */
    }

    /* Normalize L2 shadow so A ≈ 1 in the tiled band.
     * Only geometric scales tile; linear scales skip rescaling. */
    if (c.scale_kind == FAF_CWT_SCALE_GEOMETRIC) {
        double *A_pre = faf_aligned_alloc(n_bins * sizeof(double));
        if (!A_pre) {
            faf_set_error("cwt: alloc failed");
            faf_cwt_bank_destroy(bank);
            return NULL;
        }
        double A_max = 0.0;
        for (size_t k = 0; k < n_bins; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < J; j++) {
                double v = bank->Psi_l2[(wav_offset + j) * n_bins_padded + k];
                sum += v * v;
            }
            A_pre[k] = sum;
            if (sum > A_max) A_max = sum;
        }

        /* Average A only over the plateau (bins within 5% of the peak A).
         * This excludes the roll-off edges where A tapers to zero. */
        double plateau_thresh = A_max * 0.95;
        double mean_a = 0.0;
        size_t cnt = 0;
        for (size_t k = 0; k < n_bins; k++) {
            if (A_pre[k] >= plateau_thresh) {
                mean_a += A_pre[k];
                cnt++;
            }
        }
        if (cnt > 0) mean_a /= (double)cnt;
        if (mean_a <= 0.0) mean_a = 1.0;

        double rescale = 1.0 / sqrt(mean_a);
        for (size_t j = 0; j < J; j++) {
            double *row = bank->Psi_l2 + (wav_offset + j) * n_bins_padded;
            for (size_t k = 0; k < n_bins; k++)
                row[k] *= rescale;
        }
        faf_aligned_free(A_pre);
    }

    /* Build residual lowpass from L2 shadow */
    if (has_lp && !(c.flags & FAF_CWT_FLAG_FIXED_LOWPASS)) {
        double *phi_row = bank->Psi_l2; /* row 0 */
        for (size_t k = 0; k < n_bins; k++) {
            double sum_sq = 0.0;
            for (size_t j = 0; j < J; j++) {
                double v = bank->Psi_l2[(wav_offset + j) * n_bins_padded + k];
                sum_sq += v * v;
            }
            double residual = 1.0 - sum_sq;
            phi_row[k] = (residual > 0.0) ? sqrt(residual) : 0.0;
        }
    }

    /* Compute LP sum A[k] from L2 shadow */
    for (size_t k = 0; k < n_bins; k++) {
        double sum = 0.0;
        for (size_t r = 0; r < n_rows; r++) {
            double v = bank->Psi_l2[r * n_bins_padded + k];
            sum += v * v;
        }
        bank->A[k] = sum;
    }

    /* Dual is built from the live analysis filters after they are filled. */

    /* Build live analysis table (Psi) with chosen norm */
    if (c.precision == FAF_PREC_FP64) {
        double *psi_live = (double *)bank->Psi;
        for (size_t r = 0; r < n_rows; r++) {
            const double *l2_row = bank->Psi_l2 + r * n_bins_padded;
            double *live_row = psi_live + r * n_bins_padded;

            if ((int)r < has_lp) {
                /* Lowpass: copy L2 */
                for (size_t k = 0; k < n_bins; k++)
                    live_row[k] = l2_row[k];
                continue;
            }

            size_t j = r - wav_offset;
            double s = bank->scales[j];
            if (c.norm == FAF_CWT_NORM_L1) {
                /* L1 dilation: ψ̂_s(ξ) = ψ̂(sξ), already peak-normalized. */
                for (size_t k = 0; k < n_bins; k++) {
                    double omega = s * bank->xi[k];
                    live_row[k] = cwt_eval_proto(omega, &c);
                }
                live_row[0] = 0.0;
            } else if (c.norm == FAF_CWT_NORM_L2) {
                for (size_t k = 0; k < n_bins; k++)
                    live_row[k] = l2_row[k];
            } else { /* bandpass */
                double peak = 0.0;
                for (size_t k = 0; k < n_bins; k++) {
                    double omega = s * bank->xi[k];
                    live_row[k] = cwt_eval_proto(omega, &c);
                    if (live_row[k] > peak) peak = live_row[k];
                }
                live_row[0] = 0.0;
                if (peak > 0.0) {
                    double sc = c.bandpass_peak / peak;
                    for (size_t k = 0; k < n_bins; k++)
                        live_row[k] *= sc;
                }
            }
            if (n_bins > 1) live_row[n_bins - 1] = 0.0;
        }
    } else {
        float *psi_live = (float *)bank->Psi;
        for (size_t r = 0; r < n_rows; r++) {
            const double *l2_row = bank->Psi_l2 + r * n_bins_padded;
            float *live_row = psi_live + r * n_bins_padded;

            if ((int)r < has_lp) {
                for (size_t k = 0; k < n_bins; k++)
                    live_row[k] = (float)l2_row[k];
                continue;
            }

            size_t j = r - wav_offset;
            double s = bank->scales[j];
            if (c.norm == FAF_CWT_NORM_L1) {
                for (size_t k = 0; k < n_bins; k++) {
                    double omega = s * bank->xi[k];
                    live_row[k] = (float)cwt_eval_proto(omega, &c);
                }
                live_row[0] = 0.0f;
            } else if (c.norm == FAF_CWT_NORM_L2) {
                for (size_t k = 0; k < n_bins; k++)
                    live_row[k] = (float)l2_row[k];
            } else {
                float peak = 0.0f;
                for (size_t k = 0; k < n_bins; k++) {
                    double omega = s * bank->xi[k];
                    live_row[k] = (float)cwt_eval_proto(omega, &c);
                    if (live_row[k] > peak) peak = live_row[k];
                }
                live_row[0] = 0.0f;
                if (peak > 0.0f) {
                    float sc = (float)(c.bandpass_peak / (double)peak);
                    for (size_t k = 0; k < n_bins; k++)
                        live_row[k] *= sc;
                }
            }
            if (n_bins > 1) live_row[n_bins - 1] = 0.0f;
        }
    }

    /* Dual-frame filters from the live analysis table, so inversion
     * matches whatever normalization produced the coefficients. */
    {
        double eps_dual = (c.precision == FAF_PREC_FP64) ? 1e-12 : 1e-6;
        for (size_t k = 0; k < n_bins; k++) {
            double sum = 0.0;
            if (c.precision == FAF_PREC_FP64) {
                const double *psi = (const double *)bank->Psi;
                for (size_t r = 0; r < n_rows; r++) {
                    double v = psi[r * n_bins_padded + k];
                    sum += v * v;
                }
                for (size_t r = 0; r < n_rows; r++) {
                    double v = psi[r * n_bins_padded + k];
                    bank->Psi_dual[r * n_bins_padded + k] =
                        (sum > 0.0) ? v / (sum + eps_dual) : 0.0;
                }
            } else {
                const float *psi = (const float *)bank->Psi;
                for (size_t r = 0; r < n_rows; r++) {
                    double v = (double)psi[r * n_bins_padded + k];
                    sum += v * v;
                }
                for (size_t r = 0; r < n_rows; r++) {
                    double v = (double)psi[r * n_bins_padded + k];
                    bank->Psi_dual[r * n_bins_padded + k] =
                        (sum > 0.0) ? v / (sum + eps_dual) : 0.0;
                }
            }
        }
        if (c.precision == FAF_PREC_FP32) {
            bank->Psi_dual_f32 = faf_aligned_alloc(n_rows * n_bins_padded * sizeof(float));
            if (!bank->Psi_dual_f32) {
                faf_set_error("cwt: alloc failed");
                faf_cwt_bank_destroy(bank);
                return NULL;
            }
            for (size_t i = 0; i < n_rows * n_bins_padded; i++)
                bank->Psi_dual_f32[i] = (float)bank->Psi_dual[i];
        }
    }

    /* Fill LP report */
    faf_cwt_lp_report *rep = &bank->report;
    rep->n = n;
    rep->n_bins = n_bins;
    rep->n_scales = J;
    rep->has_lowpass = has_lp;
    rep->alpha = c.lp_alpha;
    rep->beta = c.lp_beta;
    rep->A = bank->A;

    /* Certified band: skip DC and Nyquist edges */
    size_t k_lo = 1;
    if (!has_lp) {
        /* Find where lowest wavelet rises above floor */
        double *lowest = bank->Psi_l2 + wav_offset * n_bins_padded;
        double peak_val = 0.0;
        for (size_t k = 0; k < n_bins; k++)
            if (lowest[k] > peak_val) peak_val = lowest[k];
        double floor_thresh = peak_val * c.lp_floor;
        for (k_lo = 1; k_lo < n_bins; k_lo++)
            if (lowest[k_lo] >= floor_thresh) break;
    }

    size_t k_hi = n_bins - 2;
    {
        double *highest = bank->Psi_l2 + (wav_offset + J - 1) * n_bins_padded;
        double peak_val = 0.0;
        for (size_t k = 0; k < n_bins; k++)
            if (highest[k] > peak_val) peak_val = highest[k];
        double floor_thresh = peak_val * c.lp_floor;
        for (k_hi = n_bins - 2; k_hi > k_lo; k_hi--)
            if (highest[k_hi] >= floor_thresh) break;
    }

    rep->k_lo = k_lo;
    rep->k_hi = k_hi;
    rep->hole_bins_dc = k_lo;
    rep->hole_bins_nyq = (n_bins - 1) - k_hi;

    /* LP deviation on certified band */
    double max_dev = 0.0, sum_dev = 0.0;
    double min_a = 1e30, max_a = -1e30;
    size_t cert_count = 0;
    for (size_t k = k_lo; k <= k_hi; k++) {
        double a = bank->A[k];
        double dev = fabs(a - 1.0);
        if (dev > max_dev) max_dev = dev;
        sum_dev += dev;
        if (a < min_a) min_a = a;
        if (a > max_a) max_a = a;
        cert_count++;
    }
    if (cert_count == 0) {
        min_a = 0.0;
        max_a = 0.0;
    }
    rep->max_abs_dev = max_dev;
    rep->mean_abs_dev = cert_count > 0 ? sum_dev / (double)cert_count : 0.0;
    rep->min_A = min_a;
    rep->max_A = max_a;
    rep->frame_cond = (min_a > 0.0) ? max_a / min_a : INFINITY;

    /* DC check */
    double max_dc = 0.0;
    for (size_t j = 0; j < J; j++) {
        double v = fabs(bank->Psi_l2[(wav_offset + j) * n_bins_padded]);
        if (v > max_dc) max_dc = v;
    }
    rep->max_dc_wavelet = max_dc;

    /* Wrap energy placeholder */
    rep->max_wrap_energy = 0.0;

    /* Admissibility */
    rep->admissibility_C = cwt_admissibility(&c, xi0);

    /* Pass/fail: max |A-1| against alpha, mean |A-1| against beta. */
    rep->passed = (max_dev <= c.lp_alpha &&
                   rep->mean_abs_dev <= c.lp_beta &&
                   cert_count > 0) ? 1 : 0;

    /* Strict validation */
    if ((c.flags & FAF_CWT_FLAG_VALIDATE_STRICT) && !rep->passed &&
        !(c.flags & FAF_CWT_FLAG_ALLOW_UNTILED)) {
        faf_set_error("cwt: LP bound violated max|A-1|=%g on [%zu,%zu]",
                      max_dev, k_lo, k_hi);
        faf_cwt_bank_destroy(bank);
        return NULL;
    }

    /* Create inner RFFT/IRFFT */
    faf_config fc = faf_config_init(n);
    fc.precision = c.precision;
    fc.layout = FAF_LAYOUT_HERMITIAN;
    fc.norm = FAF_NORM_NONE;
    fc.backend = c.backend;

    bank->rfft = faf_create_rfft(&fc);
    if (!bank->rfft) {
        faf_cwt_bank_destroy(bank);
        return NULL;
    }

    fc.dir = FAF_DIR_INVERSE;
    bank->irfft = faf_create_rfft(&fc);
    if (!bank->irfft) {
        faf_cwt_bank_destroy(bank);
        return NULL;
    }

    /* Allocate scratch */
    size_t scratch_elem = (c.precision == FAF_PREC_FP64) ? sizeof(double) : sizeof(float);
    bank->X_re  = faf_aligned_alloc(n_bins_padded * scratch_elem);
    bank->X_im  = faf_aligned_alloc(n_bins_padded * scratch_elem);
    bank->Y_re  = faf_aligned_alloc(n_bins_padded * scratch_elem);
    bank->Y_im  = faf_aligned_alloc(n_bins_padded * scratch_elem);

    if (!bank->X_re || !bank->X_im || !bank->Y_re || !bank->Y_im) {
        faf_set_error("cwt: scratch alloc failed");
        faf_cwt_bank_destroy(bank);
        return NULL;
    }

    /* Wrap / duration check: energy of the coarsest time-domain kernel
     * near t = n/2 (the circular far point). A wavelet whose support
     * exceeds n is that wavelet on a torus, not the prototype. */
    {
        void *wave = faf_aligned_alloc(n * scratch_elem);
        if (!wave) {
            faf_set_error("cwt: scratch alloc failed");
            faf_cwt_bank_destroy(bank);
            return NULL;
        }
        size_t coarse = wav_offset; /* first wavelet row, largest scale */
        memset(bank->Y_im, 0, n_bins_padded * scratch_elem);
        if (c.precision == FAF_PREC_FP64) {
            double *yre = (double *)bank->Y_re;
            double *yim = (double *)bank->Y_im;
            const double *psi = bank->Psi_l2 + coarse * n_bins_padded;
            memcpy(yre, psi, n_bins * sizeof(double));
            memset(yim, 0, n_bins * sizeof(double));
            faf_buffer herm = faf_buffer_hermitian(yre, yim, n_bins);
            faf_buffer time = faf_buffer_real(wave, n);
            if (faf_execute(bank->irfft, &time, &herm) != 0) {
                faf_aligned_free(wave);
                faf_cwt_bank_destroy(bank);
                return NULL;
            }
            const double *w = (const double *)wave;
            double total = 0.0, wrap = 0.0;
            size_t half = n / 2;
            size_t win = n / 32;
            if (win < 1) win = 1;
            for (size_t i = 0; i < n; i++) total += w[i] * w[i];
            for (size_t i = half - win; i < half + win && i < n; i++)
                wrap += w[i] * w[i];
            rep->max_wrap_energy = (total > 0.0) ? wrap / total : 0.0;
        } else {
            float *yre = (float *)bank->Y_re;
            float *yim = (float *)bank->Y_im;
            const double *psi = bank->Psi_l2 + coarse * n_bins_padded;
            for (size_t k = 0; k < n_bins; k++)
                yre[k] = (float)psi[k];
            memset(yim, 0, n_bins * sizeof(float));
            faf_buffer herm = faf_buffer_hermitian(yre, yim, n_bins);
            faf_buffer time = faf_buffer_real(wave, n);
            if (faf_execute(bank->irfft, &time, &herm) != 0) {
                faf_aligned_free(wave);
                faf_cwt_bank_destroy(bank);
                return NULL;
            }
            const float *w = (const float *)wave;
            double total = 0.0, wrap = 0.0;
            size_t half = n / 2;
            size_t win = n / 32;
            if (win < 1) win = 1;
            for (size_t i = 0; i < n; i++) total += (double)w[i] * (double)w[i];
            for (size_t i = half - win; i < half + win && i < n; i++)
                wrap += (double)w[i] * (double)w[i];
            rep->max_wrap_energy = (total > 0.0) ? wrap / total : 0.0;
        }
        faf_aligned_free(wave);

        if ((c.flags & FAF_CWT_FLAG_VALIDATE_STRICT) &&
            !(c.flags & FAF_CWT_FLAG_ALLOW_UNTILED) &&
            rep->max_wrap_energy > 1e-3) {
            faf_set_error("cwt: coarsest scale wraps (energy=%g)",
                          rep->max_wrap_energy);
            faf_cwt_bank_destroy(bank);
            return NULL;
        }
    }

    return bank;
}

void faf_cwt_bank_destroy(faf_cwt_bank *bank) {
    if (!bank) return;
    if (bank->scales)   faf_aligned_free(bank->scales);
    if (bank->freqs_hz) faf_aligned_free(bank->freqs_hz);
    if (bank->Psi)      faf_aligned_free(bank->Psi);
    if (bank->Psi_l2)   faf_aligned_free(bank->Psi_l2);
    if (bank->A)        faf_aligned_free(bank->A);
    if (bank->Psi_dual) faf_aligned_free(bank->Psi_dual);
    if (bank->Psi_dual_f32) faf_aligned_free(bank->Psi_dual_f32);
    if (bank->xi)       faf_aligned_free(bank->xi);
    if (bank->X_re)     faf_aligned_free(bank->X_re);
    if (bank->X_im)     faf_aligned_free(bank->X_im);
    if (bank->Y_re)     faf_aligned_free(bank->Y_re);
    if (bank->Y_im)     faf_aligned_free(bank->Y_im);
    if (bank->rfft)     faf_destroy_transform(bank->rfft);
    if (bank->irfft)    faf_destroy_transform(bank->irfft);
    free(bank);
}

/* ------------------------------------------------------------------ */
/* Transform wrappers                                                  */
/* ------------------------------------------------------------------ */

faf_transform *faf_create_cwt(const faf_cwt_config *cfg) {
    faf_cwt_bank *bank = faf_cwt_bank_create(cfg);
    if (!bank) return NULL;

    faf_transform *t = calloc(1, sizeof(faf_transform));
    if (!t) {
        faf_cwt_bank_destroy(bank);
        faf_set_error("cwt: alloc failed");
        return NULL;
    }

    t->type = FAF_TRANSFORM_CWT;
    t->n = bank->n;
    t->precision = bank->cfg.precision;
    t->cfg = faf_config_init(bank->n);
    t->cfg.precision = bank->cfg.precision;
    t->cfg.layout = FAF_LAYOUT_REAL;
    t->cfg.norm = FAF_NORM_NONE;
    t->cfg.backend = cfg->backend;
    t->user_aux = bank;
    t->user_aux_n = bank->n_rows;
    bank->bank_owned = 1;

    return t;
}

faf_transform *faf_create_icwt(const faf_cwt_config *cfg, faf_cwt_inverse_kind kind) {
    if (!cfg) {
        faf_set_error("cwt: null argument");
        return NULL;
    }
    if (kind == FAF_CWT_INV_L1 && cfg->norm != FAF_CWT_NORM_L1) {
        faf_set_error("cwt: L1 inverse requires L1 bank");
        return NULL;
    }

    faf_cwt_bank *bank = faf_cwt_bank_create(cfg);
    if (!bank) return NULL;

    faf_transform *t = calloc(1, sizeof(faf_transform));
    if (!t) {
        faf_cwt_bank_destroy(bank);
        faf_set_error("cwt: alloc failed");
        return NULL;
    }

    t->type = FAF_TRANSFORM_ICWT;
    t->n = bank->n;
    t->precision = bank->cfg.precision;
    t->cfg = faf_config_init(bank->n);
    t->cfg.precision = bank->cfg.precision;
    t->cfg.layout = FAF_LAYOUT_REAL;
    t->cfg.norm = FAF_NORM_NONE;
    t->cfg.dir = FAF_DIR_INVERSE;
    t->cfg.backend = cfg->backend;
    t->user_aux = bank;
    t->user_aux_n = bank->n_rows;
    bank->bank_owned = 1;
    bank->inv_kind = kind;

    return t;
}

/* ------------------------------------------------------------------ */
/* Forward execute                                                     */
/* ------------------------------------------------------------------ */

int faf_cwt_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in) {
    if (!t || !out || !in) {
        faf_set_error("cwt: null argument");
        return -1;
    }

    faf_cwt_bank *bank = (faf_cwt_bank *)t->user_aux;
    if (!bank) {
        faf_set_error("cwt: no bank");
        return -1;
    }

    if (!in->re || !out->re) {
        faf_set_error("cwt: buffers must provide re");
        return -1;
    }
    if (in->layout != FAF_LAYOUT_REAL || out->layout != FAF_LAYOUT_REAL) {
        faf_set_error("cwt: layout mismatch");
        return -1;
    }
    if (in->n != bank->n) {
        faf_set_error("cwt: buffer n mismatch");
        return -1;
    }
    if (in->re == out->re) {
        faf_set_error("cwt: in and out alias");
        return -1;
    }

    size_t n = bank->n;
    size_t n_bins = bank->n_bins;
    size_t n_bins_padded = bank->n_bins_padded;

    /* RFFT the input (once) */
    faf_buffer x_herm = faf_buffer_hermitian(bank->X_re, bank->X_im, n_bins);
    faf_buffer x_in = *in;
    int ret = faf_execute(bank->rfft, &x_herm, &x_in);
    if (ret != 0) return ret;

    if (t->precision == FAF_PREC_FP64) {
        double *x_re = (double *)bank->X_re;
        double *x_im = (double *)bank->X_im;
        double *y_re = (double *)bank->Y_re;
        double *y_im = (double *)bank->Y_im;
        double *out_re = (double *)out->re;

        for (size_t row = 0; row < bank->n_rows; row++) {
            const double *psi_row = (const double *)bank->Psi + row * n_bins_padded;
            faf_cwt_mul_hermitian_f64(y_re, y_im, x_re, x_im, psi_row, n_bins);

            faf_buffer row_herm = faf_buffer_hermitian(y_re, y_im, n_bins);
            faf_buffer row_out = faf_buffer_real(out_re + row * n, n);
            ret = faf_execute(bank->irfft, &row_out, &row_herm);
            if (ret != 0) return ret;
        }
    } else {
        float *x_re = (float *)bank->X_re;
        float *x_im = (float *)bank->X_im;
        float *y_re = (float *)bank->Y_re;
        float *y_im = (float *)bank->Y_im;
        float *out_re = (float *)out->re;

        for (size_t row = 0; row < bank->n_rows; row++) {
            const float *psi_row = (const float *)bank->Psi + row * n_bins_padded;
            faf_cwt_mul_hermitian_f32(y_re, y_im, x_re, x_im, psi_row, n_bins);

            faf_buffer row_herm = faf_buffer_hermitian(y_re, y_im, n_bins);
            faf_buffer row_out = faf_buffer_real(out_re + row * n, n);
            ret = faf_execute(bank->irfft, &row_out, &row_herm);
            if (ret != 0) return ret;
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Inverse execute                                                     */
/* ------------------------------------------------------------------ */

int faf_icwt_execute(const faf_transform *t, faf_buffer *out, const faf_buffer *in) {
    if (!t || !out || !in) {
        faf_set_error("cwt: null argument");
        return -1;
    }

    faf_cwt_bank *bank = (faf_cwt_bank *)t->user_aux;
    if (!bank) {
        faf_set_error("cwt: no bank");
        return -1;
    }

    if (!in->re || !out->re) {
        faf_set_error("cwt: buffers must provide re");
        return -1;
    }
    if (in->layout != FAF_LAYOUT_REAL || out->layout != FAF_LAYOUT_REAL) {
        faf_set_error("cwt: layout mismatch");
        return -1;
    }
    if (out->n != bank->n) {
        faf_set_error("cwt: buffer n mismatch");
        return -1;
    }

    size_t n = bank->n;
    size_t n_bins = bank->n_bins;
    size_t n_bins_padded = bank->n_bins_padded;

    if (bank->inv_kind == FAF_CWT_INV_L1) {
        /* L1 one-integral inverse */
        double dlog = M_LN2 / (double)bank->cfg.voices;
        double C = bank->report.admissibility_C;
        if (C <= 0.0) C = 1.0;
        size_t wav_offset = bank->has_lowpass ? 1 : 0;

        if (t->precision == FAF_PREC_FP64) {
            double *result = (double *)out->re;
            const double *w_re = (const double *)in->re;
            memset(result, 0, n * sizeof(double));
            for (size_t j = 0; j < bank->n_scales; j++) {
                size_t row = wav_offset + j;
                const double *row_ptr = w_re + row * n;
                for (size_t i = 0; i < n; i++)
                    result[i] += row_ptr[i] * dlog;
            }
            for (size_t i = 0; i < n; i++)
                result[i] /= C;
            if (bank->has_lowpass) {
                const double *lp = w_re;
                for (size_t i = 0; i < n; i++)
                    result[i] += lp[i];
            }
        } else {
            float *result = (float *)out->re;
            const float *w_re = (const float *)in->re;
            memset(result, 0, n * sizeof(float));
            float dlogf = (float)dlog;
            float Cf = (float)C;
            for (size_t j = 0; j < bank->n_scales; j++) {
                size_t row = wav_offset + j;
                const float *row_ptr = w_re + row * n;
                for (size_t i = 0; i < n; i++)
                    result[i] += row_ptr[i] * dlogf;
            }
            for (size_t i = 0; i < n; i++)
                result[i] /= Cf;
            if (bank->has_lowpass) {
                const float *lp = w_re;
                for (size_t i = 0; i < n; i++)
                    result[i] += lp[i];
            }
        }
        return 0;
    }

    /* Dual-frame inverse */
    if (t->precision == FAF_PREC_FP64) {
        double *acc_re = (double *)bank->X_re;
        double *acc_im = (double *)bank->X_im;
        double *y_re = (double *)bank->Y_re;
        double *y_im = (double *)bank->Y_im;
        const double *w_re = (const double *)in->re;

        memset(acc_re, 0, n_bins * sizeof(double));
        memset(acc_im, 0, n_bins * sizeof(double));

        for (size_t row = 0; row < bank->n_rows; row++) {
            /* Pack time row as real */
            faf_buffer row_in = faf_buffer_real((void *)(w_re + row * n), n);
            faf_buffer row_herm = faf_buffer_hermitian(y_re, y_im, n_bins);

            int ret = faf_execute(bank->rfft, &row_herm, &row_in);
            if (ret != 0) return ret;

            const double *dual_row = bank->Psi_dual + row * n_bins_padded;
            faf_cwt_mulacc_hermitian_f64(acc_re, acc_im, y_re, y_im, dual_row, n_bins);
        }

        acc_im[0] = 0.0;
        acc_im[n_bins - 1] = 0.0;

        faf_buffer acc_herm = faf_buffer_hermitian(acc_re, acc_im, n_bins);
        faf_buffer result = faf_buffer_real(out->re, n);
        return faf_execute(bank->irfft, &result, &acc_herm);
    } else {
        float *acc_re = (float *)bank->X_re;
        float *acc_im = (float *)bank->X_im;
        float *y_re = (float *)bank->Y_re;
        float *y_im = (float *)bank->Y_im;
        const float *w_re = (const float *)in->re;

        memset(acc_re, 0, n_bins * sizeof(float));
        memset(acc_im, 0, n_bins * sizeof(float));

        for (size_t row = 0; row < bank->n_rows; row++) {
            faf_buffer row_in = faf_buffer_real((void *)(w_re + row * n), n);
            faf_buffer row_herm = faf_buffer_hermitian(y_re, y_im, n_bins);

            int ret = faf_execute(bank->rfft, &row_herm, &row_in);
            if (ret != 0) return ret;

            const float *dual_row = bank->Psi_dual_f32 + row * n_bins_padded;
            faf_cwt_mulacc_hermitian_f32(acc_re, acc_im, y_re, y_im, dual_row, n_bins);
        }

        acc_im[0] = 0.0f;
        acc_im[n_bins - 1] = 0.0f;

        faf_buffer acc_herm = faf_buffer_hermitian(acc_re, acc_im, n_bins);
        faf_buffer result = faf_buffer_real(out->re, n);
        return faf_execute(bank->irfft, &result, &acc_herm);
    }
}

/* ------------------------------------------------------------------ */
/* Query functions                                                     */
/* ------------------------------------------------------------------ */

static const faf_cwt_bank *get_bank(const faf_transform *t) {
    if (!t) return NULL;
    if (t->type != FAF_TRANSFORM_CWT && t->type != FAF_TRANSFORM_ICWT)
        return NULL;
    return (const faf_cwt_bank *)t->user_aux;
}

const faf_cwt_lp_report *faf_cwt_bank_report(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    return b ? &b->report : NULL;
}

size_t faf_cwt_n_scales(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    return b ? b->n_scales : 0;
}

size_t faf_cwt_n_rows(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    return b ? b->n_rows : 0;
}

size_t faf_cwt_n_bins(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    return b ? b->n_bins : 0;
}

int faf_cwt_has_lowpass(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    return b ? b->has_lowpass : 0;
}

int faf_cwt_freqs(const faf_transform *t, double *hz, size_t cap) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || !hz) return -1;
    size_t n = b->n_scales < cap ? b->n_scales : cap;
    memcpy(hz, b->freqs_hz, n * sizeof(double));
    return (int)n;
}

int faf_cwt_scales(const faf_transform *t, double *s, size_t cap) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || !s) return -1;
    size_t n = b->n_scales < cap ? b->n_scales : cap;
    memcpy(s, b->scales, n * sizeof(double));
    return (int)n;
}

void faf_cwt_report_fprint(FILE *fp, const faf_cwt_lp_report *r) {
    if (!fp || !r) return;
    fprintf(fp, "CWT LP Report:\n");
    fprintf(fp, "  n=%zu  n_bins=%zu  n_scales=%zu  lowpass=%d\n",
            r->n, r->n_bins, r->n_scales, r->has_lowpass);
    fprintf(fp, "  passed=%d  max|A-1|=%.6g  mean|A-1|=%.6g\n",
            r->passed, r->max_abs_dev, r->mean_abs_dev);
    fprintf(fp, "  A range: [%.6g, %.6g]  cond=%.4g\n",
            r->min_A, r->max_A, r->frame_cond);
    fprintf(fp, "  certified band: [%zu, %zu]\n", r->k_lo, r->k_hi);
    fprintf(fp, "  holes: DC=%zu bins, Nyquist=%zu bins\n",
            r->hole_bins_dc, r->hole_bins_nyq);
    fprintf(fp, "  max DC wavelet=%.3e  wrap energy=%.3e\n",
            r->max_dc_wavelet, r->max_wrap_energy);
    fprintf(fp, "  admissibility C=%.6g\n", r->admissibility_C);
}

const float *faf_cwt_psi_f32(const faf_transform *t, size_t idx) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || b->cfg.precision != FAF_PREC_FP32) return NULL;
    size_t row = b->has_lowpass ? idx + 1 : idx;
    if (row >= b->n_rows) return NULL;
    return (const float *)b->Psi + row * b->n_bins_padded;
}

const double *faf_cwt_psi_f64(const faf_transform *t, size_t idx) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || b->cfg.precision != FAF_PREC_FP64) return NULL;
    size_t row = b->has_lowpass ? idx + 1 : idx;
    if (row >= b->n_rows) return NULL;
    return (const double *)b->Psi + row * b->n_bins_padded;
}

const float *faf_cwt_phi_f32(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || !b->has_lowpass || b->cfg.precision != FAF_PREC_FP32) return NULL;
    return (const float *)b->Psi;
}

const double *faf_cwt_phi_f64(const faf_transform *t) {
    const faf_cwt_bank *b = get_bank(t);
    if (!b || !b->has_lowpass || b->cfg.precision != FAF_PREC_FP64) return NULL;
    return (const double *)b->Psi;
}
