/**
 * @file faf_twiddles.c
 * @brief Twiddle factor generation for all transform types
 */

#include "faf.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Generate complex twiddle factors for FFT
 * 
 * Twiddles are stored as [re0, im0, re1, im1, ...]
 * For forward FFT: W_N^k = exp(-2*pi*i*k/N)
 * For inverse FFT: W_N^k = exp(+2*pi*i*k/N)
 */
void faf_gen_twiddles_f32(float *tw, size_t n, bool inverse) {
    const float sign = inverse ? 1.0f : -1.0f;
    const float twopi_over_n = 2.0f * (float)M_PI / (float)n;
    
    /* Generate n/2 twiddle factors (exploiting symmetry) */
    for (size_t k = 0; k < n / 2; k++) {
        float angle = sign * twopi_over_n * (float)k;
        tw[2*k]     = cosf(angle);  /* Real part */
        tw[2*k + 1] = sinf(angle);  /* Imaginary part */
    }
}

void faf_gen_twiddles_f64(double *tw, size_t n, bool inverse) {
    const double sign = inverse ? 1.0 : -1.0;
    const double twopi_over_n = 2.0 * M_PI / (double)n;
    
    for (size_t k = 0; k < n / 2; k++) {
        double angle = sign * twopi_over_n * (double)k;
        tw[2*k]     = cos(angle);
        tw[2*k + 1] = sin(angle);
    }
}

void faf_gen_twiddles_full_f32(float *tw, size_t n, bool inverse) {
    const float sign = inverse ? 1.0f : -1.0f;
    const float twopi_over_n = 2.0f * (float)M_PI / (float)n;
    for (size_t k = 0; k < n; k++) {
        float angle = sign * twopi_over_n * (float)k;
        tw[2 * k]     = cosf(angle);
        tw[2 * k + 1] = sinf(angle);
    }
}

void faf_gen_twiddles_full_f64(double *tw, size_t n, bool inverse) {
    const double sign = inverse ? 1.0 : -1.0;
    const double twopi_over_n = 2.0 * M_PI / (double)n;
    for (size_t k = 0; k < n; k++) {
        double angle = sign * twopi_over_n * (double)k;
        tw[2 * k]     = cos(angle);
        tw[2 * k + 1] = sin(angle);
    }
}

/**
 * @brief Generate twiddle factors for DCT-II
 * 
 * DCT-II can be computed via FFT with pre/post processing.
 * These twiddles are for the pre-processing step.
 */
void faf_gen_dct_twiddles_f32(float *tw, size_t n, int type) {
    switch (type) {
        case 2: {
            /* DCT-II: pre-twiddles for FFT-based computation */
            /* W = exp(-i*pi*k/(2*n)) */
            for (size_t k = 0; k < n; k++) {
                float angle = -(float)M_PI * (float)k / (2.0f * (float)n);
                tw[2*k]     = cosf(angle);
                tw[2*k + 1] = sinf(angle);
            }
            break;
        }
        case 4: {
            /* DCT-IV: twiddles for direct computation */
            /* W = exp(-i*pi*(k+0.5)*(j+0.5)/n) */
            for (size_t k = 0; k < n; k++) {
                float angle = -(float)M_PI * ((float)k + 0.5f) / (float)n;
                tw[2*k]     = cosf(angle);
                tw[2*k + 1] = sinf(angle);
            }
            break;
        }
        default:
            /* Generic twiddles */
            for (size_t k = 0; k < n; k++) {
                float angle = -(float)M_PI * (float)k / (float)n;
                tw[2*k]     = cosf(angle);
                tw[2*k + 1] = sinf(angle);
            }
            break;
    }
}

/**
 * @brief Generate twiddle factors for DST
 */
void faf_gen_dst_twiddles_f32(float *tw, size_t n, int type) {
    switch (type) {
        case 2: {
            /* DST-II twiddles */
            for (size_t k = 0; k < n; k++) {
                float angle = -(float)M_PI * ((float)k + 0.5f) / (float)n;
                tw[2*k]     = cosf(angle);
                tw[2*k + 1] = sinf(angle);
            }
            break;
        }
        default: {
            for (size_t k = 0; k < n; k++) {
                float angle = -(float)M_PI * (float)(k + 1) / (float)(n + 1);
                tw[2*k]     = cosf(angle);
                tw[2*k + 1] = sinf(angle);
            }
            break;
        }
    }
}

/**
 * @brief Generate MDCT twiddle factors
 * 
 * MDCT uses a combination of windowing and FFT-based computation
 */
void faf_gen_mdct_twiddles_f32(float *tw, size_t n) {
    /* MDCT rotation twiddles */
    size_t n2 = n / 2;
    size_t n4 = n / 4;
    
    /* Pre-rotation twiddles */
    for (size_t k = 0; k < n4; k++) {
        float angle1 = -(float)M_PI * ((float)k + 0.5f) / (float)n;
        float angle2 = -(float)M_PI * ((float)(n2 - 1 - k) + 0.5f) / (float)n;
        
        tw[4*k]     = cosf(angle1);
        tw[4*k + 1] = sinf(angle1);
        tw[4*k + 2] = cosf(angle2);
        tw[4*k + 3] = sinf(angle2);
    }
    
    /* Post-rotation twiddles (stored after pre-rotation at offset n = 4*n4) */
    for (size_t k = 0; k < n2; k++) {
        float angle = -(float)M_PI * ((float)k + 0.5f) / (float)n2;
        tw[n + 2*k]     = cosf(angle);
        tw[n + 2*k + 1] = sinf(angle);
    }
}

/**
 * @brief Generate window functions
 */
void faf_gen_hann_window_f32(float *win, size_t n) {
    for (size_t i = 0; i < n; i++) {
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1)));
    }
}

void faf_gen_hamming_window_f32(float *win, size_t n) {
    const float a0 = 0.54f;
    const float a1 = 0.46f;
    for (size_t i = 0; i < n; i++) {
        win[i] = a0 - a1 * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
}

void faf_gen_blackman_window_f32(float *win, size_t n) {
    const float a0 = 0.42f;
    const float a1 = 0.5f;
    const float a2 = 0.08f;
    for (size_t i = 0; i < n; i++) {
        float x = 2.0f * (float)M_PI * (float)i / (float)(n - 1);
        win[i] = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x);
    }
}

/**
 * @brief Generate Kaiser window
 */
void faf_gen_kaiser_window_f32(float *win, size_t n, float beta) {
    /* Modified Bessel function of the first kind, order 0 */
    float i0_beta = 0.0f;
    float x = 1.0f;
    float term = 1.0f;
    for (int k = 1; k < 20; k++) {
        x *= (beta / 2.0f) / (float)k;
        term += x * x;
    }
    i0_beta = term;
    
    for (size_t i = 0; i < n; i++) {
        float alpha = 2.0f * (float)i / (float)(n - 1) - 1.0f;
        float arg = beta * sqrtf(1.0f - alpha * alpha);
        
        /* Compute I0(arg) */
        float i0_arg = 0.0f;
        x = 1.0f;
        term = 1.0f;
        for (int k = 1; k < 20; k++) {
            x *= (arg / 2.0f) / (float)k;
            term += x * x;
        }
        i0_arg = term;
        
        win[i] = i0_arg / i0_beta;
    }
}

/**
 * @brief Generate wavelet filter coefficients
 */
void faf_gen_haar_coeffs_f32(float *lo, float *hi) {
    /* Haar wavelet: simplest wavelet */
    lo[0] = 0.7071067811865476f;  /* 1/sqrt(2) */
    lo[1] = 0.7071067811865476f;
    hi[0] = 0.7071067811865476f;
    hi[1] = -0.7071067811865476f;
}

void faf_gen_daubechies4_coeffs_f32(float *lo, float *hi) {
    /* Daubechies-4 wavelet coefficients (D4) */
    /* These are the famous db4 coefficients */
    const float h0 = 0.4829629131445341f;
    const float h1 = 0.8365163037378077f;
    const float h2 = 0.2241438680420134f;
    const float h3 = -0.1294095225512603f;
    
    lo[0] = h0;
    lo[1] = h1;
    lo[2] = h2;
    lo[3] = h3;
    
    hi[0] = h3;
    hi[1] = -h2;
    hi[2] = h1;
    hi[3] = -h0;
}

void faf_gen_cdf53_coeffs_f32(float *lo, float *hi) {
    /* LeGall 5/3 analysis filters (unnormalized lifting equivalent) */
    lo[0] = -0.125f;
    lo[1] =  0.25f;
    lo[2] =  0.75f;
    lo[3] =  0.25f;
    lo[4] = -0.125f;
    hi[0] = -0.5f;
    hi[1] =  1.0f;
    hi[2] = -0.5f;
    hi[3] =  0.0f;
    hi[4] =  0.0f;
}

void faf_gen_sym4_coeffs_f32(float *lo, float *hi) {
    lo[0] = -0.07576571478927333f;
    lo[1] = -0.02963552764599851f;
    lo[2] =  0.49761866763201545f;
    lo[3] =  0.8037387518059161f;
    lo[4] =  0.29785779560527736f;
    lo[5] = -0.09921954357684722f;
    lo[6] = -0.012603967262037833f;
    lo[7] =  0.0322231006040427f;
    for (int i = 0; i < 8; i++) {
        hi[i] = ((i & 1) ? -1.0f : 1.0f) * lo[7 - i];
    }
}

void faf_gen_cdf97_coeffs_f32(float *lo, float *hi, float *lo_r, float *hi_r) {
    /* CDF 9/7 analysis low-pass (9 taps) */
    lo[0] =  0.02674875741080976f;
    lo[1] = -0.01686411844287495f;
    lo[2] = -0.07822326652898785f;
    lo[3] =  0.2668641184428723f;
    lo[4] =  0.6029490182363579f;
    lo[5] =  0.2668641184428723f;
    lo[6] = -0.07822326652898785f;
    lo[7] = -0.01686411844287495f;
    lo[8] =  0.02674875741080976f;

    /* Analysis high-pass (7 taps, centered in 9-tap buffer) */
    hi[0] =  0.0f;
    hi[1] =  0.09127176311424948f;
    hi[2] = -0.05754352622849957f;
    hi[3] = -0.5912717631142470f;
    hi[4] =  1.115087052456994f;
    hi[5] = -0.5912717631142470f;
    hi[6] = -0.05754352622849957f;
    hi[7] =  0.09127176311424948f;
    hi[8] =  0.0f;

    /* Dual synthesis pair used by JPEG 2000 (not a time-reversal of analysis) */
    if (lo_r && hi_r) {
        lo_r[0] = -0.09127176311424948f;
        lo_r[1] = -0.05754352622849957f;
        lo_r[2] =  0.5912717631142470f;
        lo_r[3] =  1.115087052456994f;
        lo_r[4] =  0.5912717631142470f;
        lo_r[5] = -0.05754352622849957f;
        lo_r[6] = -0.09127176311424948f;
        lo_r[7] =  0.0f;
        lo_r[8] =  0.0f;

        hi_r[0] =  0.02674875741080976f;
        hi_r[1] =  0.01686411844287495f;
        hi_r[2] = -0.07822326652898785f;
        hi_r[3] = -0.2668641184428723f;
        hi_r[4] =  0.6029490182363579f;
        hi_r[5] = -0.2668641184428723f;
        hi_r[6] = -0.07822326652898785f;
        hi_r[7] =  0.01686411844287495f;
        hi_r[8] =  0.02674875741080976f;
    }
}

/**
 * @brief Generate STFT window and twiddles
 */
void faf_gen_stft_twiddles_f32(float *tw, size_t n_fft, size_t hop_length, size_t win_length) {
    (void)hop_length;
    
    /* FFT twiddles */
    faf_gen_twiddles_f32(tw, n_fft, false);
    
    /* Window function (Hann by default) */
    float *win = &tw[n_fft];
    if (win_length <= n_fft) {
        /* Zero-padded window */
        for (size_t i = 0; i < win_length; i++) {
            win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(win_length - 1)));
        }
        for (size_t i = win_length; i < n_fft; i++) {
            win[i] = 0.0f;
        }
    } else {
        /* Truncated window */
        faf_gen_hann_window_f32(win, n_fft);
    }
}

void faf_init_twiddles(void *tw, size_t n, bool inverse) {
    faf_gen_twiddles_f32((float*)tw, n, inverse);
}
