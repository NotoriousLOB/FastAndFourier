/**
 * @file faf_cwt_wavelets.c
 * @brief CWT wavelet prototype functions (Fourier closed form, double precision)
 */

#include "faf_cwt.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double faf_cwt_proto_morlet(double omega, double mu) {
    if (omega <= 0.0) return 0.0;
    double main_term = exp(-0.5 * (omega - mu) * (omega - mu));
    double correction = exp(-0.5 * mu * mu) * exp(-0.5 * omega * omega);
    return main_term - correction;
}

double faf_cwt_proto_morse(double omega, double gamma, double beta) {
    if (omega <= 0.0) return 0.0;
    double omega_p = pow(beta / gamma, 1.0 / gamma);
    double log_peak = beta * log(omega_p) - pow(omega_p, gamma);
    double log_psi = beta * log(omega) - pow(omega, gamma);
    return exp(log_psi - log_peak);
}

double faf_cwt_proto_bump(double omega, double center, double width) {
    if (omega <= 0.0) return 0.0;
    double mu = (omega - center) / width;
    if (fabs(mu) >= 1.0) return 0.0;
    return exp(1.0 - 1.0 / (1.0 - mu * mu));
}

double faf_cwt_proto_shannon(double omega, double lo, double hi) {
    if (omega <= 0.0) return 0.0;
    if (omega >= lo && omega <= hi) return 1.0;
    return 0.0;
}

/* Meyer auxiliary ν(x) = x^4 (35 − 84x + 70x² − 20x³) on (0, 1). */
static double cwt_meyer_aux(double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x2 * x2;
    return x4 * (35.0 - 84.0 * x + 70.0 * x2 - 20.0 * x3);
}

double faf_cwt_proto_meyer(double omega) {
    if (omega <= 0.0) return 0.0;
    const double two_pi_3  = 2.0 * M_PI / 3.0;
    const double four_pi_3 = 4.0 * M_PI / 3.0;
    const double eight_pi_3 = 8.0 * M_PI / 3.0;
    if (omega < two_pi_3) return 0.0;
    if (omega <= four_pi_3)
        return sin(0.5 * M_PI * cwt_meyer_aux(3.0 * omega / (2.0 * M_PI) - 1.0));
    if (omega <= eight_pi_3)
        return cos(0.5 * M_PI * cwt_meyer_aux(3.0 * omega / (4.0 * M_PI) - 1.0));
    return 0.0;
}

double faf_cwt_proto_peak_omega(faf_cwt_wavelet kind, const faf_cwt_config *cfg) {
    switch (kind) {
        case FAF_CWT_WAVELET_MORLET:
            return cfg->morlet_mu;
        case FAF_CWT_WAVELET_MORSE:
            return pow(cfg->morse_beta / cfg->morse_gamma,
                       1.0 / cfg->morse_gamma);
        case FAF_CWT_WAVELET_BUMP:
            return cfg->bump_center;
        case FAF_CWT_WAVELET_SHANNON:
            return 0.75 * M_PI;
        case FAF_CWT_WAVELET_MEYER:
            return 4.0 * M_PI / 3.0; /* junction of the two Meyer pieces */
        default:
            return 1.0;
    }
}
