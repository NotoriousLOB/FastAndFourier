/**
 * @file chirp_builtins.h
 * @brief Standard builtin functions for the Chirp DSL
 * 
 * Provides a comprehensive library of mathematical functions, distributions,
 * and DSP primitives that can be called from Chirp programs.
 * 
 * @version 1.1.0
 */

#ifndef CHIRP_BUILTINS_H
#define CHIRP_BUILTINS_H

#include "fastandfourier.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * BUILTIN REGISTRATION
 * ============================================================================ */

/**
 * @brief Register all standard Chirp builtins
 * 
 * Call this at initialization to make all standard functions available
 * to Chirp programs.
 * 
 * @return Number of builtins registered, or -1 on error
 */
int chirp_register_standard_builtins(void);

/**
 * @brief Print a list of all registered builtins
 */
void chirp_list_builtins(void);

/* ============================================================================
 * TRIGONOMETRIC FUNCTIONS
 * ============================================================================ */

/* Single precision */
float chirp_sin_f32(float x);
float chirp_cos_f32(float x);
float chirp_tan_f32(float x);
float chirp_asin_f32(float x);
float chirp_acos_f32(float x);
float chirp_atan_f32(float x);
float chirp_atan2_f32(float y, float x);
float chirp_sinh_f32(float x);
float chirp_cosh_f32(float x);
float chirp_tanh_f32(float x);

/* Double precision */
double chirp_sin_f64(double x);
double chirp_cos_f64(double x);
double chirp_tan_f64(double x);
double chirp_asin_f64(double x);
double chirp_acos_f64(double x);
double chirp_atan_f64(double x);
double chirp_atan2_f64(double y, double x);
double chirp_sinh_f64(double x);
double chirp_cosh_f64(double x);
double chirp_tanh_f64(double x);

/* ============================================================================
 * EXPONENTIAL AND LOGARITHMIC
 * ============================================================================ */

float chirp_exp_f32(float x);
float chirp_exp2_f32(float x);
float chirp_log_f32(float x);
float chirp_log2_f32(float x);
float chirp_log10_f32(float x);
float chirp_pow_f32(float x, float y);
float chirp_sqrt_f32(float x);
float chirp_cbrt_f32(float x);
float chirp_hypot_f32(float x, float y);

double chirp_exp_f64(double x);
double chirp_exp2_f64(double x);
double chirp_log_f64(double x);
double chirp_log2_f64(double x);
double chirp_log10_f64(double x);
double chirp_pow_f64(double x, double y);
double chirp_sqrt_f64(double x);
double chirp_cbrt_f64(double x);
double chirp_hypot_f64(double x, double y);

/* ============================================================================
 * SPECIAL FUNCTIONS
 * ============================================================================ */

/* Error function and gamma */
float chirp_erf_f32(float x);
float chirp_erfc_f32(float x);
float chirp_gamma_f32(float x);
float chirp_lgamma_f32(float x);
float chirp_tgamma_f32(float x);

double chirp_erf_f64(double x);
double chirp_erfc_f64(double x);
double chirp_gamma_f64(double x);
double chirp_lgamma_f64(double x);
double chirp_tgamma_f64(double x);

/* Bessel functions */
float chirp_j0_f32(float x);
float chirp_j1_f32(float x);
float chirp_y0_f32(float x);
float chirp_y1_f32(float x);

double chirp_j0_f64(double x);
double chirp_j1_f64(double x);
double chirp_y0_f64(double x);
double chirp_y1_f64(double x);

/* ============================================================================
 * DISTRIBUTIONS (PROBABILITY DENSITY FUNCTIONS)
 * ============================================================================ */

/**
 * @brief Gaussian (normal) distribution PDF
 * @param x Input value
 * @param mu Mean
 * @param sigma Standard deviation
 */
float chirp_gaussian_pdf_f32(float x, float mu, float sigma);
double chirp_gaussian_pdf_f64(double x, double mu, double sigma);

/**
 * @brief Cauchy (Lorentzian) distribution PDF
 * @param x Input value
 * @param x0 Location parameter (median)
 * @param gamma Scale parameter (half-width at half-maximum)
 */
float chirp_cauchy_pdf_f32(float x, float x0, float gamma);
double chirp_cauchy_pdf_f64(double x, double x0, double gamma);

/**
 * @brief Exponential distribution PDF
 * @param x Input value (>= 0)
 * @param lambda Rate parameter
 */
float chirp_exponential_pdf_f32(float x, float lambda);
double chirp_exponential_pdf_f64(double x, double lambda);

/**
 * @brief Log-normal distribution PDF
 * @param x Input value (> 0)
 * @param mu Mean of underlying normal distribution
 * @param sigma Standard deviation of underlying normal
 */
float chirp_lognormal_pdf_f32(float x, float mu, float sigma);
double chirp_lognormal_pdf_f64(double x, double mu, double sigma);

/**
 * @brief Laplace (double exponential) distribution PDF
 * @param x Input value
 * @param mu Location parameter
 * @param b Scale parameter
 */
float chirp_laplace_pdf_f32(float x, float mu, float b);
double chirp_laplace_pdf_f64(double x, double mu, double b);

/**
 * @brief Student's t-distribution PDF
 * @param x Input value
 * @param nu Degrees of freedom
 */
float chirp_student_t_pdf_f32(float x, float nu);
double chirp_student_t_pdf_f64(double x, double nu);

/**
 * @brief Beta distribution PDF
 * @param x Input value (0 <= x <= 1)
 * @param alpha Shape parameter
 * @param beta Shape parameter
 */
float chirp_beta_pdf_f32(float x, float alpha, float beta);
double chirp_beta_pdf_f64(double x, double alpha, double beta);

/**
 * @brief Gamma distribution PDF
 * @param x Input value (>= 0)
 * @param k Shape parameter
 * @param theta Scale parameter
 */
float chirp_gamma_pdf_f32(float x, float k, float theta);
double chirp_gamma_pdf_f64(double x, double k, double theta);

/**
 * @brief Weibull distribution PDF
 * @param x Input value (>= 0)
 * @param k Shape parameter
 * @param lambda Scale parameter
 */
float chirp_weibull_pdf_f32(float x, float k, float lambda);
double chirp_weibull_pdf_f64(double x, double k, double lambda);

/**
 * @brief Uniform distribution PDF
 * @param x Input value
 * @param a Lower bound
 * @param b Upper bound
 */
float chirp_uniform_pdf_f32(float x, float a, float b);
double chirp_uniform_pdf_f64(double x, double a, double b);

/**
 * @brief Sigmoid (logistic) function
 * @param x Input value
 */
float chirp_sigmoid_f32(float x);
double chirp_sigmoid_f64(double x);

/**
 * @brief Softmax function (vector)
 * @param in Input array
 * @param out Output array (probabilities)
 * @param n Length
 */
void chirp_softmax_f32(const float *in, float *out, size_t n);
void chirp_softmax_f64(const double *in, double *out, size_t n);

/**
 * @brief Rectified Linear Unit (ReLU)
 * @param x Input value
 */
float chirp_relu_f32(float x);
double chirp_relu_f64(double x);

/**
 * @brief Gaussian Error Linear Unit (GELU)
 * @param x Input value
 */
float chirp_gelu_f32(float x);
double chirp_gelu_f64(double x);

/**
 * @brief Swish activation (x * sigmoid(x))
 * @param x Input value
 */
float chirp_swish_f32(float x);
double chirp_swish_f64(double x);

/* ============================================================================
 * VECTOR/MATRIX OPERATIONS
 * ============================================================================ */

/**
 * @brief Sum of array elements
 * @param vec Input array
 * @param n Length
 * @return Sum
 */
float chirp_sum_f32(const float *vec, size_t n);
double chirp_sum_f64(const double *vec, size_t n);

/**
 * @brief Dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @param n Length
 * @return Dot product
 */
float chirp_dot_f32(const float *a, const float *b, size_t n);
double chirp_dot_f64(const double *a, const double *b, size_t n);

/**
 * @brief Vector norm (L2)
 * @param vec Input array
 * @param n Length
 * @return sqrt(sum(vec[i]^2))
 */
float chirp_norm_f32(const float *vec, size_t n);
double chirp_norm_f64(const double *vec, size_t n);

/**
 * @brief L1 norm (sum of absolute values)
 * @param vec Input array
 * @param n Length
 * @return sum(|vec[i]|)
 */
float chirp_norm1_f32(const float *vec, size_t n);
double chirp_norm1_f64(const double *vec, size_t n);

/**
 * @brief Infinity norm (max absolute value)
 * @param vec Input array
 * @param n Length
 * @return max(|vec[i]|)
 */
float chirp_norm_inf_f32(const float *vec, size_t n);
double chirp_norm_inf_f64(const double *vec, size_t n);

/**
 * @brief Mean of array elements
 * @param vec Input array
 * @param n Length
 * @return Mean
 */
float chirp_mean_f32(const float *vec, size_t n);
double chirp_mean_f64(const double *vec, size_t n);

/**
 * @brief Variance of array elements
 * @param vec Input array
 * @param n Length
 * @return Variance
 */
float chirp_variance_f32(const float *vec, size_t n);
double chirp_variance_f64(const double *vec, size_t n);

/**
 * @brief Standard deviation
 * @param vec Input array
 * @param n Length
 * @return Standard deviation
 */
float chirp_std_f32(const float *vec, size_t n);
double chirp_std_f64(const double *vec, size_t n);

/**
 * @brief Cumulative sum (scan)
 * @param in Input array
 * @param out Output array (same size)
 * @param n Length
 */
void chirp_cumsum_f32(const float *in, float *out, size_t n);
void chirp_cumsum_f64(const double *in, double *out, size_t n);

/**
 * @brief Element-wise multiply (Hadamard product)
 * @param a First vector
 * @param b Second vector
 * @param out Result (can be a or b)
 * @param n Length
 */
void chirp_mul_f32(const float *a, const float *b, float *out, size_t n);
void chirp_mul_f64(const double *a, const double *b, double *out, size_t n);

/**
 * @brief Element-wise add
 */
void chirp_add_f32(const float *a, const float *b, float *out, size_t n);
void chirp_add_f64(const double *a, const double *b, double *out, size_t n);

/**
 * @brief Element-wise subtract
 */
void chirp_sub_f32(const float *a, const float *b, float *out, size_t n);
void chirp_sub_f64(const double *a, const double *b, double *out, size_t n);

/**
 * @brief Element-wise divide
 */
void chirp_div_f32(const float *a, const float *b, float *out, size_t n);
void chirp_div_f64(const double *a, const double *b, double *out, size_t n);

/**
 * @brief Scale vector by scalar (out = alpha * in)
 */
void chirp_scale_f32(float alpha, const float *in, float *out, size_t n);
void chirp_scale_f64(double alpha, const double *in, double *out, size_t n);

/**
 * @brief Scale and add (saxpy: out = alpha * x + y)
 */
void chirp_saxpy_f32(float alpha, const float *x, const float *y, float *out, size_t n);
void chirp_saxpy_f64(double alpha, const double *x, const double *y, double *out, size_t n);

/* ============================================================================
 * WINDOW FUNCTIONS
 * ============================================================================ */

/**
 * @brief Hann window
 * @param out Output array
 * @param n Length
 */
void chirp_hann_window_f32(float *out, size_t n);
void chirp_hann_window_f64(double *out, size_t n);

/**
 * @brief Hamming window
 */
void chirp_hamming_window_f32(float *out, size_t n);
void chirp_hamming_window_f64(double *out, size_t n);

/**
 * @brief Blackman window
 */
void chirp_blackman_window_f32(float *out, size_t n);
void chirp_blackman_window_f64(double *out, size_t n);

/**
 * @brief Gaussian window
 * @param sigma Standard deviation (in samples)
 */
void chirp_gaussian_window_f32(float *out, size_t n, float sigma);
void chirp_gaussian_window_f64(double *out, size_t n, double sigma);

/**
 * @brief Kaiser window
 * @param beta Shape parameter
 */
void chirp_kaiser_window_f32(float *out, size_t n, float beta);
void chirp_kaiser_window_f64(double *out, size_t n, double beta);

/**
 * @brief Flat-top window
 */
void chirp_flattop_window_f32(float *out, size_t n);
void chirp_flattop_window_f64(double *out, size_t n);

/* ============================================================================
 * WAVELET FUNCTIONS
 * ============================================================================ */

/**
 * @brief Haar wavelet transform (1 level)
 * @param in Input signal (length n)
 * @param approx Approximation coefficients (length n/2)
 * @param detail Detail coefficients (length n/2)
 * @param n Length (must be power of 2)
 */
void chirp_haar_f32(const float *in, float *approx, float *detail, size_t n);
void chirp_haar_f64(const double *in, double *approx, double *detail, size_t n);

/**
 * @brief Inverse Haar wavelet transform
 */
void chirp_haar_inverse_f32(const float *approx, const float *detail, float *out, size_t n);
void chirp_haar_inverse_f64(const double *approx, const double *detail, double *out, size_t n);

/**
 * @brief Daubechies-4 wavelet transform
 */
void chirp_daubechies4_f32(const float *in, float *approx, float *detail, size_t n);
void chirp_daubechies4_f64(const double *in, double *approx, double *detail, size_t n);

/**
 * @brief Meyer wavelet scaling function approximation
 * @param t Time value
 * @return Scaling function value
 */
float chirp_meyer_scaling_f32(float t);
double chirp_meyer_scaling_f64(double t);

/**
 * @brief Morlet wavelet
 * @param t Time value
 * @param omega0 Center frequency (typically 6)
 * @return Wavelet value
 */
float chirp_morlet_f32(float t, float omega0);
double chirp_morlet_f64(double t, double omega0);

/**
 * @brief Mexican hat (Ricker) wavelet
 * @param t Time value
 * @param sigma Width parameter
 * @return Wavelet value
 */
float chirp_mexican_hat_f32(float t, float sigma);
double chirp_mexican_hat_f64(double t, double sigma);

/**
 * @brief Shannon wavelet (sinc function)
 */
float chirp_shannon_wavelet_f32(float t);
double chirp_shannon_wavelet_f64(double t);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Clamp value to range [min, max]
 */
float chirp_clamp_f32(float x, float min, float max);
double chirp_clamp_f64(double x, double min, double max);

/**
 * @brief Linear interpolation
 * @param a Start value
 * @param b End value
 * @param t Interpolation factor (0 to 1)
 */
float chirp_lerp_f32(float a, float b, float t);
double chirp_lerp_f64(double a, double b, double t);

/**
 * @brief Smoothstep interpolation (3x^2 - 2x^3)
 */
float chirp_smoothstep_f32(float edge0, float edge1, float x);
double chirp_smoothstep_f64(double edge0, double edge1, double x);

/**
 * @brief Smootherstep interpolation (6x^5 - 15x^4 + 10x^3)
 */
float chirp_smootherstep_f32(float edge0, float edge1, float x);
double chirp_smootherstep_f64(double edge0, double edge1, double x);

/**
 * @brief Step function (Heaviside)
 */
float chirp_step_f32(float x);
double chirp_step_f64(double x);

/**
 * @brief Sign function
 */
float chirp_sign_f32(float x);
double chirp_sign_f64(double x);

/**
 * @brief Convert degrees to radians
 */
float chirp_deg2rad_f32(float deg);
double chirp_deg2rad_f64(double deg);

/**
 * @brief Convert radians to degrees
 */
float chirp_rad2deg_f32(float rad);
double chirp_rad2deg_f64(double rad);

#ifdef __cplusplus
}
#endif

#endif /* CHIRP_BUILTINS_H */
