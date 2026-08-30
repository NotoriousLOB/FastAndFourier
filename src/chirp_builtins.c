/**
 * @file chirp_builtins.c
 * @brief Implementation of standard Chirp builtin functions
 *
 * A comprehensive library of mathematical functions, distributions,
 * and DSP primitives for the Chirp DSL.
 *
 * @version 1.1.0
 */

/* Enable POSIX/XOPEN extensions for Bessel functions (j0f, j1f, y0f, y1f) */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include "chirp_builtins.h"
#include "chirp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* External declarations from chirp.c — must match chirp_builtin layout */
typedef struct {
    char *name;
    void (*fn)(void);
    void (*fn_f64)(void);
    int kind;
    void *ctx;
} chirp_builtin_t;
extern chirp_builtin_t chirp_table[];
extern int g_chirp_count;

/* ============================================================================
 * ALIAS REGISTRATION
 * ============================================================================ */

/**
 * @brief Register friendly aliases for the standard builtins.
 *
 * The standard registration stores functions under their full C-style names
 * (e.g. "chirp_sin_f32"). Chirp programs are expected to refer to them by
 * shorter names ("sin", "sin_f32", "sigmoid", "hann_window", etc.). This
 * function walks the registry and adds those aliases.
 */
static void chirp_register_builtin_aliases(void) {
    /* Only walk the original entries; aliases are appended as we go and
     * must not be re-aliased themselves. */
    int n = g_chirp_count;

    for (int i = 0; i < n; i++) {
        const char *name = chirp_table[i].name;
        if (strncmp(name, "chirp_", 6) != 0) continue;

        const char *stripped = name + 6;

        /* Register the stripped name, e.g. "chirp_sin_f32" -> "sin_f32". */
        chirp_register_ex(stripped, chirp_table[i].fn, chirp_table[i].kind);

        /* For f32 variants, also register the bare name without the _f32
         * suffix, e.g. "sin_f32" -> "sin". This matches the names used in
         * the Chirp documentation and examples. */
        if (strstr(stripped, "_f32")) {
            char bare[256];
            size_t len = strlen(stripped);
            if (len >= sizeof(bare)) len = sizeof(bare) - 1;
            memcpy(bare, stripped, len);
            bare[len] = '\0';

            char *suff = strstr(bare, "_f32");
            if (suff) *suff = '\0';

            if (strlen(bare) > 0) {
                chirp_register_ex(bare, chirp_table[i].fn, chirp_table[i].kind);
            }
        }
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

/* ============================================================================
 * REGISTRATION
 * ============================================================================ */

/* Helper macro to register both f32 and f64 versions */
#define REGISTER_F32_F64(name_f32, name_f64) do { \
    chirp_register(#name_f32, (void(*)(void))name_f32); \
    chirp_register(#name_f64, (void(*)(void))name_f64); \
} while(0)

/* Register vector functions with size parameter */
#define REGISTER_VEC(name) do { \
    chirp_register_ex(#name "_f32", (void(*)(void))name##_f32, CHIRP_KIND_OTHER); \
    chirp_register_ex(#name "_f64", (void(*)(void))name##_f64, CHIRP_KIND_OTHER); \
} while(0)

#define REGISTER_OTHER_F32_F64(name_f32, name_f64) do { \
    chirp_register_ex(#name_f32, (void(*)(void))name_f32, CHIRP_KIND_OTHER); \
    chirp_register_ex(#name_f64, (void(*)(void))name_f64, CHIRP_KIND_OTHER); \
} while(0)

int chirp_register_standard_builtins(void) {
    int start_count = g_chirp_count;
    
    /* Trigonometric functions */
    REGISTER_F32_F64(chirp_sin_f32, chirp_sin_f64);
    REGISTER_F32_F64(chirp_cos_f32, chirp_cos_f64);
    REGISTER_F32_F64(chirp_tan_f32, chirp_tan_f64);
    REGISTER_F32_F64(chirp_asin_f32, chirp_asin_f64);
    REGISTER_F32_F64(chirp_acos_f32, chirp_acos_f64);
    REGISTER_F32_F64(chirp_atan_f32, chirp_atan_f64);
    REGISTER_F32_F64(chirp_sinh_f32, chirp_sinh_f64);
    REGISTER_F32_F64(chirp_cosh_f32, chirp_cosh_f64);
    REGISTER_F32_F64(chirp_tanh_f32, chirp_tanh_f64);
    
    /* Exponential and logarithmic */
    REGISTER_F32_F64(chirp_exp_f32, chirp_exp_f64);
    REGISTER_F32_F64(chirp_exp2_f32, chirp_exp2_f64);
    REGISTER_F32_F64(chirp_log_f32, chirp_log_f64);
    REGISTER_F32_F64(chirp_log2_f32, chirp_log2_f64);
    REGISTER_F32_F64(chirp_log10_f32, chirp_log10_f64);
    REGISTER_F32_F64(chirp_sqrt_f32, chirp_sqrt_f64);
    REGISTER_F32_F64(chirp_cbrt_f32, chirp_cbrt_f64);
    
    /* Special functions */
    REGISTER_F32_F64(chirp_erf_f32, chirp_erf_f64);
    REGISTER_F32_F64(chirp_erfc_f32, chirp_erfc_f64);
    REGISTER_F32_F64(chirp_tgamma_f32, chirp_tgamma_f64);
    REGISTER_F32_F64(chirp_lgamma_f32, chirp_lgamma_f64);
    
    /* Bessel functions */
    REGISTER_F32_F64(chirp_j0_f32, chirp_j0_f64);
    REGISTER_F32_F64(chirp_j1_f32, chirp_j1_f64);
    REGISTER_F32_F64(chirp_y0_f32, chirp_y0_f64);
    REGISTER_F32_F64(chirp_y1_f32, chirp_y1_f64);
    
    /* Distributions (extra parameters — not unary) */
    REGISTER_OTHER_F32_F64(chirp_gaussian_pdf_f32, chirp_gaussian_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_cauchy_pdf_f32, chirp_cauchy_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_exponential_pdf_f32, chirp_exponential_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_lognormal_pdf_f32, chirp_lognormal_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_laplace_pdf_f32, chirp_laplace_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_beta_pdf_f32, chirp_beta_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_gamma_pdf_f32, chirp_gamma_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_weibull_pdf_f32, chirp_weibull_pdf_f64);
    REGISTER_OTHER_F32_F64(chirp_uniform_pdf_f32, chirp_uniform_pdf_f64);
    
    /* Activation functions */
    REGISTER_F32_F64(chirp_sigmoid_f32, chirp_sigmoid_f64);
    REGISTER_F32_F64(chirp_relu_f32, chirp_relu_f64);
    REGISTER_F32_F64(chirp_gelu_f32, chirp_gelu_f64);
    REGISTER_F32_F64(chirp_swish_f32, chirp_swish_f64);
    REGISTER_VEC(chirp_softmax);
    
    /* Vector operations */
    REGISTER_VEC(chirp_sum);
    REGISTER_VEC(chirp_dot);
    REGISTER_VEC(chirp_norm);
    REGISTER_VEC(chirp_norm1);
    REGISTER_VEC(chirp_norm_inf);
    REGISTER_VEC(chirp_mean);
    REGISTER_VEC(chirp_variance);
    REGISTER_VEC(chirp_std);
    REGISTER_VEC(chirp_cumsum);
    REGISTER_VEC(chirp_mul);
    REGISTER_VEC(chirp_add);
    REGISTER_VEC(chirp_sub);
    REGISTER_VEC(chirp_div);
    REGISTER_VEC(chirp_scale);
    REGISTER_VEC(chirp_saxpy);
    
    /* Window functions */
    REGISTER_VEC(chirp_hann_window);
    REGISTER_VEC(chirp_hamming_window);
    REGISTER_VEC(chirp_blackman_window);
    REGISTER_VEC(chirp_flattop_window);
    
    /* Wavelet functions */
    REGISTER_VEC(chirp_haar);
    REGISTER_VEC(chirp_haar_inverse);
    REGISTER_VEC(chirp_daubechies4);
    REGISTER_OTHER_F32_F64(chirp_morlet_f32, chirp_morlet_f64);
    REGISTER_OTHER_F32_F64(chirp_mexican_hat_f32, chirp_mexican_hat_f64);
    
    /* Utilities (clamp/lerp/smoothstep take extra args) */
    REGISTER_OTHER_F32_F64(chirp_clamp_f32, chirp_clamp_f64);
    REGISTER_OTHER_F32_F64(chirp_lerp_f32, chirp_lerp_f64);
    REGISTER_OTHER_F32_F64(chirp_smoothstep_f32, chirp_smoothstep_f64);
    REGISTER_OTHER_F32_F64(chirp_smootherstep_f32, chirp_smootherstep_f64);
    REGISTER_F32_F64(chirp_step_f32, chirp_step_f64);
    REGISTER_F32_F64(chirp_sign_f32, chirp_sign_f64);
    REGISTER_F32_F64(chirp_deg2rad_f32, chirp_deg2rad_f64);
    REGISTER_F32_F64(chirp_rad2deg_f32, chirp_rad2deg_f64);

    /* Add friendly aliases so Chirp programs can use short names like
     * "sin", "sigmoid", "hann_window" instead of "chirp_sin_f32". */
    chirp_register_builtin_aliases();

    return g_chirp_count - start_count; /* Return number of builtins registered */
}

void chirp_list_builtins(void) {
    printf("Registered Chirp builtins (%d total):\n", g_chirp_count);
    printf("%-30s %-10s\n", "Name", "Type");
    printf("%-30s %-10s\n", "----", "----");
    
    for (int i = 0; i < g_chirp_count; i++) {
        const char *type = "scalar";
        if (strstr(chirp_table[i].name, "_f64")) type = "f64";
        else if (strstr(chirp_table[i].name, "_f32")) type = "f32";
        else if (strstr(chirp_table[i].name, "_pdf_")) type = "dist";
        else if (strstr(chirp_table[i].name, "window")) type = "window";
        else if (strstr(chirp_table[i].name, "haar") || 
                 strstr(chirp_table[i].name, "daubechies")) type = "wavelet";
        
        printf("%-30s %-10s\n", chirp_table[i].name, type);
    }
}

/* ============================================================================
 * TRIGONOMETRIC FUNCTIONS
 * ============================================================================ */

float chirp_sin_f32(float x) { return sinf(x); }
float chirp_cos_f32(float x) { return cosf(x); }
float chirp_tan_f32(float x) { return tanf(x); }
float chirp_asin_f32(float x) { return asinf(x); }
float chirp_acos_f32(float x) { return acosf(x); }
float chirp_atan_f32(float x) { return atanf(x); }
float chirp_sinh_f32(float x) { return sinhf(x); }
float chirp_cosh_f32(float x) { return coshf(x); }
float chirp_tanh_f32(float x) { return tanhf(x); }

double chirp_sin_f64(double x) { return sin(x); }
double chirp_cos_f64(double x) { return cos(x); }
double chirp_tan_f64(double x) { return tan(x); }
double chirp_asin_f64(double x) { return asin(x); }
double chirp_acos_f64(double x) { return acos(x); }
double chirp_atan_f64(double x) { return atan(x); }
double chirp_sinh_f64(double x) { return sinh(x); }
double chirp_cosh_f64(double x) { return cosh(x); }
double chirp_tanh_f64(double x) { return tanh(x); }

/* ============================================================================
 * EXPONENTIAL AND LOGARITHMIC
 * ============================================================================ */

float chirp_exp_f32(float x) { return expf(x); }
float chirp_exp2_f32(float x) { return exp2f(x); }
float chirp_log_f32(float x) { return logf(x); }
float chirp_log2_f32(float x) { return log2f(x); }
float chirp_log10_f32(float x) { return log10f(x); }
float chirp_pow_f32(float x, float y) { return powf(x, y); }
float chirp_sqrt_f32(float x) { return sqrtf(x); }
float chirp_cbrt_f32(float x) { return cbrtf(x); }
float chirp_hypot_f32(float x, float y) { return hypotf(x, y); }

double chirp_exp_f64(double x) { return exp(x); }
double chirp_exp2_f64(double x) { return exp2(x); }
double chirp_log_f64(double x) { return log(x); }
double chirp_log2_f64(double x) { return log2(x); }
double chirp_log10_f64(double x) { return log10(x); }
double chirp_pow_f64(double x, double y) { return pow(x, y); }
double chirp_sqrt_f64(double x) { return sqrt(x); }
double chirp_cbrt_f64(double x) { return cbrt(x); }
double chirp_hypot_f64(double x, double y) { return hypot(x, y); }

/* ============================================================================
 * SPECIAL FUNCTIONS
 * ============================================================================ */

float chirp_erf_f32(float x) { return erff(x); }
float chirp_erfc_f32(float x) { return erfcf(x); }
float chirp_tgamma_f32(float x) { return tgammaf(x); }
float chirp_lgamma_f32(float x) { return lgammaf(x); }

double chirp_erf_f64(double x) { return erf(x); }
double chirp_erfc_f64(double x) { return erfc(x); }
double chirp_tgamma_f64(double x) { return tgamma(x); }
double chirp_lgamma_f64(double x) { return lgamma(x); }

/* Bessel functions (j0f/j1f/y0f/y1f absent on macOS; cast through double) */
float chirp_j0_f32(float x) { return (float)j0((double)x); }
float chirp_j1_f32(float x) { return (float)j1((double)x); }
float chirp_y0_f32(float x) { return (float)y0((double)x); }
float chirp_y1_f32(float x) { return (float)y1((double)x); }

double chirp_j0_f64(double x) { return j0(x); }
double chirp_j1_f64(double x) { return j1(x); }
double chirp_y0_f64(double x) { return y0(x); }
double chirp_y1_f64(double x) { return y1(x); }

/* ============================================================================
 * DISTRIBUTIONS (PDFs)
 * ============================================================================ */

float chirp_gaussian_pdf_f32(float x, float mu, float sigma) {
    float z = (x - mu) / sigma;
    return expf(-0.5f * z * z) / (sigma * sqrtf(2.0f * (float)M_PI));
}

double chirp_gaussian_pdf_f64(double x, double mu, double sigma) {
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z) / (sigma * sqrt(2.0 * M_PI));
}

float chirp_cauchy_pdf_f32(float x, float x0, float gamma) {
    float z = (x - x0) / gamma;
    return 1.0f / ((float)M_PI * gamma * (1.0f + z * z));
}

double chirp_cauchy_pdf_f64(double x, double x0, double gamma) {
    double z = (x - x0) / gamma;
    return 1.0 / (M_PI * gamma * (1.0 + z * z));
}

float chirp_exponential_pdf_f32(float x, float lambda) {
    if (x < 0.0f) return 0.0f;
    return lambda * expf(-lambda * x);
}

double chirp_exponential_pdf_f64(double x, double lambda) {
    if (x < 0.0) return 0.0;
    return lambda * exp(-lambda * x);
}

float chirp_lognormal_pdf_f32(float x, float mu, float sigma) {
    if (x <= 0.0f) return 0.0f;
    float logx = logf(x);
    float z = (logx - mu) / sigma;
    return expf(-0.5f * z * z) / (x * sigma * sqrtf(2.0f * (float)M_PI));
}

double chirp_lognormal_pdf_f64(double x, double mu, double sigma) {
    if (x <= 0.0) return 0.0;
    double logx = log(x);
    double z = (logx - mu) / sigma;
    return exp(-0.5 * z * z) / (x * sigma * sqrt(2.0 * M_PI));
}

float chirp_laplace_pdf_f32(float x, float mu, float b) {
    return expf(-fabsf(x - mu) / b) / (2.0f * b);
}

double chirp_laplace_pdf_f64(double x, double mu, double b) {
    return exp(-fabs(x - mu) / b) / (2.0 * b);
}

float chirp_beta_pdf_f32(float x, float alpha, float beta) {
    if (x <= 0.0f || x >= 1.0f) return 0.0f;
    /* Use log-gamma for numerical stability */
    float logB = lgammaf(alpha) + lgammaf(beta) - lgammaf(alpha + beta);
    return expf((alpha - 1.0f) * logf(x) + (beta - 1.0f) * logf(1.0f - x) - logB);
}

double chirp_beta_pdf_f64(double x, double alpha, double beta) {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double logB = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta);
    return exp((alpha - 1.0) * log(x) + (beta - 1.0) * log(1.0 - x) - logB);
}

float chirp_gamma_pdf_f32(float x, float k, float theta) {
    if (x < 0.0f) return 0.0f;
    if (x == 0.0f && k < 1.0f) return 0.0f;
    float log_norm = lgammaf(k) + k * logf(theta);
    return expf((k - 1.0f) * logf(x) - x / theta - log_norm);
}

double chirp_gamma_pdf_f64(double x, double k, double theta) {
    if (x < 0.0) return 0.0;
    if (x == 0.0 && k < 1.0) return 0.0;
    double log_norm = lgamma(k) + k * log(theta);
    return exp((k - 1.0) * log(x) - x / theta - log_norm);
}

float chirp_weibull_pdf_f32(float x, float k, float lambda) {
    if (x < 0.0f) return 0.0f;
    if (x == 0.0f && k < 1.0f) return 0.0f;
    float z = x / lambda;
    return (k / lambda) * powf(z, k - 1.0f) * expf(-powf(z, k));
}

double chirp_weibull_pdf_f64(double x, double k, double lambda) {
    if (x < 0.0) return 0.0;
    if (x == 0.0 && k < 1.0) return 0.0;
    double z = x / lambda;
    return (k / lambda) * pow(z, k - 1.0) * exp(-pow(z, k));
}

float chirp_uniform_pdf_f32(float x, float a, float b) {
    if (x < a || x > b) return 0.0f;
    return 1.0f / (b - a);
}

double chirp_uniform_pdf_f64(double x, double a, double b) {
    if (x < a || x > b) return 0.0;
    return 1.0 / (b - a);
}

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

float chirp_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

double chirp_sigmoid_f64(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void chirp_softmax_f32(const float *in, float *out, size_t n) {
    /* Find max for numerical stability */
    float max_val = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    
    /* Compute exp and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    
    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

void chirp_softmax_f64(const double *in, double *out, size_t n) {
    double max_val = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out[i] = exp(in[i] - max_val);
        sum += out[i];
    }
    
    double inv_sum = 1.0 / sum;
    for (size_t i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

float chirp_relu_f32(float x) {
    return x > 0.0f ? x : 0.0f;
}

double chirp_relu_f64(double x) {
    return x > 0.0 ? x : 0.0;
}

/* GELU: x * Φ(x) where Φ is the CDF of the standard normal */
float chirp_gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

double chirp_gelu_f64(double x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

float chirp_swish_f32(float x) {
    return x * chirp_sigmoid_f32(x);
}

double chirp_swish_f64(double x) {
    return x * chirp_sigmoid_f64(x);
}

/* ============================================================================
 * VECTOR OPERATIONS
 * ============================================================================ */

float chirp_sum_f32(const float *vec, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += vec[i];
    return sum;
}

double chirp_sum_f64(const double *vec, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += vec[i];
    return sum;
}

float chirp_dot_f32(const float *a, const float *b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

double chirp_dot_f64(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

float chirp_norm_f32(const float *vec, size_t n) {
    return sqrtf(chirp_dot_f32(vec, vec, n));
}

double chirp_norm_f64(const double *vec, size_t n) {
    return sqrt(chirp_dot_f64(vec, vec, n));
}

float chirp_norm1_f32(const float *vec, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += fabsf(vec[i]);
    return sum;
}

double chirp_norm1_f64(const double *vec, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += fabs(vec[i]);
    return sum;
}

float chirp_norm_inf_f32(const float *vec, size_t n) {
    float max_val = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float abs_val = fabsf(vec[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

double chirp_norm_inf_f64(const double *vec, size_t n) {
    double max_val = 0.0;
    for (size_t i = 0; i < n; i++) {
        double abs_val = fabs(vec[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

float chirp_mean_f32(const float *vec, size_t n) {
    return chirp_sum_f32(vec, n) / (float)n;
}

double chirp_mean_f64(const double *vec, size_t n) {
    return chirp_sum_f64(vec, n) / (double)n;
}

float chirp_variance_f32(const float *vec, size_t n) {
    float mean = chirp_mean_f32(vec, n);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = vec[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (float)n;
}

double chirp_variance_f64(const double *vec, size_t n) {
    double mean = chirp_mean_f64(vec, n);
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = vec[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (double)n;
}

float chirp_std_f32(const float *vec, size_t n) {
    return sqrtf(chirp_variance_f32(vec, n));
}

double chirp_std_f64(const double *vec, size_t n) {
    return sqrt(chirp_variance_f64(vec, n));
}

void chirp_cumsum_f32(const float *in, float *out, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += in[i];
        out[i] = sum;
    }
}

void chirp_cumsum_f64(const double *in, double *out, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += in[i];
        out[i] = sum;
    }
}

void chirp_mul_f32(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

void chirp_mul_f64(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

void chirp_add_f32(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void chirp_add_f64(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void chirp_sub_f32(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void chirp_sub_f64(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void chirp_div_f32(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}

void chirp_div_f64(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}

void chirp_scale_f32(float alpha, const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = alpha * in[i];
}

void chirp_scale_f64(double alpha, const double *in, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = alpha * in[i];
}

void chirp_saxpy_f32(float alpha, const float *x, const float *y, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = alpha * x[i] + y[i];
}

void chirp_saxpy_f64(double alpha, const double *x, const double *y, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = alpha * x[i] + y[i];
}

/* ============================================================================
 * WINDOW FUNCTIONS
 * ============================================================================ */

void chirp_hann_window_f32(float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
    }
}

void chirp_hann_window_f64(double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));
    }
}

void chirp_hamming_window_f32(float *out, size_t n) {
    const float a0 = 0.54f;
    const float a1 = 0.46f;
    for (size_t i = 0; i < n; i++) {
        out[i] = a0 - a1 * cosf(2.0f * (float)M_PI * i / (n - 1));
    }
}

void chirp_hamming_window_f64(double *out, size_t n) {
    const double a0 = 0.54;
    const double a1 = 0.46;
    for (size_t i = 0; i < n; i++) {
        out[i] = a0 - a1 * cos(2.0 * M_PI * i / (n - 1));
    }
}

void chirp_blackman_window_f32(float *out, size_t n) {
    const float a0 = 0.42f;
    const float a1 = 0.5f;
    const float a2 = 0.08f;
    for (size_t i = 0; i < n; i++) {
        float x = 2.0f * (float)M_PI * i / (n - 1);
        out[i] = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x);
    }
}

void chirp_blackman_window_f64(double *out, size_t n) {
    const double a0 = 0.42;
    const double a1 = 0.5;
    const double a2 = 0.08;
    for (size_t i = 0; i < n; i++) {
        double x = 2.0 * M_PI * i / (n - 1);
        out[i] = a0 - a1 * cos(x) + a2 * cos(2.0 * x);
    }
}

void chirp_flattop_window_f32(float *out, size_t n) {
    const float a0 = 0.21557895f;
    const float a1 = 0.41663158f;
    const float a2 = 0.277263158f;
    const float a3 = 0.083578947f;
    const float a4 = 0.006947368f;
    for (size_t i = 0; i < n; i++) {
        float x = 2.0f * (float)M_PI * i / (n - 1);
        out[i] = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x) 
                 - a3 * cosf(3.0f * x) + a4 * cosf(4.0f * x);
    }
}

void chirp_flattop_window_f64(double *out, size_t n) {
    const double a0 = 0.21557895;
    const double a1 = 0.41663158;
    const double a2 = 0.277263158;
    const double a3 = 0.083578947;
    const double a4 = 0.006947368;
    for (size_t i = 0; i < n; i++) {
        double x = 2.0 * M_PI * i / (n - 1);
        out[i] = a0 - a1 * cos(x) + a2 * cos(2.0 * x) 
                 - a3 * cos(3.0 * x) + a4 * cos(4.0 * x);
    }
}

/* ============================================================================
 * WAVELET FUNCTIONS
 * ============================================================================ */

void chirp_haar_f32(const float *in, float *approx, float *detail, size_t n) {
    size_t half = n / 2;
    const float s = 0.7071067811865476f;
    for (size_t i = 0; i < half; i++) {
        float a = in[2*i];
        float b = in[2*i + 1];
        approx[i] = (a + b) * s;
        detail[i] = (a - b) * s;
    }
}

void chirp_haar_f64(const double *in, double *approx, double *detail, size_t n) {
    size_t half = n / 2;
    const double s = 0.70710678118654752440;
    for (size_t i = 0; i < half; i++) {
        double a = in[2*i];
        double b = in[2*i + 1];
        approx[i] = (a + b) * s;
        detail[i] = (a - b) * s;
    }
}

void chirp_haar_inverse_f32(const float *approx, const float *detail, float *out, size_t n) {
    size_t half = n / 2;
    const float s = 0.7071067811865476f;
    for (size_t i = 0; i < half; i++) {
        float a = approx[i];
        float d = detail[i];
        out[2*i] = (a + d) * s;
        out[2*i + 1] = (a - d) * s;
    }
}

void chirp_haar_inverse_f64(const double *approx, const double *detail, double *out, size_t n) {
    size_t half = n / 2;
    const double s = 0.70710678118654752440;
    for (size_t i = 0; i < half; i++) {
        double a = approx[i];
        double d = detail[i];
        out[2*i] = (a + d) * s;
        out[2*i + 1] = (a - d) * s;
    }
}

void chirp_daubechies4_f32(const float *in, float *approx, float *detail, size_t n) {
    const float h[4] = {
        0.4829629131445341f, 0.8365163037378079f,
        0.2241438680420134f, -0.1294095225512604f
    };
    size_t half = n / 2;
    for (size_t k = 0; k < half; k++) {
        float a = 0.0f, d = 0.0f;
        for (int i = 0; i < 4; i++) {
            float s = in[(2 * k + (size_t)i) % n];
            float g = ((i & 1) ? -1.0f : 1.0f) * h[3 - i];
            a += h[i] * s;
            d += g * s;
        }
        approx[k] = a;
        detail[k] = d;
    }
}

void chirp_daubechies4_f64(const double *in, double *approx, double *detail, size_t n) {
    const double h[4] = {
        0.4829629131445341, 0.8365163037378079,
        0.2241438680420134, -0.1294095225512604
    };
    size_t half = n / 2;
    for (size_t k = 0; k < half; k++) {
        double a = 0.0, d = 0.0;
        for (int i = 0; i < 4; i++) {
            double s = in[(2 * k + (size_t)i) % n];
            double g = ((i & 1) ? -1.0 : 1.0) * h[3 - i];
            a += h[i] * s;
            d += g * s;
        }
        approx[k] = a;
        detail[k] = d;
    }
}

float chirp_morlet_f32(float t, float omega0) {
    return expf(-t * t * 0.5f) * cosf(omega0 * t);
}

double chirp_morlet_f64(double t, double omega0) {
    return exp(-t * t * 0.5) * cos(omega0 * t);
}

float chirp_mexican_hat_f32(float t, float sigma) {
    float x = t / sigma;
    float x2 = x * x;
    return (2.0f / (sqrtf(3.0f * (float)M_PI) * sigma)) * 
           (1.0f - x2) * expf(-x2 * 0.5f);
}

double chirp_mexican_hat_f64(double t, double sigma) {
    double x = t / sigma;
    double x2 = x * x;
    return (2.0 / (sqrt(3.0 * M_PI) * sigma)) * 
           (1.0 - x2) * exp(-x2 * 0.5);
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

float chirp_clamp_f32(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

double chirp_clamp_f64(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

float chirp_lerp_f32(float a, float b, float t) {
    return a + t * (b - a);
}

double chirp_lerp_f64(double a, double b, double t) {
    return a + t * (b - a);
}

float chirp_smoothstep_f32(float edge0, float edge1, float x) {
    float t = chirp_clamp_f32((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

double chirp_smoothstep_f64(double edge0, double edge1, double x) {
    double t = chirp_clamp_f64((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float chirp_smootherstep_f32(float edge0, float edge1, float x) {
    float t = chirp_clamp_f32((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

double chirp_smootherstep_f64(double edge0, double edge1, double x) {
    double t = chirp_clamp_f64((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float chirp_step_f32(float x) {
    return x >= 0.0f ? 1.0f : 0.0f;
}

double chirp_step_f64(double x) {
    return x >= 0.0 ? 1.0 : 0.0;
}

float chirp_sign_f32(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

double chirp_sign_f64(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

float chirp_deg2rad_f32(float deg) {
    return deg * (float)M_PI / 180.0f;
}

double chirp_deg2rad_f64(double deg) {
    return deg * M_PI / 180.0;
}

float chirp_rad2deg_f32(float rad) {
    return rad * 180.0f / (float)M_PI;
}

double chirp_rad2deg_f64(double rad) {
    return rad * 180.0 / M_PI;
}
