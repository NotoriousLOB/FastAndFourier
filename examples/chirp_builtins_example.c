/**
 * @file chirp_builtins_example.c
 * @brief Demonstration of Chirp's comprehensive builtin library
 * 
 * This example shows how to use the standard mathematical functions,
 * distributions, vector operations, and wavelets available in Chirp.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fastandfourier.h"
#include "chirp.h"
#include "chirp_builtins.h"

/* Custom wavelet prediction function using Gaussian */
static void my_predict_gaussian(void) {
    printf("  [predict] Gaussian smoothing step\n");
}

/* Custom wavelet update function using Laplace */
static void my_update_laplace(void) {
    printf("  [update] Laplace edge-preserving step\n");
}

/* Print banner */
static void print_section(const char *title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║ %-60s ║\n", title);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

int main(void) {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║      CHIRP BUILTINS - THE FULL TOOLBOX (v1.0.0)                ║\n");
    printf("║      \"You want functions? We got functions.\"                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    /* Initialize */
    faf_init();
    
    /* Register ALL standard builtins (trig, distributions, etc.) */
    print_section("REGISTERING STANDARD BUILTINS");
    int n_builtins = chirp_register_standard_builtins();
    printf("✓ Registered %d builtin functions\n\n", n_builtins);
    
    /* Show a sample of what's available */
    printf("Sample of available functions:\n");
    printf("  • Trig: sin_f32, cos_f64, tanh_f32, ...\n");
    printf("  • Distributions: gaussian_pdf, cauchy_pdf, laplace_pdf, ...\n");
    printf("  • Vector: sum, dot, norm, mean, std, ...\n");
    printf("  • Windows: hann, hamming, blackman, flattop\n");
    printf("  • Wavelets: haar, daubechies4, morlet, mexican_hat\n");
    printf("  • Utils: lerp, smoothstep, clamp, sigmoid, gelu\n");
    
    /* Example 1: Simple trig pipeline */
    print_section("EXAMPLE 1: TRIGONOMETRIC PIPELINE");
    printf("Program: (pipeline (fft :size 512) twiddle)\n\n");
    
    faf_transform *t1 = chirp_compile("(pipeline (fft :size 512) twiddle)");
    if (t1) {
        printf("Generated %zu instructions\n", t1->n_inst);
        faf_destroy_transform(t1);
    }
    
    /* Example 2: Distribution-based filtering */
    print_section("EXAMPLE 2: DISTRIBUTION-BASED WAVELET");
    printf("Program:\n");
    printf("  (pipeline\n");
    printf("    (fft :size 1024)\n");
    printf("    (lift :predict gaussian_pdf :update laplace_pdf)\n");
    printf("    (custom sigmoid)\n");
    printf("    reduce-sum)\n\n");
    
    const char *dist_program = 
        "(pipeline "
        "  (fft :size 1024) "
        "  (lift :predict gaussian_pdf :update laplace_pdf) "
        "  (custom sigmoid) "
        "  reduce-sum)";
    
    faf_transform *t2 = chirp_compile(dist_program);
    if (t2) {
        printf("✓ Compiled successfully with %zu instructions\n", t2->n_inst);
        faf_destroy_transform(t2);
    }
    
    /* Example 3: Multi-stage butterfly with reductions */
    print_section("EXAMPLE 3: BUTTERFLY CASCADE + REDUCTIONS");
    printf("Program:\n");
    printf("  (pipeline\n");
    printf("    (bfly 2) (bfly 4) (bfly 8)\n");
    printf("    reduce-max\n");
    printf("    (custom relu))\n\n");
    
    faf_transform *t3 = chirp_compile(
        "(pipeline (bfly 2) (bfly 4) (bfly 8) reduce-max (custom relu))"
    );
    if (t3) {
        printf("✓ Butterfly cascade compiled: %zu instructions\n", t3->n_inst);
        faf_destroy_transform(t3);
    }
    
    /* Example 4: Activation function pipeline */
    print_section("EXAMPLE 4: NEURAL ACTIVATION PIPELINE");
    printf("Available activations: sigmoid, relu, gelu, swish, softmax\n");
    printf("New first-class syntax: (pipeline sigmoid gelu reduce-sum)\n");
    printf("Old syntax also works: (pipeline (custom sigmoid) (custom gelu))\n\n");
    
    faf_transform *t4 = chirp_compile(
        "(pipeline sigmoid gelu reduce-sum)"
    );
    if (t4) {
        printf("✓ Neural pipeline compiled: %zu instructions\n", t4->n_inst);
        faf_destroy_transform(t4);
    }
    
    /* Example 5: Window function generation */
    print_section("EXAMPLE 5: WINDOW FUNCTION PIPELINE");
    printf("Using builtin window functions:\n");
    printf("  • hann_window_f32 - Classic Hann window\n");
    printf("  • hamming_window_f32 - Hamming window\n");
    printf("  • blackman_window_f32 - Blackman window\n");
    printf("  • flattop_window_f32 - Flat-top (low amplitude error)\n\n");
    
    /* Show the C API usage of these functions */
    printf("C API Example:\n");
    float hann[64];
    chirp_hann_window_f32(hann, 64);
    printf("  chirp_hann_window_f32(hann, 64);\n");
    printf("  First 5 values: %.4f, %.4f, %.4f, %.4f, %.4f\n\n",
           hann[0], hann[1], hann[2], hann[3], hann[4]);
    
    /* Example 6: Vector operations */
    print_section("EXAMPLE 6: VECTOR OPERATIONS");
    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[8] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
    
    printf("Input vectors:\n");
    printf("  a = [1, 2, 3, 4, 5, 6, 7, 8]\n");
    printf("  b = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]\n\n");
    
    printf("Vector operations:\n");
    printf("  chirp_sum_f32(a, 8)     = %.2f\n", chirp_sum_f32(a, 8));
    printf("  chirp_mean_f32(a, 8)    = %.2f\n", chirp_mean_f32(a, 8));
    printf("  chirp_dot_f32(a, b, 8)  = %.2f\n", chirp_dot_f32(a, b, 8));
    printf("  chirp_norm_f32(a, 8)    = %.4f\n", chirp_norm_f32(a, 8));
    printf("  chirp_std_f32(a, 8)     = %.4f\n", chirp_std_f32(a, 8));
    
    float result[8];
    chirp_saxpy_f32(2.0f, a, b, result, 8);
    printf("  2*a + b (saxpy)         = [%.1f, %.1f, %.1f, ...]\n",
           result[0], result[1], result[2]);
    
    /* Example 7: Distribution evaluation */
    print_section("EXAMPLE 7: DISTRIBUTION PDFs");
    printf("Evaluating probability density functions at x=0.5:\n\n");
    
    float x = 0.5f;
    printf("  Gaussian(μ=0, σ=1):     %.4f\n", chirp_gaussian_pdf_f32(x, 0.0f, 1.0f));
    printf("  Cauchy(x₀=0, γ=1):      %.4f\n", chirp_cauchy_pdf_f32(x, 0.0f, 1.0f));
    printf("  Laplace(μ=0, b=1):      %.4f\n", chirp_laplace_pdf_f32(x, 0.0f, 1.0f));
    printf("  Exponential(λ=2):       %.4f\n", chirp_exponential_pdf_f32(x, 2.0f));
    printf("  LogNormal(μ=0, σ=0.5):  %.4f\n", chirp_lognormal_pdf_f32(1.5f, 0.0f, 0.5f));
    printf("  Beta(α=2, β=3):         %.4f\n", chirp_beta_pdf_f32(x, 2.0f, 3.0f));
    printf("  Gamma(k=2, θ=0.5):      %.4f\n", chirp_gamma_pdf_f32(x, 2.0f, 0.5f));
    printf("  Weibull(k=1.5, λ=1):    %.4f\n", chirp_weibull_pdf_f32(x, 1.5f, 1.0f));
    printf("  Uniform(a=0, b=1):      %.4f\n", chirp_uniform_pdf_f32(x, 0.0f, 1.0f));
    
    /* Example 8: Wavelet transform */
    print_section("EXAMPLE 8: WAVELET TRANSFORMS");
    printf("Haar wavelet transform of 8 samples:\n\n");
    
    float signal[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float approx[4], detail[4];
    float reconstructed[8];
    
    printf("Original:  [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           signal[0], signal[1], signal[2], signal[3],
           signal[4], signal[5], signal[6], signal[7]);
    
    chirp_haar_f32(signal, approx, detail, 8);
    printf("Approx:    [%.2f, %.2f, %.2f, %.2f]\n",
           approx[0], approx[1], approx[2], approx[3]);
    printf("Detail:    [%.2f, %.2f, %.2f, %.2f]\n",
           detail[0], detail[1], detail[2], detail[3]);
    
    chirp_haar_inverse_f32(approx, detail, reconstructed, 8);
    printf("Recon:     [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           reconstructed[0], reconstructed[1], reconstructed[2], reconstructed[3],
           reconstructed[4], reconstructed[5], reconstructed[6], reconstructed[7]);
    
    /* Example 9: Interpolation utilities */
    print_section("EXAMPLE 9: INTERPOLATION UTILITIES");
    printf("Interpolation functions:\n\n");
    
    printf("  lerp(0, 10, 0.3)        = %.2f\n", chirp_lerp_f32(0.0f, 10.0f, 0.3f));
    printf("  clamp(15, 0, 10)        = %.2f\n", chirp_clamp_f32(15.0f, 0.0f, 10.0f));
    printf("  smoothstep(0, 1, 0.3)   = %.4f\n", chirp_smoothstep_f32(0.0f, 1.0f, 0.3f));
    printf("  smootherstep(0, 1, 0.3) = %.4f\n", chirp_smootherstep_f32(0.0f, 1.0f, 0.3f));
    printf("  deg2rad(180)            = %.4f\n", chirp_deg2rad_f32(180.0f));
    printf("  rad2deg(π)              = %.4f\n", chirp_rad2deg_f32((float)M_PI));
    
    /* Example 10: Complex pipeline with multiple builtins */
    print_section("EXAMPLE 10: COMPLEX MULTI-STAGE PIPELINE");
    printf("This pipeline uses (first-class function syntax):\n");
    printf("  • FFT for frequency analysis\n");
    printf("  • Hann window for smoothing\n");
    printf("  • Sigmoid activation\n");
    printf("  • ReLU for thresholding\n");
    printf("  • Final reduction\n\n");
    
    const char *complex_pipeline = 
        "(pipeline "
        "  (fft :size 2048) "
        "  hann_window "
        "  sigmoid "
        "  relu "
        "  reduce-max)";
    
    faf_transform *t10 = chirp_compile(complex_pipeline);
    if (t10) {
        printf("✓ Complex pipeline: %zu instructions\n", t10->n_inst);
        faf_destroy_transform(t10);
    }
    
    /* Summary */
    print_section("SUMMARY");
    printf("✓ All %d standard builtins registered\n", n_builtins);
    printf("✓ Trigonometric functions: sin, cos, tan, hyperbolics\n");
    printf("✓ Distributions: 9 probability density functions\n");
    printf("✓ Vector ops: sum, dot, norm, mean, std, saxpy, etc.\n");
    printf("✓ Window functions: Hann, Hamming, Blackman, Flat-top\n");
    printf("✓ Wavelets: Haar, Daubechies-4, Morlet, Mexican Hat\n");
    printf("✓ Activations: sigmoid, ReLU, GELU, Swish, softmax\n");
    printf("✓ Utilities: lerp, smoothstep, clamp, sign\n");
    printf("✓ Special functions: erf, gamma, Bessel j0/y0\n\n");
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  \"With great functions comes great DSP capability\" - Uncle Ben ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    faf_cleanup();
    return 0;
}
