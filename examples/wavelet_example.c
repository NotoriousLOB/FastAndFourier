/**
 * @file wavelet_example.c
 * @brief Wavelet transform example
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - Wavelet Transform Example\n");
    printf("===========================================\n\n");
    
    faf_init();
    
    /* Create a 128-point Haar wavelet with 3 decomposition levels */
    const size_t n = 128;
    const size_t levels = 3;
    
    faf_transform* haar = faf_create_haar(n, levels, FAF_PREC_FP32, 0);
    if (!haar) {
        fprintf(stderr, "Failed to create Haar transform: %s\n", faf_get_error());
        return 1;
    }
    
    float* in = (float*)aligned_alloc(64, n * sizeof(float));
    float* out = (float*)aligned_alloc(64, n * sizeof(float));
    
    /* Create a signal with a discontinuity (edge) */
    for (size_t i = 0; i < n / 2; i++) {
        in[i] = 0.0f;
    }
    for (size_t i = n / 2; i < n; i++) {
        in[i] = 1.0f;
    }
    
    printf("Input: Step function (0 for first half, 1 for second half)\n\n");
    
    /* Execute Haar transform */
    faf_execute_f32(haar, out, in);
    
    /* Display wavelet coefficients */
    printf("Wavelet coefficients after %zu-level decomposition:\n", levels);
    printf("Level\tApproximation\tDetail Coefficients\n");
    printf("-----\t-------------\t-------------------\n");
    
    size_t offset = 0;
    size_t current_n = n;
    
    for (size_t level = 0; level < levels; level++) {
        printf("%zu\t%.4f\t\t", level + 1, out[offset]);
        
        /* Show first few detail coefficients */
        printf("[");
        for (size_t i = 0; i < 4 && i < current_n / 2; i++) {
            printf("%.3f", out[offset + current_n / 2 + i]);
            if (i < 3 && i < current_n / 2 - 1) printf(", ");
        }
        printf("...]\n");
        
        offset += current_n / 2;
        current_n /= 2;
    }
    
    printf("\nNote: Large detail coefficients indicate edges in the signal.\n");
    
    free(in);
    free(out);
    faf_destroy_transform(haar);
    faf_cleanup();
    
    return 0;
}
