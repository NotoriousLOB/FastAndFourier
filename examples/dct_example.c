/**
 * @file dct_example.c
 * @brief DCT (Discrete Cosine Transform) example
 */

#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("FastAndFourier - DCT Example\n");
    printf("=============================\n\n");
    
    faf_init();
    
    /* Create a 64-point DCT-II */
    const size_t n = 64;
    faf_config cfg = faf_config_init(n);
    cfg.dct_type = 2;
    cfg.layout = FAF_LAYOUT_INTERLEAVED;
    faf_transform* dct = faf_create_dct(&cfg);
    if (!dct) {
        fprintf(stderr, "Failed to create DCT: %s\n", faf_get_error());
        return 1;
    }
    
    float* in = (float*)aligned_alloc(64, n * sizeof(float));
    float* out = (float*)aligned_alloc(64, n * sizeof(float));
    
    /* Create a cosine signal */
    for (size_t i = 0; i < n; i++) {
        in[i] = cosf((float)M_PI * (float)i / (float)n);
    }
    
    printf("Computing DCT-II of cosine signal...\n\n");
    
    /* Execute DCT */
    faf_execute_f32(dct, out, in);
    
    /* DCT of cosine should have energy concentrated at low frequencies */
    printf("DCT coefficients (showing significant values):\n");
    printf("Index\tValue\n");
    printf("-----\t-----\n");
    
    for (size_t i = 0; i < n; i++) {
        if (fabsf(out[i]) > 0.5f) {
            printf("%zu\t%.4f\n", i, out[i]);
        }
    }
    
    /* Energy compaction demonstration */
    float total_energy = 0.0f;
    float low_freq_energy = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float energy = out[i] * out[i];
        total_energy += energy;
        if (i < n / 4) {
            low_freq_energy += energy;
        }
    }
    
    printf("\nEnergy compaction:\n");
    printf("  Total energy: %.2f\n", total_energy);
    printf("  Energy in first quarter: %.2f (%.1f%%)\n", 
           low_freq_energy, 100.0f * low_freq_energy / total_energy);
    
    free(in);
    free(out);
    faf_destroy_transform(dct);
    faf_cleanup();
    
    return 0;
}
