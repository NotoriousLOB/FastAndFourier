function(detect_architecture)
    set(ARCH_NAME "unknown")
    set(ARCH_X86_64 FALSE)
    set(ARCH_AARCH64 FALSE)
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(ARCH_NAME "x86_64" PARENT_SCOPE)
        set(ARCH_X86_64 TRUE PARENT_SCOPE)
        
        # Check for instruction set support
        include(CheckCSourceRuns)
        
        # Check AVX-512
        check_c_source_runs("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_set1_ps(1.0f);
                __m512 b = _mm512_set1_ps(2.0f);
                __m512 c = _mm512_add_ps(a, b);
                return 0;
            }
        " HAVE_AVX512)
        
        # Check AVX2
        check_c_source_runs("
            #include <immintrin.h>
            int main() {
                __m256i a = _mm256_set1_epi32(1);
                __m256i b = _mm256_set1_epi32(2);
                __m256i c = _mm256_add_epi32(a, b);
                return 0;
            }
        " HAVE_AVX2)
        
        # Check SSE4.2
        check_c_source_runs("
            #include <nmmintrin.h>
            int main() {
                __m128i a = _mm_set1_epi32(1);
                __m128i b = _mm_set1_epi32(2);
                __m128i c = _mm_add_epi32(a, b);
                return 0;
            }
        " HAVE_SSE42)
        
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        set(ARCH_NAME "aarch64" PARENT_SCOPE)
        set(ARCH_AARCH64 TRUE PARENT_SCOPE)
        
        # Check for NEON (always available on aarch64)
        set(HAVE_NEON TRUE)
        
        # Check SVE
        check_c_source_runs("
            #include <arm_sve.h>
            int main() {
                svfloat32_t a = svdup_f32(1.0f);
                svfloat32_t b = svdup_f32(2.0f);
                svfloat32_t c = svadd_f32_z(svptrue_b32(), a, b);
                return 0;
            }
        " HAVE_SVE)
    endif()
    
    set(HAVE_NEON ${HAVE_NEON} PARENT_SCOPE)
    set(HAVE_SVE ${HAVE_SVE} PARENT_SCOPE)
    set(HAVE_AVX512 ${HAVE_AVX512} PARENT_SCOPE)
    set(HAVE_AVX2 ${HAVE_AVX2} PARENT_SCOPE)
    set(HAVE_SSE42 ${HAVE_SSE42} PARENT_SCOPE)
    
endfunction()
