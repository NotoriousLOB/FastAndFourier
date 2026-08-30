cmake_minimum_required(VERSION 3.18)
project(FastAndFourier VERSION 1.1.0 LANGUAGES C CXX)

# C Standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_DOCS "Build documentation" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_FP8 "Enable FP8 precision" ON)
option(ENABLE_FP16 "Enable FP16 precision" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
option(ENABLE_WERROR "Treat compiler warnings as errors" OFF)
option(FASTANDFOURIER_SINGLE_HEADER "Build single-header fallback mode" OFF)
option(FASTANDFOURIER_PRECOMPILE "Pre-generate codelets for common N" OFF)

if(ENABLE_TSAN AND (ENABLE_ASAN OR ENABLE_UBSAN OR ENABLE_MSAN))
    message(FATAL_ERROR "ENABLE_TSAN is mutually exclusive with ASan/UBSan/MSan")
endif()
if(ENABLE_MSAN AND (ENABLE_ASAN OR ENABLE_TSAN))
    message(FATAL_ERROR "ENABLE_MSAN is mutually exclusive with ASan/TSan")
endif()

set(_FAF_SANITIZER_FLAGS "")
if(ENABLE_ASAN)
    list(APPEND _FAF_SANITIZER_FLAGS -fsanitize=address)
endif()
if(ENABLE_UBSAN)
    list(APPEND _FAF_SANITIZER_FLAGS -fsanitize=undefined)
endif()
if(ENABLE_TSAN)
    list(APPEND _FAF_SANITIZER_FLAGS -fsanitize=thread)
endif()
if(ENABLE_MSAN)
    list(APPEND _FAF_SANITIZER_FLAGS -fsanitize=memory)
endif()
if(_FAF_SANITIZER_FLAGS)
    add_compile_options(${_FAF_SANITIZER_FLAGS} -fno-omit-frame-pointer)
    add_link_options(${_FAF_SANITIZER_FLAGS})
endif()
if(ENABLE_WERROR)
    add_compile_options(-Werror)
endif()

# Include CMake modules for feature detection
include(CheckCSourceRuns)
include(CheckCSourceCompiles)
include(CMakePackageConfigHelpers)

# Architecture detection
set(ARCH_NAME "Unknown")
set(ARCH_X86_64 FALSE)
set(ARCH_AARCH64 FALSE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_X86_64 TRUE)
    set(ARCH_NAME "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_AARCH64 TRUE)
    set(ARCH_NAME "aarch64")
endif()

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -funroll-loops -march=native")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")

# Coverage flags
if(ENABLE_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Architecture-specific flags
if(ARCH_X86_64)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    add_compile_definitions(FAF_ARCH_X86_64)
    
    # Check for AVX512
    check_c_source_compiles("
        #include <immintrin.h>
        int main() { __m512i a = _mm512_set1_epi32(0); return 0; }
    " HAVE_AVX512)
    if(HAVE_AVX512)
        add_compile_definitions(FAF_HAVE_AVX512)
    endif()
    
    # Check for AVX2
    check_c_source_compiles("
        #include <immintrin.h>
        int main() { __m256i a = _mm256_set1_epi32(0); return 0; }
    " HAVE_AVX2)
    if(HAVE_AVX2)
        add_compile_definitions(FAF_HAVE_AVX2)
    endif()
    
    # Check for SSE4.2
    check_c_source_compiles("
        #include <nmmintrin.h>
        int main() { int a = 0; return _mm_popcnt_u32(a); }
    " HAVE_SSE42)
    if(HAVE_SSE42)
        add_compile_definitions(FAF_HAVE_SSE42)
    endif()
    
elseif(ARCH_AARCH64)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    add_compile_definitions(FAF_ARCH_AARCH64)
    
    # Check for SVE
    check_c_source_compiles("
        #include <arm_sve.h>
        int main() { svint32_t a = svdup_s32(0); return 0; }
    " HAVE_SVE)
    if(HAVE_SVE)
        add_compile_definitions(FAF_HAVE_SVE)
    endif()
    
    # Check for NEON
    check_c_source_compiles("
        #include <arm_neon.h>
        int main() { int32x4_t a = vdupq_n_s32(0); return 0; }
    " HAVE_NEON)
    if(HAVE_NEON)
        add_compile_definitions(FAF_HAVE_NEON)
    endif()
endif()

# CUDA setup
set(CUDA_FOUND FALSE)
if(ENABLE_CUDA)
    # For CMake 3.18+, use CUDAToolkit and enable_language properly
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CUDA_FOUND TRUE)
        add_compile_definitions(FAF_HAVE_CUDA)
        
        # Set CUDA architectures if not specified
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
        endif()
        
        # CUDA-specific flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --expt-relaxed-constexpr")
    else()
        message(WARNING "CUDA requested but not found. Disabling CUDA support.")
    endif()
endif()

# Precision levels
if(ENABLE_FP8)
    add_compile_definitions(FAF_ENABLE_FP8)
endif()
if(ENABLE_FP16)
    add_compile_definitions(FAF_ENABLE_FP16)
    if(CUDA_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFAF_ENABLE_FP16")
    endif()
endif()
add_compile_definitions(FAF_ENABLE_FP32)
add_compile_definitions(FAF_ENABLE_FP64)

# Single-header fallback mode (disables JIT, uses VM only)
if(FASTANDFOURIER_SINGLE_HEADER)
    add_compile_definitions(FAF_SINGLE_HEADER)
    message(STATUS "Single-header mode: JIT disabled, VM only")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(FAF_CORE_SOURCES
    src/faf_core.c
    src/faf_vm.c
    src/faf_transforms.c
    src/faf_twiddles.c
    src/faf_wavelets.c
    src/faf_rfft.c
    src/faf_fft_stage.c
    src/faf_pipeline.c
    src/chirp.c
    src/chirp_builtins.c
    src/faf_cwt.c
    src/faf_cwt_wavelets.c
    src/faf_cwt_kernels.c
)

set(FAF_X86_SOURCES
    src/arch/x86/faf_x86_sse.c
    src/arch/x86/faf_x86_avx2.c
    src/arch/x86/faf_x86_avx512.c
    src/arch/x86/faf_cwt_avx2.c
)

set(FAF_ARM_SOURCES
    src/arch/arm/faf_arm_neon.c
    src/arch/arm/faf_arm_sve.c
    src/arch/arm/faf_cwt_neon.c
)

set(FAF_CUDA_SOURCES
    src/arch/cuda/faf_cuda.cu
)

set(FAF_JIT_SOURCES
    src/jit/faf_jit.c
)

# Architecture-specific source selection
if(ARCH_X86_64)
    list(APPEND FAF_CORE_SOURCES ${FAF_X86_SOURCES})
elseif(ARCH_AARCH64)
    list(APPEND FAF_CORE_SOURCES ${FAF_ARM_SOURCES})
endif()

if(CUDA_FOUND)
    list(APPEND FAF_CORE_SOURCES ${FAF_CUDA_SOURCES})
endif()

list(APPEND FAF_CORE_SOURCES ${FAF_JIT_SOURCES})

# Main library
add_library(fastandfourier STATIC ${FAF_CORE_SOURCES})

# Set library properties
set_target_properties(fastandfourier PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/faf.h;include/fastandfourier.h;include/chirp.h;include/chirp_builtins.h"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
if(CUDA_FOUND)
    target_link_libraries(fastandfourier CUDA::toolkit)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(fastandfourier m dl pthread)
elseif(APPLE)
    target_link_libraries(fastandfourier m pthread)
elseif(WIN32)
    target_link_libraries(fastandfourier)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                       ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Install
install(TARGETS fastandfourier
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Package config
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/FastAndFourierConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/FastAndFourierConfigVersion.cmake"
    DESTINATION lib/cmake/FastAndFourier
)

# Print configuration summary
message(STATUS "")
message(STATUS "FastAndFourier Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Architecture: ${ARCH_NAME}")
message(STATUS "  CUDA support: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  FP8 precision: ${ENABLE_FP8}")
message(STATUS "  FP16 precision: ${ENABLE_FP16}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Code coverage: ${ENABLE_COVERAGE}")
message(STATUS "  AVX512: ${HAVE_AVX512}")
message(STATUS "  AVX2: ${HAVE_AVX2}")
message(STATUS "  SSE4.2: ${HAVE_SSE42}")
message(STATUS "")

