# Benchmarks CMakeLists.txt

include(FetchContent)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# External FFT libraries (optional)
option(ENABLE_EXTERNAL_BENCHMARKS "Enable benchmarks against external FFT libraries" OFF)

if(ENABLE_EXTERNAL_BENCHMARKS)
    # Kiss FFT - uses FetchContent
    FetchContent_Declare(
        kissfft
        GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
        GIT_TAG 131.1.0
    )
    set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kissfft)

    # PocketFFT - header only, just get the source
    FetchContent_Declare(
        pocketfft
        GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
        GIT_TAG master
    )
    FetchContent_GetProperties(pocketfft)
    if(NOT pocketfft_POPULATED)
        FetchContent_Populate(pocketfft)
    endif()

    # MinFFT - single header
    FetchContent_Declare(
        minfft
        GIT_REPOSITORY https://github.com/aimukhin/minfft.git
        GIT_TAG master
    )
    FetchContent_GetProperties(minfft)
    if(NOT minfft_POPULATED)
        FetchContent_Populate(minfft)
    endif()

    # muFFT - single header
    FetchContent_Declare(
        mufft
        GIT_REPOSITORY https://github.com/Themaister/muFFT.git
        GIT_TAG master
    )
    FetchContent_GetProperties(mufft)
    if(NOT mufft_POPULATED)
        FetchContent_Populate(mufft)
    endif()
endif()

# Find FFTW3 (system or build from source)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3F fftw3f)
    pkg_check_modules(FFTW3 fftw3)
endif()

if(ENABLE_EXTERNAL_BENCHMARKS AND NOT FFTW3F_FOUND)
    # Build FFTW3 from source using FetchContent
    FetchContent_Declare(
        fftw3
        GIT_REPOSITORY https://github.com/FFTW/fftw3.git
        GIT_TAG master
    )
    set(ENABLE_FLOAT ON CACHE BOOL "" FORCE)
    set(ENABLE_DOC OFF CACHE BOOL "" FORCE)
    set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fftw3)
    set(FFTW3F_LIBRARIES fftw3f)
endif()

# Benchmark sources
set(BENCHMARK_SOURCES
    benchmark_fft.cpp
    benchmark_transforms.cpp
    benchmark_throughput.cpp
)

if(ENABLE_EXTERNAL_BENCHMARKS)
    list(APPEND BENCHMARK_SOURCES benchmark_comparison.cpp)
endif()

# Create benchmark executable
add_executable(fastandfourier_benchmarks ${BENCHMARK_SOURCES})

# Include directories
target_include_directories(fastandfourier_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

if(ENABLE_EXTERNAL_BENCHMARKS)
    target_include_directories(fastandfourier_benchmarks PRIVATE
        ${pocketfft_SOURCE_DIR}
        ${minfft_SOURCE_DIR}
        ${mufft_SOURCE_DIR}
    )
endif()

# Link libraries
target_link_libraries(fastandfourier_benchmarks
    fastandfourier
    benchmark::benchmark
    benchmark::benchmark_main
)

if(ENABLE_EXTERNAL_BENCHMARKS)
    target_link_libraries(fastandfourier_benchmarks kissfft-float)
    if(FFTW3F_FOUND)
        target_link_libraries(fastandfourier_benchmarks ${FFTW3F_LIBRARIES})
        target_compile_definitions(fastandfourier_benchmarks PRIVATE HAS_FFTW3=1)
    endif()
    if(FFTW3_LIBRARIES)
        target_link_libraries(fastandfourier_benchmarks ${FFTW3_LIBRARIES})
    endif()
endif()

# Link math library
if(UNIX)
    target_link_libraries(fastandfourier_benchmarks m)
endif()
