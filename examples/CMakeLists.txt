# Examples CMakeLists.txt

set(EXAMPLES
    que_onda_mundo
    simple_fft
    simple_rfft
    dct_example
    wavelet_example
    jit_example
    chirp_example
    chirp_builtins_example
    chirp_wavelet_dsl
    chirp_dwt_families
    chirp_dwt_denoise
    chirp_dwt_jpeg2000
    chirp_dwt_transients
    chirp_dwt_ecg
    spectral_notch
    fft_convolve
    matched_filter
    analytic_envelope
    cwt_filterbank
    cwt_reconstruct
)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.c)
    target_link_libraries(${example} fastandfourier)
endforeach()
