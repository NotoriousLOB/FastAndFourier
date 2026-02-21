# Examples CMakeLists.txt

set(EXAMPLES
    simple_fft
    dct_example
    wavelet_example
    jit_example
    chirp_example
)

foreach(example ${EXAMPLES})
    add_executable(${example} ${example}.c)
    target_link_libraries(${example} fastandfourier)
endforeach()
