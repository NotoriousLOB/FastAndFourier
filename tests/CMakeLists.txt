# Tests CMakeLists.txt

# Find or fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Test sources
set(TEST_SOURCES
    test_main.cpp
    test_core.cpp
    test_vm.cpp
    test_transforms.cpp
    test_twiddles.cpp
    test_precision.cpp
    test_jit.cpp
    test_chirp.cpp
    test_chirp_compositions.cpp
    test_wavelets.cpp
    test_rfft.cpp
    test_spectral.cpp
    test_layout.cpp
    test_mixed_radix.cpp
)

# Architecture-specific tests
if(ARCH_X86_64)
    list(APPEND TEST_SOURCES test_x86.cpp)
endif()

if(ARCH_AARCH64)
    list(APPEND TEST_SOURCES test_arm.cpp)
endif()

if(CUDA_FOUND)
    list(APPEND TEST_SOURCES test_cuda.cpp)
endif()

# Create test executable
add_executable(fastandfourier_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(fastandfourier_tests
    fastandfourier
    GTest::gtest
    GTest::gtest_main
    pthread
)

if(UNIX)
    target_link_libraries(fastandfourier_tests m)
endif()

# Coverage flags
if(ENABLE_COVERAGE)
    target_compile_options(fastandfourier_tests PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(fastandfourier_tests PRIVATE --coverage)
endif()

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(fastandfourier_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Add test
add_test(NAME FastAndFourierTests COMMAND fastandfourier_tests)

# Coverage target
if(ENABLE_COVERAGE)
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND gcovr -r ${CMAKE_SOURCE_DIR} --html --html-details 
                -o ${CMAKE_BINARY_DIR}/coverage/index.html
        COMMAND gcovr -r ${CMAKE_SOURCE_DIR} --txt
        DEPENDS fastandfourier_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )
endif()
