/**
 * @file test_main.cpp
 * @brief Main entry point for tests
 */

#include <gtest/gtest.h>
#include "fastandfourier.h"

int main(int argc, char **argv) {
    /* Initialize library */
    faf_init();
    
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    /* Cleanup */
    faf_cleanup();
    
    return result;
}
