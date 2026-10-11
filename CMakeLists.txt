cmake_minimum_required(VERSION 3.16)

project(SkewAwareNearDuplicateDetection
    VERSION 0.3.0
    LANGUAGES CXX
)

# Enforce C++17 standard strictly across all compilation units
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate OpenMP package; explicitly fail with a clear error if not found
find_package(OpenMP REQUIRED)

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR 
        "OpenMP C++ support was not found for the selected compiler. "
        "This project strictly requires OpenMP for parallel execution. "
        "Please ensure an OpenMP-compatible toolchain is installed."
    )
endif()

# Common include directories
include_directories(include)

# Reusable core library (data model, dataset generator, SimHash, candidate generation)
add_library(hpc_core STATIC
    src/dataset_generator.cpp
    src/simhash.cpp
    src/candidate_gen.cpp
)

target_include_directories(hpc_core PUBLIC
    include
)

# Main executable
add_executable(hpc_near_duplicate
    src/main.cpp
)

target_link_libraries(hpc_near_duplicate
    PRIVATE
        hpc_core
        OpenMP::OpenMP_CXX
)

# Automated testing
enable_testing()

# Test 1: Dataset Generator Test Suite (Phase 1)
add_executable(test_dataset_generator
    tests/test_generator.cpp
)

target_link_libraries(test_dataset_generator
    PRIVATE
        hpc_core
)

add_test(NAME DatasetGeneratorTest COMMAND test_dataset_generator)

# Test 2: SimHash & Hamming Distance Test Suite (Phase 2)
add_executable(test_simhash
    tests/test_simhash.cpp
)

target_link_libraries(test_simhash
    PRIVATE
        hpc_core
)

add_test(NAME SimHashTest COMMAND test_simhash)

# Test 3: Sequential Candidate Generation & Verification Pipeline (Phase 3)
add_executable(test_candidate_gen
    tests/test_candidate_gen.cpp
)

target_link_libraries(test_candidate_gen
    PRIVATE
        hpc_core
)

add_test(NAME CandidateGenTest COMMAND test_candidate_gen)
