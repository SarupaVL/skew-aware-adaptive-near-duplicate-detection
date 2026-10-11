cmake_minimum_required(VERSION 3.16)

project(SkewAwareNearDuplicateDetection
    VERSION 0.1.0
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

# Define the verification executable
add_executable(hpc_near_duplicate
    src/main.cpp
)

# Link OpenMP runtime target
target_link_libraries(hpc_near_duplicate
    PRIVATE
        OpenMP::OpenMP_CXX
)
