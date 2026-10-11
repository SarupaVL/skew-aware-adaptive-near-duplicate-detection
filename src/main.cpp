#include <iostream>
#include <omp.h>

// Returns a human-readable description of the C++ standard
// based on the standard __cplusplus preprocessor macro.
const char* get_cpp_standard_name(long standard_value) {
    if (standard_value >= 202002L) {
        return "C++20 or later";
    }
    if (standard_value >= 201703L) {
        return "C++17";
    }
    if (standard_value >= 201402L) {
        return "C++14";
    }
    if (standard_value >= 201103L) {
        return "C++11";
    }
    return "Pre-C++11";
}

int main() {
    std::cout << "===============================================================\n";
    std::cout << " Skew-Aware Adaptive Parallel Near-Duplicate Detection\n";
    std::cout << " Phase 0: Environment & OpenMP Verification\n";
    std::cout << "===============================================================\n\n";

    // 1. Report C++ Standard
    std::cout << "[C++ Standard Check]\n";
    std::cout << "  __cplusplus macro value : " << __cplusplus << "\n";
    std::cout << "  Active language standard: " << get_cpp_standard_name(__cplusplus) << "\n\n";

    // 2. Report Available Execution Resources via OpenMP API
    int max_available_threads = omp_get_max_threads();
    int processor_count = omp_get_num_procs();
    std::cout << "[System Resource Discovery]\n";
    std::cout << "  Logical processors detected : " << processor_count << "\n";
    std::cout << "  Default max threads pool    : " << max_available_threads << "\n\n";

    // 3. Execute an OpenMP Parallel Region
    std::cout << "[OpenMP Team Verification]\n";
    int actual_team_size = 0;

    #pragma omp parallel
    {
        // Only one thread records and reports the team size
        #pragma omp single
        {
            actual_team_size = omp_get_num_threads();
            std::cout << "  Entered OpenMP parallel region.\n";
            std::cout << "  Threads in active parallel team: " << actual_team_size << "\n";
        }
    }

    std::cout << "\n[Status] Phase 0 verification completed successfully.\n";
    return 0;
}
