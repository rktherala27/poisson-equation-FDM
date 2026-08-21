#include <cmath>
#include <iostream>
#include <span>
#include <vector>
#include <stdexcept>

#include "utils/error_metrics.hpp"

int main(){

    constexpr double tolerance = 1.0e-12;

    const std::vector<double> numerical{1.0,3.0};
    const std::vector<double> analytical{0.0,0.0};

    const double error = ErrorMetrics::l2_error(
        std::span<const double>{numerical},
        std::span<const double>{analytical});

    const double expected = std::sqrt(10);

    if (std::abs(error-expected)>tolerance){
        std::cerr << "Failure: incorrect L2 error\n";
        return 1;
    }

    const double zero_error = ErrorMetrics::l2_error(
        std::span<const double>{numerical},
        std::span<const double>{numerical});

    if (std::abs(zero_error)> tolerance){
        std::cerr << "Failure: identical vectors must have zero error\n"; 
        return 1;
    }

    const std::vector<double> short_vector{1.0};

    try {
        static_cast<void>(ErrorMetrics::l2_error(
            std::span<const double>{numerical},
            std::span<const double>{short_vector}));

        std::cerr << "Failure: size mismatch did not throw\n";
        return 1;

    } catch (const std::invalid_argument&) {
        // std::cout<< "L2_Error: Test for short vector passed!"<<std::endl;
    }

    const std::vector<double> empty_vector{};

    try {
        static_cast<void>(ErrorMetrics::l2_error(
            std::span<const double>{empty_vector},
            std::span<const double>{empty_vector}));
        std::cerr << "Failure: empty vectors did not throw\n";
        return 1;
    } catch (const std::invalid_argument&) {
        // std::cout<< "L2_Error: Test for empty vector passed!"<<std::endl;
    }

    return 0;
}