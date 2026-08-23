#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/vector_ops.hpp"

int main(){

    constexpr double tolerance = 1e-12;

    const std::vector<double> x{1.0,2.0,3.0};
    const std::vector<double> z{4.0,5.0,6.0};

    const double exact_dot_result = 32.0;

    const double dot_result = dot<double>(std::span<const double>{x},
                                  std::span<const double>{z});
            
    if((std::abs(dot_result)-exact_dot_result)>tolerance){
        std::cerr <<"Failure: incorrect vector norm\n";
        return 1;
    }

    const double norm_result = euclidean_norm<double>(
        std::span<const double>{x});

    if (std::abs(norm_result - std::sqrt(14.0)) > tolerance) {
        std::cerr << "Failure: incorrect euclidean norm\n";
        return 1;
    }

    

    std::vector<double> y{4.0, 5.0, 6.0};

    axpy<double>(
        std::span<const double>{x},
        std::span<double>{y},
        2.0,
        -1.0);

    const std::vector<double> expected_y{-2.0, -1.0, 0.0};

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (std::abs(y[i] - expected_y[i]) > tolerance) {
            std::cerr << "Failure: incorrect axpy result\n";
            return 1;
        }
    }

    const std::vector<double> short_vector{1.0,2.0};

    try{
        static_cast<void>(dot(std::span<const double>{x},
                              std::span<const double>{short_vector}));
        std::cerr<<"Failure: dot product size mismatch did not throw\n";
        return 1;
    }catch(const std::invalid_argument&){    
    }

    try {
        std::vector<double> short_y{4.0, 5.0};

        axpy<double>(
            std::span<const double>{x},
            std::span<double>{short_y},
            1.0,
            1.0);

        std::cerr << "Failure: axpy size mismatch did not throw\n";
        return 1;

    } catch (const std::invalid_argument&) {
    }
    
    return 0;

}

