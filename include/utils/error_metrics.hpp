#pragma once

#include <cstddef>
#include <span>
#include <cmath>
#include <concepts>
#include <stdexcept>

namespace ErrorMetrics {

    template<std::floating_point Real>
    [[nodiscard]] double l2_error(std::span<const Real> numerical,
                    std::span<const Real> analytical){

            if (numerical.size() != analytical.size()){
                throw std::invalid_argument("ErrorMetrics::l2_error: size mismatch");
            }
            else if(numerical.size() == 0){
                throw std::invalid_argument("ErrorMetrics::l2_error: size zero");
            }
            
            double result = 0.0;
            for(std::size_t i = 0; i<numerical.size();++i){
                double diff = (static_cast<double>(numerical[i]) - static_cast<double>(analytical[i]));
                result += diff*diff;
            }

            return std::sqrt(result);
    }
}

