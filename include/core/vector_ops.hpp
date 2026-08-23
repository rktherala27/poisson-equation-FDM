#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <cmath>
#include <concepts>
#include <stdexcept>

template<std::floating_point Real>
[[nodiscard]] double dot(std::span<const Real> a,
                         std::span<const Real> b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("dot: size mismatch");

    double result = 0.0;
    for(std::size_t i =0; i<a.size(); ++i){
        result += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return result;
}

template<std::floating_point Real>
[[nodiscard]] double euclidean_norm(std::span<const Real> x)
{
    return std::sqrt(dot<Real>(x,x));
}

template<std::floating_point Real>
void axpy(std::span<const Real> x,
                        std::span<Real> y,
                        double a1,
                        double a2)
{
    if(x.size() != y.size()){
        throw std::invalid_argument("axpy: size mismatch");
    }

    for (std::size_t i =0; i<x.size(); ++i){
        y[i] = static_cast<Real>(a1* static_cast<double>(x[i]) + a2*static_cast<double>(y[i]));
    }
}