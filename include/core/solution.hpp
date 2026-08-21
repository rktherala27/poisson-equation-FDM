#pragma once

#include <cstddef>
#include <vector>

template<typename Real>
struct Solution{
    std::vector<Real> x;
    std::size_t iterations;
    double  final_residual = 0.0;
    double  initial_residual = 0.0;

    explicit Solution (const std::size_t n)
        : x(n, Real{0}),
        iterations{0},
        final_residual{0.0},
        initial_residual{0.0} {}
};
