#pragma once

#include <chrono>

class Timer{

    std::chrono::steady_clock::time_point t0_;
public:
    void start(){
        t0_ = std::chrono::steady_clock::now();
    }

    [[nodiscard]] double elapsed_ms() const{
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1-t0_).count();
    }

    [[nodiscard]] double stop_ms(){
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0_).count();
        t0_ = t1;
        return ms;
    }
};