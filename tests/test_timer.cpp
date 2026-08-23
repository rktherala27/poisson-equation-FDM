#include <cstddef>
#include <iostream>
#include <thread>
#include <chrono>

#include "utils/timer.hpp"


int main(){

    Timer timer;

    timer.start();

    std::this_thread::sleep_for(std::chrono::milliseconds{5});

    const Timer& read_only_timer = timer;
    const double elapsed = read_only_timer.elapsed_ms();

    if (elapsed <= 0.0){
        std::cerr << "Failure: elapsed time should be positive\n";
        return 1;
    }

    const double stopped = timer.stop_ms();

    if (stopped <= 0.0) {
        std::cerr << "Failure: stopped time should be positive\n";
        return 1;
    }

    return 0;
}