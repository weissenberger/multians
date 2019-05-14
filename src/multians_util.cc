/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_util.h"

#include <iostream>

#ifdef CUDA
#include <cuda_runtime_api.h>
#endif


std::pair<std::string, size_t> cuhd::CUHDUtil::time(std::string s,
    std::function<void()> f) {
	std::chrono::high_resolution_clock clock;
	std::chrono::nanoseconds duration;

	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
    
    #ifdef CUDA
    cudaEvent_t startcu, stopcu;
    cudaEventCreate(&startcu);
    cudaEventCreate(&stopcu);
    #endif

    start = clock.now();
    
    #ifdef CUDA
    cudaEventRecord(startcu);
    #endif

    f();
    
    #ifdef CUDA
    cudaEventRecord(stopcu);
    #endif
    
    end = clock.now(); 

    duration = end - start;

    std::pair<std::string, size_t> p(s,
        std::chrono::duration_cast<std::chrono::microseconds>
            (duration).count());
        
    #ifdef CUDA
    cudaEventSynchronize(stopcu);
    float millisec = 0.0f;
    cudaEventElapsedTime(&millisec, startcu, stopcu);
    
    return std::pair<std::string, size_t>(s, (size_t) millisec * 1000.0f);
    #endif

    return p;
}

bool cuhd::CUHDUtil::equals(SYMBOL_TYPE* a, SYMBOL_TYPE* b, size_t size) {
    for(size_t i = 0; i < size; ++i) {
        if(a[i] != b[i]) {
        std::cout << "mismatch at: " << i << std::endl;
        return false;}}

    return true;
}

size_t cuhd::CUHDUtil::optimal_subsequence_size(size_t input_size, 
    size_t output_size, size_t pref, size_t device_pref) {
    
    float device = 1.0f / device_pref;
    size_t curr_size = pref;
    
    while(curr_size > 1 && (float) SDIV(SDIV(input_size, 4), curr_size) /
        (float) SDIV(output_size, 4) < device) {
        curr_size = SDIV(curr_size, 2);
    }
    
    return curr_size;
}

