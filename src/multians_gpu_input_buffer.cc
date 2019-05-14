/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_input_buffer.h"

cuhd::CUHDGPUInputBuffer::CUHDGPUInputBuffer(
    std::shared_ptr<CUHDInputBuffer> input_buffer)
    : CUHDGPUMemoryBuffer<UNIT_TYPE>(input_buffer->get_compressed_data(),
        input_buffer->get_compressed_size() + 4),
        input_buffer_(input_buffer) {
    
}

