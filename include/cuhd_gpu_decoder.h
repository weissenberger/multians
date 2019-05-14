/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_DECODER_
#define CUHD_GPU_DECODER_

#include "cuhd_constants.h"
#include "cuhd_gpu_input_buffer.h"
#include "cuhd_gpu_output_buffer.h"
#include "cuhd_gpu_codetable.h"
#include "cuhd_gpu_decoder_memory.h"
#include "cuhd_subsequence_sync_point.h"

namespace cuhd {
    class CUHDGPUDecoder {
        public:
            static void decode(std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
                size_t input_size,
                std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
                size_t output_size,
                std::shared_ptr<cuhd::CUHDGPUCodetable> table,
                std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
                STATE_TYPE initial_state,
                std::uint32_t initial_bit,
                std::uint32_t number_of_states,
                size_t max_codeword_length,
                size_t preferred_subsequence_size,
                size_t threads_per_block);
    };
}

#endif /* CUHD_GPU_DECODER */

