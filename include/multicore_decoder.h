/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_constants.h"
#include "cuhd_codetable.h"
#include "cuhd_input_buffer.h"
#include "cuhd_output_buffer.h"
#include "cuhd_util.h"
#include "ans_encoder_table.h"

#include <functional>
#include <memory>

#ifndef MULTICORE_DECODER_
#define MULTICORE_DECODER_

struct SubsequenceSyncPoint {
    UNIT_TYPE state;
    std::uint32_t bit;
    std::uint32_t unit;
    std::uint32_t num_symbols;
};

struct DecoderInterval {
    size_t begin;
    size_t end;
    size_t sub;
};

class MulticoreDecoder {
    public:
        static void decode(
            size_t subsequence_size,
            size_t num_threads,
            size_t input_size_units,
            std::shared_ptr<CUHDOutputBuffer> out,
            std::shared_ptr<CUHDInputBuffer> in,
            std::shared_ptr<CUHDCodetable> tab);
    
    private:
        static std::vector<DecoderInterval> get_decoder_intervals(
            size_t subsequence_size,
            size_t num_threads,
            size_t input_size_units);
    
        static void decode_phase1(
            size_t thread_id,
            size_t begin,
            size_t end,
            size_t subsequence,
            size_t subsequence_size,
            size_t num_units,
            size_t num_threads,
            std::shared_ptr<size_t[]> out_positions,
            std::shared_ptr<CUHDOutputBuffer> out,
            std::shared_ptr<CUHDInputBuffer> in,
            std::shared_ptr<CUHDCodetable> tab,
            std::shared_ptr<SubsequenceSyncPoint[]> sync_info,
            std::shared_ptr<std::vector<size_t>> thread_synced,
            bool overflow,
            bool write);
            
        static void prefix_sum(
            std::shared_ptr<SubsequenceSyncPoint[]> sync_info,
            std::shared_ptr<size_t[]> out_positions,
            size_t num_subsequences,
            size_t num_threads);
};

#endif /* MULTICORE_DECODER_H_ */
