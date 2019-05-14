/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef ANS_TABLE_GENERATOR_
#define ANS_TABLE_GENERATOR_

#include "cuhd_constants.h"
#include "cuhd_definitions.h"
#include "cuhd_codetable.h"
#include "ans_encoder_table.h"

#include <memory>
#include <vector>
#include <functional>

struct Distribution {
    std::shared_ptr<std::vector<double>> prob;
    std::shared_ptr<std::vector<size_t>> dist;
    std::shared_ptr<std::vector<SYMBOL_TYPE>> symbols;
};

struct XS_pair {SYMBOL_TYPE s; std::uint32_t next_state;};

struct Queue_Entry {double p; SYMBOL_TYPE s;};

struct Encoder_Table_Entry {
    std::uint32_t x;
    std::uint32_t rem;
    std::uint32_t shift;
    SYMBOL_TYPE symbol;};

class ANSTableGenerator {
    public:
        static Distribution generate_distribution(
            size_t seed,
            size_t n,
            size_t N,
            std::function<double(double)> fun);
        
        static Distribution generate_distribution_from_buffer(
            size_t seed,
            size_t N,
            std::uint8_t* in,
            size_t size);
        
        static std::shared_ptr<std::vector<SYMBOL_TYPE>> generate_test_data(
            std::shared_ptr<std::vector<size_t>> distr,
            size_t size,
            size_t num_states,
            size_t seed);
        
        static std::shared_ptr<std::vector<std::vector<Encoder_Table_Entry>>>
            generate_table(
            std::shared_ptr<std::vector<double>> P_s,
            std::shared_ptr<std::vector<size_t>> L_s,
            std::shared_ptr<std::vector<SYMBOL_TYPE>> symbols,
            size_t num_symbols,
            size_t num_states);
            
        static std::shared_ptr<CUHDCodetable> get_decoder_table(
            std::shared_ptr<ANSEncoderTable> enc_table);
        
        static std::shared_ptr<ANSEncoderTable> generate_encoder_table(
            std::shared_ptr<std::vector<std::vector<Encoder_Table_Entry>>> tab);
        
        static size_t get_max_compressed_size(
            std::shared_ptr<ANSEncoderTable> encoder_table, size_t input_size);
};

#endif /* ANS_TABLE_GENERATOR_H_ */

