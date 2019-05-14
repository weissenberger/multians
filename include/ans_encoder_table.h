/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef ANS_ENCODER_TABLE_
#define ANS_ENCODER_TABLE_

#include "cuhd_constants.h"

#include <memory>
#include <vector>

struct ANSEncoderTable {
    struct ANSEncoderTableItem {
        
        // next state index
        std::uint32_t next_state;
        
        // code sequence to write to output
        std::uint32_t code_sequence;
        
        // length of code sequence
        std::uint32_t code_length;
        
        SYMBOL_TYPE symbol;
    };
    
    // maximum number of symbols
    size_t max_number_of_symbols;
    
    // number of symbols
    size_t number_of_symbols;
    
    // number of ANS states
    size_t number_of_states;
    
    // the table
    std::vector<std::vector<ANSEncoderTableItem>> table;
};

#endif /* ANS_ENCODER_TABLE */
