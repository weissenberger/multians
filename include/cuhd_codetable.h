/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_CODETABLE_
#define CUHD_CODETABLE_

#include "cuhd_constants.h"
#include "cuhd_definitions.h"

#include <memory>
#include <cstring>

struct CUHDCodetableItem {
    std::uint16_t next_state;
    std::uint8_t symbol;
    std::uint8_t min_num_bits;
};

class CUHDCodetable {
    public:
        CUHDCodetable(size_t num_entries);

        size_t get_size();
        size_t get_num_entries();
        size_t get_max_codeword_length();
        
        CUHDCodetableItem* get();
        
    private:
        
        // total number of rows
        size_t size_;

        // actual number of items
        size_t num_entries_;
        
        cuhd_buf(CUHDCodetableItem, table_);
};

#endif /* CUHD_CODETABLE_H_ */

