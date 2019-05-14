/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_codetable.h"

CUHDCodetable::CUHDCodetable(size_t num_entries)
    : size_(num_entries),
      num_entries_(num_entries) {
      
      std::shared_ptr<CUHDCodetableItem[]> table(
        new CUHDCodetableItem[get_size()]);
     
     std::memset(table.get(), 0, get_size() * sizeof(CUHDCodetableItem));
     
     table_ = table;
}

size_t CUHDCodetable::get_size() {
    return size_;
}

size_t CUHDCodetable::get_num_entries() {
    return num_entries_;
}

size_t CUHDCodetable::get_max_codeword_length() {
    return MAX_CODEWORD_LENGTH;
}

CUHDCodetableItem* CUHDCodetable::get() {
    return table_.get();
}

