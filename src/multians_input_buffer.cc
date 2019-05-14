/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_input_buffer.h"

#include <algorithm>

CUHDInputBuffer::CUHDInputBuffer(UNIT_TYPE* buffer, size_t size,
    size_t first_bit, size_t first_state)
    : first_bit_(first_bit),
    first_state_(first_state),
    compressed_size_(size) {
      
	// allocate buffer
	// avoid invalid read at end of input during decoding
	buffer_ = std::make_unique<UNIT_TYPE[]>(compressed_size_ + 4);
    
	// pad unused bytes at the end of the buffer with zeroes
	buffer_.get()[compressed_size_ - 1] = 0;

	// copy compressed data into buffer and reverse order of units
	std::reverse_copy(buffer, buffer + size, buffer_.get());
}

UNIT_TYPE* CUHDInputBuffer::get_compressed_data() {
	return buffer_.get();
}

size_t CUHDInputBuffer::get_first_bit() {
    return first_bit_;
}

size_t CUHDInputBuffer::get_first_state() {
    return first_state_;
}

size_t CUHDInputBuffer::get_compressed_size() {
	return compressed_size_;
}

size_t CUHDInputBuffer::get_unit_size() {
	return unit_size_;	
}
