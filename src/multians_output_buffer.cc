/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_output_buffer.h"

#include <algorithm>

CUHDOutputBuffer::CUHDOutputBuffer(size_t size) {
	uncompressed_size_ = size;

	// allocate buffer
	buffer_ = std::make_unique<SYMBOL_TYPE[]>(size);
}

std::shared_ptr<SYMBOL_TYPE[]>&
    CUHDOutputBuffer::get_decompressed_data() {
	return buffer_;	
}

void CUHDOutputBuffer::reverse() {
    std::reverse(buffer_.get(), buffer_.get() + uncompressed_size_);
}

size_t CUHDOutputBuffer::get_uncompressed_size() {
	return uncompressed_size_;
}

size_t CUHDOutputBuffer::get_symbol_size() {
	return symbol_size_;
}
