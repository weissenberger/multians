/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_INPUT_BUFFER_
#define CUHD_INPUT_BUFFER_

#include <memory>

#include "cuhd_constants.h"
#include "cuhd_definitions.h"

class CUHDInputBuffer {
    public:
	    CUHDInputBuffer(UNIT_TYPE* buffer, size_t size,
	        size_t first_bit, size_t first_state);

	    // returns reference to compressed data
	    UNIT_TYPE* get_compressed_data();
	    
	    size_t get_first_bit();
	    size_t get_first_state();
	    size_t get_compressed_size();
	    size_t get_unit_size();

    private:
	    
	    // encoded data begins at this index
	    size_t first_bit_;
	    
	    // initial decoder state
	    size_t first_state_;
	    
	    // compressed size of input
	    size_t compressed_size_;

	    // size of a unit
	    const size_t unit_size_ = sizeof(UNIT_TYPE);
	
	    // buffer containing the compressed input
	    cuhd_buf(UNIT_TYPE, buffer_);
};

#endif /* CUHD_INPUT_BUFFER_H_ */

