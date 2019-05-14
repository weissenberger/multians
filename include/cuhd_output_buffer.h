/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_OUTPUT_BUFFER_
#define CUHD_OUTPUT_BUFFER_

#include "cuhd_constants.h"
#include "cuhd_definitions.h"

#include <memory>

class CUHDOutputBuffer {
    public:
	    CUHDOutputBuffer(size_t size);
	
	    // returns reference to uncompressed data
	    std::shared_ptr<SYMBOL_TYPE[]>& get_decompressed_data();

        void reverse();
	    size_t get_uncompressed_size();
	    size_t get_symbol_size();

    private:
	    // total size of decompressed output in symbols
	    size_t uncompressed_size_;

	    // size of a symbol in bytes
	    const size_t symbol_size_ = sizeof(SYMBOL_TYPE);

	    // buffer containing the decompressed output
	    cuhd_out_buf(SYMBOL_TYPE, buffer_);
};

#endif /* CUHD_OUTPUT_BUFFER_H_ */

