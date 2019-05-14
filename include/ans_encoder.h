/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef ANS_ENCODER_
#define ANS_ENCODER_

#include "ans_encoder_table.h"
#include "ans_table_generator.h"
#include "cuhd_input_buffer.h"
#include "cuhd_definitions.h"
#include "cuhd_constants.h"

#include <memory>

struct Decoder_Info {
    UNIT_TYPE state;
    size_t bit;
    
    // encoded size in units
    size_t size;
};

class ANSEncoder {
    public:
        static std::shared_ptr<CUHDInputBuffer> encode(
            SYMBOL_TYPE* in,
            size_t size_in,
            std::shared_ptr<ANSEncoderTable> encoder_table);
    
    private:
        static void encode_memory(
            UNIT_TYPE* out,
            size_t size_out,
            SYMBOL_TYPE* in,
            size_t size_in,
            std::shared_ptr<ANSEncoderTable> encoder_table,
            std::shared_ptr<Decoder_Info> decoder_info);
};

#endif /* ANS_ENCODER_H_ */

