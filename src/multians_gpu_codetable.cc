/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_codetable.h"

cuhd::CUHDGPUCodetable::CUHDGPUCodetable(
    std::shared_ptr<CUHDCodetable> codetable)
    : CUHDGPUMemoryBuffer<CUHDCodetableItem>(codetable->get(),
        codetable->get_size()),
        table_(codetable) {
      
}

