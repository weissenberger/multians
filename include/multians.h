/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_constants.h"
#include "cuhd_codetable.h"
#include "cuhd_input_buffer.h"
#include "cuhd_output_buffer.h"
#include "cuhd_util.h"
#include "ans_encoder_table.h"
#include "ans_table_generator.h"
#include "ans_encoder.h"

#ifdef CUDA
#include "cuhd_gpu_codetable.h"
#include "cuhd_gpu_input_buffer.h"
#include "cuhd_gpu_output_buffer.h"
#include "cuhd_gpu_decoder_memory.h"
#include "cuhd_gpu_decoder.h"
#endif

#ifdef MULTI
#include "multicore_decoder.h"
#endif
