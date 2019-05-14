/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_DEFINITIONS_
#define CUHD_DEFINITIONS_

#define cuhd_buf(TYPE, IDENTIFIER) std::shared_ptr<TYPE[]> IDENTIFIER
#define cuhd_out_buf(TYPE, IDENTIFIER) std::shared_ptr<TYPE[]> IDENTIFIER

#endif /* CUHD_DEFINITIONS_H_ */
