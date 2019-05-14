/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_CONSTANTS_
#define CUHD_CONSTANTS_

// maximum codeword length this implementation can process
#define MAX_CODEWORD_LENGTH 11

// data type of a unit
#define UNIT_TYPE std::uint32_t

// state register type
#define STATE_TYPE std::uint32_t

// data type of a symbol
#define SYMBOL_TYPE std::uint8_t

// data type for storing the bit length of codewords
#define BIT_COUNT_TYPE std::uint8_t

#endif /* CUHD_CONSTANTS_H_ */

