/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_decoder.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__device__ __forceinline__ void decode_subsequence(
    std::uint32_t subsequence_size,
    std::uint32_t current_subsequence,
    std::uint32_t subsequences_processed,
    UNIT_TYPE mask,
    std::uint32_t shift,
    std::uint32_t start_bit,
    std::uint32_t &in_pos,
    UNIT_TYPE* in_ptr,
    UNIT_TYPE &window,
    UNIT_TYPE &next,
    STATE_TYPE &state,
    std::uint32_t &last_word_unit,
    std::uint32_t &last_word_bit,
    std::uint32_t &num_symbols,
    std::uint32_t &out_pos,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t &next_out_pos,
    const uint* __restrict__ table,
    const std::uint32_t bits_in_unit,
    const std::uint32_t number_of_states,
    std::uint32_t &last_at,
    STATE_TYPE &last_state,
    bool overflow,
    bool write_output) {
    
    // current unit in this subsequence
    std::uint32_t current_unit = 0;
    
    // current bit position in unit
    std::uint32_t at = start_bit;

    // number of symbols found in this subsequence
    std::uint32_t num_symbols_l = 0;

    UNIT_TYPE copy_next = next;
    
    auto load_next = [&]() {
        window = in_ptr[in_pos];
        next = in_ptr[in_pos + 1];
        copy_next = next;
    };
    
    // shift to start
    if(current_subsequence == 0 || subsequences_processed == 0) {
        
        copy_next <<= bits_in_unit - at;

        next >>= at;
        window >>= at;
        window |= copy_next;
    }

    // perform overflow from previous subsequence
    if(overflow && current_subsequence > 0 && subsequences_processed == 0) {

        // decode first symbol
        const uint hit_p = table[state - number_of_states];
        const STATE_TYPE next_state_p = (std::uint16_t) (hit_p & 0x0000FFFF);
        std::uint32_t taken = hit_p >> 24;
        
        state = (next_state_p << taken) | (~(mask << taken) & window);
        
        while(state < number_of_states) {
            std::uint32_t shift_w = window >> taken;
            ++taken;
            state = (state << 1) | (~(mask << 1) & shift_w);
        }
        
        if(at == 0) {
            ++num_symbols_l;

            if(write_output) {
                if(out_pos < next_out_pos) {
                    out_ptr[out_pos] = (hit_p & ((std::uint32_t) 0x00FF0000)) >> 16;
                    ++out_pos;
                }
            }
        }
        
        if(taken > 0) {
            copy_next = next;
            copy_next <<= bits_in_unit - taken;
        }
        
        else copy_next = 0;

        next >>= taken;
        window >>= taken;
        at += taken;
        window |= copy_next;

        // overflow
        if(at > bits_in_unit) {
            ++in_pos;
            load_next();
            at -= bits_in_unit;
            window >>= at;
            next >>= at;

            copy_next <<= bits_in_unit - at;
            window |= copy_next;
        }
        
        if(at == bits_in_unit) {
            ++in_pos;
            load_next();
            at = 0;
        }
    }
    
    while(current_unit < subsequence_size) {
        
        while(at < bits_in_unit) {

            last_state = state;

            const uint hit = table[state - number_of_states];
            
            // decode a symbol
            const STATE_TYPE next_state = (std::uint16_t) (hit & 0x0000FFFF);
            std::uint32_t taken = hit >> 24;
            
            state = (next_state << taken) | (~(mask << taken) & window);
        
            while(state < number_of_states) {
                shift = window >> taken;
                ++taken;
                state = (state << 1) | (~(mask << 1) & shift);
            }
            
            ++num_symbols_l;

            if(write_output) {
                if(out_pos < next_out_pos) {
                    out_ptr[out_pos] = (hit & ((std::uint32_t) 0x00FF0000)) >> 16;
                    ++out_pos;
                }
            }
            
            if(taken > 0) {
                copy_next = next;
                copy_next <<= bits_in_unit - taken;
            }
            
            else copy_next = 0;

            next >>= taken;
            window >>= taken;
            last_word_bit = at;
            at += taken;
            window |= copy_next;
            last_word_unit = current_unit;
        }
        
        // refill decoder window if necessary
        ++current_unit;
        ++in_pos;
        
        load_next();
        
        if(at == bits_in_unit) {
            at = 0;
        }

        else {
            at -= bits_in_unit;
            window >>= at;
            next >>= at;
            
            copy_next <<= bits_in_unit - at;
            window |= copy_next;
        }
    }
    
    num_symbols = num_symbols_l;
    last_at = at;
}

__global__ void phase1_decode_subseq(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    UNIT_TYPE* in_ptr,
    const uint* __restrict__ table,
    uint4* sync_points,
    const std::uint32_t bits_in_unit,
    const std::uint32_t number_of_states,
    const STATE_TYPE initial_state,
    const std::uint32_t initial_bit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(gid * 4 < total_num_subsequences) {
        std::uint32_t current_subsequence = gid * 4;
        std::uint32_t in_pos = gid * subsequence_size * 4;
        
        // mask
        const UNIT_TYPE mask = (UNIT_TYPE) (0) - 1;

        // shift right
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint8_t* out_ptr = 0;
        
        // current state
        STATE_TYPE state = initial_state;
        
        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];

        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;
        
        // last state in this subsequence
        STATE_TYPE last_state;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        // bit position of next codeword
        std::uint32_t last_at = (gid == 0) ?
            bits_in_unit - initial_bit : 0;
        
        std::uint32_t subsequences_processed = 0;
        bool synchronised_flag = false;
        
        std::uint32_t last_subsequence = blockDim.x * (blockIdx.x + 1) * 4;
        if(last_subsequence > total_num_subsequences)
            last_subsequence = total_num_subsequences;
        
        auto sync = [&](std::uint32_t i) {
            if(subsequences_processed >= 4) {
                uint4 sync_point = sync_points[current_subsequence + i];

                if(sync_point.x == last_word_unit
                    && sync_point.y == last_word_bit
                    && sync_point.w == last_state) {
                        synchronised_flag = true;
                }
            }
        };
        
        uint4 s0, s1, s2, s3;
        bool wrt0 = false;
        bool wrt1 = false;
        bool wrt2 = false;
        bool wrt3 = false;
        
        while(subsequences_processed < blockDim.x * 4) {
            
            if(!synchronised_flag
                && current_subsequence < last_subsequence) {

                decode_subsequence(subsequence_size, current_subsequence,
                    subsequences_processed,
                    mask, shift, last_at, in_pos, in_ptr,
                    window, next,
                    state, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    number_of_states, last_at, last_state, false, false);
                
                sync(0);
                s0 = {last_word_unit, last_word_bit, num_symbols, last_state};
                wrt0 = true;
            }
            
            if(!synchronised_flag
                && current_subsequence < last_subsequence) {

                decode_subsequence(subsequence_size, current_subsequence + 1,
                    subsequences_processed + 1,
                    mask, shift, last_at, in_pos, in_ptr,
                    window, next,
                    state, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    number_of_states, last_at, last_state, false, false);
                
                sync(1);
                s1 = {last_word_unit, last_word_bit, num_symbols, last_state};
                wrt1 = true;
            }
            
            if(!synchronised_flag
                && current_subsequence < last_subsequence) {

                decode_subsequence(subsequence_size, current_subsequence + 2,
                    subsequences_processed + 2,
                    mask, shift, last_at, in_pos, in_ptr,
                    window, next,
                    state, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    number_of_states, last_at, last_state, false, false);
                
                sync(2);
                s2 = {last_word_unit, last_word_bit, num_symbols, last_state};
                wrt2 = true;
            }
            
            if(!synchronised_flag
                && current_subsequence < last_subsequence) {

                decode_subsequence(subsequence_size, current_subsequence + 3,
                    subsequences_processed + 3,
                    mask, shift, last_at, in_pos, in_ptr,
                    window, next,
                    state, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    number_of_states, last_at, last_state, false, false);
                
                sync(3);
                s3 = {last_word_unit, last_word_bit, num_symbols, last_state};
                wrt3 = true;
            }
                        
            if(wrt0) {
                sync_points[current_subsequence] = s0;
                wrt0 = false;
            }
            
            if(wrt1) {
                sync_points[current_subsequence + 1] = s1;
                wrt1 = false;
            }
            
            if(wrt2) {
                sync_points[current_subsequence + 2] = s2;
                wrt2 = false;
            }
            
            if(wrt3) {
                sync_points[current_subsequence + 3] = s3;
                wrt3 = false;
            }
            
            current_subsequence += 4;
            subsequences_processed += 4;
            
            __syncthreads();
        }
    }
}

__global__ void phase2_synchronise_blocks(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    std::uint32_t num_blocks,
    UNIT_TYPE* in_ptr,
    const uint* __restrict__ table,
    uint4* sync_points,
    SYMBOL_TYPE* block_synchronised,
    const std::uint32_t bits_in_unit,
    const std::uint32_t number_of_states,
    const STATE_TYPE initial_state) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t num_of_seams = num_blocks - 1;

    if(gid < num_of_seams) {
    
        // mask
        const UNIT_TYPE mask = (UNIT_TYPE) (0) - 1;

        // shift
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint8_t* out_ptr = 0;
        
        // jump to first sequence of the block
        std::uint32_t current_subsequence = (gid + 1) * blockDim.x;
        
        // search for synchronised sequences at the end of previous block
        uint4 sync_point = sync_points[current_subsequence - 1];
        
        // current unit
        std::uint32_t in_pos = (current_subsequence - 1) * subsequence_size;
        
        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = in_pos + sync_point.x;
        std::uint32_t last_word_bit = sync_point.y;
        
        // state
        STATE_TYPE state = sync_point.w;
        
        // last state in this subsequence
        STATE_TYPE last_state;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        std::uint32_t last_at = sync_point.y;
        
        in_pos += sync_point.x;
        
        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];
        
        std::uint32_t subsequences_processed = 0;
        bool synchronised_flag = false;

        while(subsequences_processed < blockDim.x) {
        
            if(!synchronised_flag) {
                decode_subsequence(subsequence_size, current_subsequence,
                    subsequences_processed,
                    mask, shift, last_at, in_pos, in_ptr, 
                    window, next, state,
                    last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    number_of_states, last_at, last_state, true, false);
                
                sync_point = sync_points[current_subsequence];
                
                // if sync point detected
                if(sync_point.x == last_word_unit
                    && sync_point.y == last_word_bit
                    && sync_point.w == last_state) {
                        sync_point.z = num_symbols;
                    
                        block_synchronised[gid + 1] = 1;
                        synchronised_flag = true;
                }
                
                // correct erroneous position data
                else {
                    sync_point.x = last_word_unit;
                    sync_point.y = last_word_bit;
                    sync_point.z = num_symbols;
                    sync_point.w = last_state;
                    
                    block_synchronised[gid + 1] = 0;
                }
                
                sync_points[current_subsequence] = sync_point;
            }
            
            ++current_subsequence;
            ++subsequences_processed;
            
            __syncthreads();
        }
    }
}

__global__ void phase3_copy_num_symbols_from_sync_points_to_aux(
    std::uint32_t total_num_subsequences,
    const uint4* __restrict__ sync_points,
    std::uint32_t* subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        subsequence_output_sizes[gid] = sync_points[gid].z;
    }
}

__global__ void phase3_copy_num_symbols_from_aux_to_sync_points(
    std::uint32_t total_num_subsequences,
    uint4* sync_points,
    const std::uint32_t* __restrict__ subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        sync_points[gid].z = subsequence_output_sizes[gid];
    }
}

__global__ void phase4_decode_write_output(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    UNIT_TYPE* in_ptr,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t output_size,
    const uint* __restrict__ table,
    const uint4* __restrict__ sync_points,
    const std::uint32_t bits_in_unit,
    const std::uint32_t number_of_states,
    const STATE_TYPE initial_state,
    const std::uint32_t initial_bit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        
        // mask
        const UNIT_TYPE mask = (UNIT_TYPE) (0) - 1;

        // shift
        const size_t shift = bits_in_unit - table_size;

        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;
        
        // state
        STATE_TYPE state = initial_state;
        
        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        // bit position of next codeword
        std::uint32_t last_at = 0;
        
        // last state in this subsequence
        STATE_TYPE last_state;
        
        std::uint32_t subsequences_processed = 0;
        
        std::uint32_t current_subsequence = gid;
        std::uint32_t in_pos = current_subsequence * subsequence_size;
        
        uint4 sync_point = sync_points[current_subsequence];
        uint4 next_sync_point = sync_points[current_subsequence + 1];
        
        std::uint32_t out_pos = sync_point.z;
        std::uint32_t next_out_pos = gid == total_num_subsequences - 1 ?
            output_size : next_sync_point.z;
        
        if(gid > 0) {
            sync_point = sync_points[current_subsequence - 1];
            in_pos = (current_subsequence - 1) * subsequence_size;
            state = sync_point.w;
        }
        
        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];
        
        // start bit
        std::uint32_t start = bits_in_unit - initial_bit;
        
        if(gid > 0) {
            in_pos += sync_point.x;
            start = sync_point.y;

            window = in_ptr[in_pos];
            next = in_ptr[in_pos + 1];
        }

        // overflow from previous subsequence, decode, write output
        decode_subsequence(subsequence_size, current_subsequence,
            subsequences_processed, mask, shift,
            start, in_pos, in_ptr, window, next, state,
            last_word_unit, last_word_bit, num_symbols, out_pos, out_ptr,
            next_out_pos, table, bits_in_unit, number_of_states,
            last_at, last_state, true, true);
    }
}

void cuhd::CUHDGPUDecoder::decode(
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
    size_t input_size,
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
    size_t output_size,
    std::shared_ptr<cuhd::CUHDGPUCodetable> table,
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
    STATE_TYPE initial_state,
    std::uint32_t initial_bit,
    std::uint32_t number_of_states,
    size_t max_codeword_length,
    size_t preferred_subsequence_size,
    size_t threads_per_block) {
    
    UNIT_TYPE* in_ptr = input->get();
    SYMBOL_TYPE* out_ptr = output->get();
    CUHDCodetableItem* table_ptr = table->get();
    
    uint4* sync_info = reinterpret_cast<uint4*>(aux->get_sync_info());
    std::uint32_t* output_sizes = aux->get_output_sizes();
    std::uint8_t* sequence_synced_device = aux->get_sequence_synced_device();
    std::uint8_t* sequence_synced_host = aux->get_sequence_synced_host();

    size_t num_subseq = SDIV(input_size, preferred_subsequence_size);
    size_t num_sequences = SDIV(num_subseq, threads_per_block);
    
    const std::uint32_t bits_in_unit = sizeof(UNIT_TYPE) * 8;

    // launch phase 1 (intra-sequence synchronisation)
    phase1_decode_subseq<<<num_sequences, threads_per_block>>>(
        preferred_subsequence_size,
        num_subseq,
        max_codeword_length,
        in_ptr,
        reinterpret_cast<uint*>(table_ptr),
        sync_info,
        bits_in_unit,
        number_of_states,
        initial_state,
        initial_bit);
    CUERR
    
    // launch phase 2 (inter-sequence synchronisation)
    bool blocks_synchronised = true;
    
    do {
        phase2_synchronise_blocks<<<num_sequences, threads_per_block>>>(
                preferred_subsequence_size,
                num_subseq,
                max_codeword_length,
                num_sequences,
                in_ptr,
                reinterpret_cast<uint*>(table_ptr),
                sync_info,
                sequence_synced_device,
                bits_in_unit,
                number_of_states,
                initial_state);
        CUERR

        aux->retrieve_sync_data();
        
        const size_t num_blocks = num_sequences - 1;
        
        bool zero_found = false;
        
        for(size_t i = 1; i < num_blocks; ++i) {
            if(sequence_synced_host[i] == 0) {
                zero_found = true;
                break;
            }
        }
        
        if(zero_found) {
            blocks_synchronised = false;
        }
        
        else {
            blocks_synchronised = true;
        }
        
    } while(!blocks_synchronised);
    
    // launch phase 3 (parallel prefix sum)
    thrust::device_ptr<std::uint32_t> thrust_sync_points(output_sizes);

    phase3_copy_num_symbols_from_sync_points_to_aux<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR

    thrust::exclusive_scan(thrust_sync_points,
        thrust_sync_points + num_subseq, thrust_sync_points);

    phase3_copy_num_symbols_from_aux_to_sync_points<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR
    
    // launch phase 4 (final decoding)
    phase4_decode_write_output<<<num_sequences, threads_per_block>>>(
            preferred_subsequence_size,
            num_subseq,
            max_codeword_length,
            in_ptr,
            out_ptr,
            output_size,
            reinterpret_cast<uint*>(table_ptr),
            sync_info,
            bits_in_unit,
            number_of_states,
            initial_state,
            initial_bit);
    CUERR
}

