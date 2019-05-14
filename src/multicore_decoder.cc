/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "multicore_decoder.h"

#include <memory>
#include <cassert>
#include <thread>

void MulticoreDecoder::decode(
    size_t subsequence_size,
    size_t num_threads,
    size_t input_size_units,
    std::shared_ptr<CUHDOutputBuffer> out,
    std::shared_ptr<CUHDInputBuffer> in,
    std::shared_ptr<CUHDCodetable> tab) {
    
    // split units into subsequences
    size_t num_subsequences = input_size_units / subsequence_size;
    size_t num_remaining = input_size_units % subsequence_size;
    if(num_remaining != 0) ++num_subsequences;
    
    // spread subsequences over multiple threads
    std::vector<DecoderInterval> intervals = get_decoder_intervals(
        subsequence_size, num_threads, input_size_units);
    
    // create array to track synchronization points
    std::shared_ptr<SubsequenceSyncPoint[]> sync_info(
        new SubsequenceSyncPoint[num_subsequences]);
    
    std::vector<std::thread> threads(num_threads);
    std::shared_ptr<std::vector<size_t>> thread_synced(
        new std::vector<size_t>(num_threads, false));
    std::shared_ptr<size_t[]> out_positions(new size_t[num_threads]);
    
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(decode_phase1, i,
            intervals.at(i).begin, intervals.at(i).end, intervals.at(i).sub,
            subsequence_size, input_size_units, num_threads,
            out_positions, out, in, tab, sync_info, thread_synced,
            false, false);
    }
    
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    
    bool synchronized = false;
    
    while(!synchronized) {
        for(size_t i = 1; i < num_threads; ++i) {
            threads[i] = std::thread(decode_phase1, i,
                intervals.at(i).begin, intervals.at(i).end, intervals.at(i).sub,
                subsequence_size, input_size_units, num_threads,
                out_positions, out, in, tab, sync_info, thread_synced,
                true, false);
        }
        
        for(size_t i = 1; i < num_threads; ++i) {
            threads[i].join();
        }
        
        synchronized = true;
        
        for(size_t i = 1; i < num_threads; ++i) {
            if(!thread_synced->at(i)) synchronized = false;
        }
    }
    
    prefix_sum(sync_info, out_positions, num_subsequences, num_threads);
    
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(decode_phase1, i,
            intervals.at(i).begin, intervals.at(i).end, intervals.at(i).sub,
            subsequence_size, input_size_units, num_threads,
            out_positions, out, in, tab, sync_info, thread_synced,
            false, true);
    }
    
    for(size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
}

void MulticoreDecoder::decode_phase1(
    size_t thread_id,
    size_t begin,
    size_t end,
    size_t subsequence,
    size_t subsequence_size,
    size_t num_units,
    size_t num_threads,
    std::shared_ptr<size_t[]> out_positions,
    std::shared_ptr<CUHDOutputBuffer> out,
    std::shared_ptr<CUHDInputBuffer> in,
    std::shared_ptr<CUHDCodetable> tab,
    std::shared_ptr<SubsequenceSyncPoint[]> sync_info,
    std::shared_ptr<std::vector<size_t>> thread_synced,
    bool overflow,
    bool write) {

    if(overflow) {
        if(thread_synced->at(thread_id) == true) return;
    }
    
    SYMBOL_TYPE* out_ptr = out->get_decompressed_data().get();
    const size_t size_out = out->get_uncompressed_size();
    
    UNIT_TYPE* in_ptr = in->get_compressed_data();
    
    const CUHDCodetableItem* table = tab->get();
    
    SubsequenceSyncPoint* sync = sync_info.get();
    
    const size_t number_of_states = tab->get_num_entries();
    const size_t bits_in_unit = in->get_unit_size() * 8;

    UNIT_TYPE current_state = in->get_first_state();
    
    std::uint8_t at = (thread_id == 0) ? bits_in_unit - in->get_first_bit()
        : 0;
    
    size_t in_pos = begin;
    size_t out_pos = 0;
    size_t out_size = 0;
    size_t current_subsequence = subsequence;
    std::uint32_t current_unit = 0;
    
    if(overflow || (write && thread_id > 0)) {
        SubsequenceSyncPoint sp = sync[current_subsequence - 1];
        current_state = sp.state;
        at = sp.bit;
        in_pos -= subsequence_size;
        in_pos += sp.unit;
        current_unit = sp.unit;
    }
    
    if(write) {
        SubsequenceSyncPoint sp = sync[current_subsequence];
        out_pos = out_positions.get()[thread_id];
        
        if(thread_id < num_threads - 1) {
            sp = sync[current_subsequence + 1];
            out_size = out_positions.get()[thread_id + 1];
        }
        
        else out_size = size_out;
    }
    
    UNIT_TYPE window = in_ptr[in_pos];
    UNIT_TYPE next = in_ptr[in_pos + 1];
    const UNIT_TYPE mask = (UNIT_TYPE) (0) - 1;

    UNIT_TYPE last_state = 0;
    std::uint32_t last_bit = 0;
    bool reset = false;
    if(write && thread_id == 0) reset = true;
    
    // shift to start
    UNIT_TYPE copy_next = next;
    copy_next <<= bits_in_unit - at;

    next >>= at;
    window >>= at;
    window += copy_next;
    
    std::uint32_t num_symbols = 0;
    
    while(in_pos < end) {
        while(at < bits_in_unit) {
        
            last_state = current_state;

            const CUHDCodetableItem hit
                = table[current_state - number_of_states];
            
            const STATE_TYPE next_state = hit.next_state;
            
            // decode a symbol
            size_t taken = hit.min_num_bits;
            ++num_symbols;
            
            UNIT_TYPE reversed = ~(mask << taken) & window;
            current_state = (next_state << taken) + reversed;
            
            while(current_state < number_of_states) {
                const UNIT_TYPE shift = window >> taken;
                ++taken;
                current_state = (current_state << 1) + (~(mask << 1) & shift);
            }

            if(write && reset && out_pos < out_size) {
                out_ptr[out_pos] = hit.symbol;
                ++out_pos;
            }
            
            if(taken > 0) {
                copy_next = next;
                copy_next <<= bits_in_unit - taken;
            }
            
            else copy_next = 0;
            
            last_bit = at;

            next >>= taken;
            window >>= taken;
            at += taken;
            window += copy_next;
        }
        
        // refill decoder window if necessary
        ++in_pos;
        ++current_unit;
        
        if(current_unit == subsequence_size) {
            if(overflow && reset) {
                SubsequenceSyncPoint sp = sync[current_subsequence];
                
                if(sp.state == last_state
                    && sp.bit == last_bit
                    && sp.unit == current_unit - 1) {

                    sync[current_subsequence].num_symbols = num_symbols;
                    thread_synced->at(thread_id) = true;

                    return;
                }
            }

            if(!overflow || reset) {
                if(!write) {
                    sync[current_subsequence] = {last_state, last_bit,
                        current_unit - 1, num_symbols};
                }    
                
                ++current_subsequence;
            }
            
            if(overflow && in_pos > num_units)
                thread_synced->at(thread_id) = true;
            
            reset = true;
            
            current_unit = 0;
            num_symbols = 0;
        }
        
        window = in_ptr[in_pos];
        next = in_ptr[in_pos + 1];
        
        if(at == bits_in_unit) {
            at = 0;
        }

        else {
            at -= bits_in_unit;
            window >>= at;
            next >>= at;
            
            UNIT_TYPE copy_next = in_ptr[in_pos + 1];
            copy_next <<= bits_in_unit - at;
            window += copy_next;
        }
    }
}

std::vector<DecoderInterval> MulticoreDecoder::get_decoder_intervals(
    size_t subsequence_size,
    size_t num_threads,
    size_t input_size_units) {
    
    size_t num_subsequences = input_size_units / subsequence_size;
    size_t num_remaining = input_size_units % subsequence_size;
    
    if(num_remaining != 0) ++num_subsequences;
    assert(num_subsequences >= num_threads);
    
    if(num_subsequences == 1) num_remaining = 0;
    
    size_t subs_per_thread = num_subsequences / num_threads;
    size_t remaining_subs = num_subsequences % num_threads;
    
    std::vector<DecoderInterval> vals(num_threads);
    
    size_t at = 0;
    for(size_t i = 0; i < num_threads; ++i) {
        vals.at(i) = {at, subs_per_thread * subsequence_size + at,
            subs_per_thread * i};
        at += subs_per_thread * subsequence_size;
    }
    
    vals.at(num_threads - 1).end += remaining_subs * subsequence_size;
    
    return vals;
}

void MulticoreDecoder::prefix_sum(
    std::shared_ptr<SubsequenceSyncPoint[]> sync_info,
    std::shared_ptr<size_t[]> out_positions,
    size_t num_subsequences,
    size_t num_threads) {
    
    size_t subs_per_thread = num_subsequences / num_threads;
    size_t pos = 0;
    size_t next = subs_per_thread;
    
    std::uint32_t sum = 0;
    SubsequenceSyncPoint* in_ptr = sync_info.get();
    size_t* out_ptr = out_positions.get();
    
    out_ptr[0] = 0;
    
    for(size_t i = 1; i < num_threads; ++i) {        
        
        for(; pos < next; ++pos) {
            std::uint32_t num = in_ptr[pos].num_symbols;
            sum += num;
        }
        
        out_ptr[i] = sum;
        
        pos = next;
        next += subs_per_thread;
    }
}

