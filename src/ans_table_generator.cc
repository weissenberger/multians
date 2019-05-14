/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include "ans_table_generator.h"

#include <unordered_map>
#include <random>
#include <algorithm>
#include <cassert>

Distribution ANSTableGenerator::generate_distribution(
    size_t seed, size_t n, size_t N, std::function<double(double)> fun) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::vector<double> prob(n);
    std::vector<double> prob_ans(n);
    std::vector<size_t> distr(n);

    std::iota(prob.begin(), prob.end(), 0);
    std::transform(prob.begin(), prob.end(), prob.begin(), fun);
    std::transform(prob.begin(), prob.end(), prob.begin(), 
        [&](double x) {if(x < 0.001) return x += 0.001; else return x;});
    
    double sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
    std::transform(prob.begin(), prob.end(), prob.begin(), 
        [&](double x) {return x / sum;});
    
    std::transform(prob.begin(), prob.end(), distr.begin(), 
        [&](double x) {return ceil(x * N);});
    
    size_t del = std::accumulate(distr.begin(), distr.end(), 0) - N;
    
    while (del > 0) {
        size_t idx = floor(dist(engine) * n);
        dist(engine);
        
        while(distr.at(idx) == 1) {
            idx = floor(dist(engine) * n);
            dist(engine);
        }
        
        --distr.at(idx);
        --del;
    }

    std::transform(distr.begin(), distr.end(), prob_ans.begin(),
        [&](size_t x) -> double {return (double) x / N;});

    return {std::make_shared<std::vector<double>>(prob_ans),
            std::make_shared<std::vector<size_t>>(distr)};
}

Distribution ANSTableGenerator::generate_distribution_from_buffer(
    size_t seed, size_t N, std::uint8_t* in, size_t size) {
    
    const size_t max_num_symbols = 1 << (sizeof(SYMBOL_TYPE) * 8);
    
    std::uint32_t frequencies[max_num_symbols];
    
    for(size_t i = 0; i < max_num_symbols; ++i)
        frequencies[i] = 0;
    
    for(size_t i = 0; i < size; ++i)
        ++frequencies[in[i]];
    
    std::vector<size_t> freq_compact;
    std::vector<SYMBOL_TYPE> symbols_compact;
    
    for(size_t i = 0; i < max_num_symbols; ++i) {
        if(frequencies[i] != 0) {
            freq_compact.push_back(frequencies[i]);
            symbols_compact.push_back(i);
        }
    }
    
    std::sort(freq_compact.begin(), freq_compact.end(),
        [](size_t a, size_t b) {return a > b;});
    
    const size_t num_symbols = freq_compact.size();
    std::vector<double> prob_ans(num_symbols);
    std::vector<double> prob_a(num_symbols);
    
    // normalisation process
    double sum_a = std::accumulate(freq_compact.begin(), freq_compact.end(), 0.0f);
    std::transform(freq_compact.begin(), freq_compact.end(), prob_a.begin(), 
        [&](double x) {return x / sum_a;});
    
    std::transform(prob_a.begin(), prob_a.end(), prob_a.begin(), 
        [&](double x) {if(x < 0.001) return x += 0.001; else return x;});
    
    sum_a = std::accumulate(prob_a.begin(), prob_a.end(), 0.0f);
    std::transform(prob_a.begin(), prob_a.end(), prob_ans.begin(), 
        [&](double x) {return x / sum_a;});
    
    std::transform(prob_ans.begin(), prob_ans.end(), freq_compact.begin(), 
        [&](double x) {return ceil(x * N);});
    
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    size_t del = std::accumulate(freq_compact.begin(), freq_compact.end(), 0)
        - N;
    
    while (del > 0) {
        size_t idx = floor(dist(engine) * num_symbols);
        dist(engine);
        
        while(freq_compact.at(idx) == 1) {
            idx = floor(dist(engine) * num_symbols);
            dist(engine);
        }
        
        --freq_compact.at(idx);
        --del;
    }
    
    std::transform(freq_compact.begin(), freq_compact.end(), prob_ans.begin(),
        [&](size_t x) -> double {return (double) x / N;});
        
    return {std::make_shared<std::vector<double>>(prob_ans),
        std::make_shared<std::vector<size_t>>(freq_compact),
        std::make_shared<std::vector<SYMBOL_TYPE>>(symbols_compact)};
}

std::shared_ptr<std::vector<SYMBOL_TYPE>>
    ANSTableGenerator::generate_test_data(
    std::shared_ptr<std::vector<size_t>> distr,
    size_t size,
    size_t num_states,
    size_t seed) {
    
    std::vector<SYMBOL_TYPE> buffer(size + 8);
    SYMBOL_TYPE symbols[num_states];
    const size_t dist_len = distr->size();
    
    size_t index = 0;
    for(size_t i = 0; i < dist_len; ++i) {
        
        for(size_t j = 0; j < distr->at(i); ++j) {
            symbols[index] = i;
            ++index;
        }
    }
    
    std::mt19937 engine(seed);
    std::uniform_int_distribution<size_t> dist(0, num_states - 1);
    
    for(size_t i = 0; i < size; ++i) {
        buffer[i] = symbols[dist(engine)];
        dist(engine);
    }
    
    return std::make_shared<std::vector<SYMBOL_TYPE>> (buffer);
}

std::shared_ptr<std::vector<std::vector<Encoder_Table_Entry>>>
    ANSTableGenerator::generate_table(std::shared_ptr<std::vector<double>> P_s,
        std::shared_ptr<std::vector<size_t>> L_s,
        std::shared_ptr<std::vector<SYMBOL_TYPE>> symbols,
        size_t num_symbols, size_t num_states) {
    
    std::unordered_map<std::uint32_t, XS_pair> pretab;
    std::unordered_map<std::uint32_t, std::uint32_t> X_s;
    std::vector<Queue_Entry> queue(num_symbols);
    
    // n x N table
    const size_t table_size = (1 << (sizeof(SYMBOL_TYPE) * 8));
    std::vector<std::vector<Encoder_Table_Entry>> table;
    
    if(symbols == nullptr) {
        table.resize(num_symbols);
    
        for(size_t i = 0; i < num_symbols; ++i)
            table[i].resize(num_states);
    
        for(size_t i = 0; i < num_symbols; ++i) {
            queue[i] = {0.5f / P_s->at(i), (SYMBOL_TYPE) i};
            X_s[i] = L_s->at(i);
        }
    }
    
    else {
        table.resize(table_size);
    
        for(size_t i = 0; i < table_size; ++i)
            table[i].resize(num_states);
    
        for(size_t i = 0; i < num_symbols; ++i) {
            queue[i] = {0.5f / P_s->at(i), symbols->at(i)};
            X_s[i] = L_s->at(i);
        }
    }
    
    if(symbols == nullptr) {
        for(size_t i = num_states; i < 2 * num_states; ++i) {
            Queue_Entry q = *std::min_element(std::begin(queue), std::end(queue),
                [](Queue_Entry a, Queue_Entry b) {return a.p < b.p;});
            
            double v = q.p;
            size_t idx = q.s;
            SYMBOL_TYPE s = queue[idx].s;
            
            queue[idx].p = v + (1 / P_s->at(s));
            pretab[i] = {s, X_s[s]};
            
            X_s[s] = X_s[s] + 1;
        }
    }
    
    else {
        for(size_t i = num_states; i < 2 * num_states; ++i) {
            std::vector<Queue_Entry>::iterator q_it =
                std::min_element(std::begin(queue), std::end(queue),
                [](Queue_Entry a, Queue_Entry b) {return a.p < b.p;});
            
            size_t idx = std::distance(queue.begin(), q_it);
            
            double v = queue[idx].p;
            SYMBOL_TYPE s = queue[idx].s;
            
            queue[idx].p = v + (1 / P_s->at(idx));
            pretab[i] = {s, X_s[idx]};
            
            X_s[idx] = X_s[idx] + 1;
        }
    }
    
    for(std::uint32_t i = num_states; i < 2 * num_states; ++i) {
        XS_pair tab = pretab[i];
        SYMBOL_TYPE symbol = tab.s;
        std::uint32_t slide = tab.next_state;
        std::uint32_t shift = 0;
        
        for(; slide < 2 * num_states; slide = slide * 2) {
            for(std::uint32_t rem = 0; rem < (unsigned) (1 << shift); ++rem) {
                if(num_states <= (slide + rem)
                    && (slide + rem) < 2 * num_states) {
                    
                    size_t idx2 = slide + rem - num_states;
                    
                    table.at(symbol).at(idx2) = {i, rem, shift, symbol};
                }
            }
            
            ++shift;
        }
    }
    
    return std::make_shared<std::vector<std::vector<Encoder_Table_Entry>>>
        (table);
}

std::shared_ptr<CUHDCodetable> ANSTableGenerator::get_decoder_table(
    std::shared_ptr<ANSEncoderTable> enc_table) {
    
    const size_t number_of_states = enc_table->number_of_states;
    const size_t number_of_symbols = enc_table->number_of_symbols;
    
    CUHDCodetable table(number_of_states);
    CUHDCodetableItem* tab = table.get();
    
    for(size_t i = 0; i < number_of_symbols; ++i) {
        for(size_t j = 0; j < number_of_states; ++j) {
            CUHDCodetableItem item;
            
            std::uint32_t prev = enc_table->table.at(i).at(j).next_state
                - number_of_states;
            std::uint32_t len = enc_table->table.at(i).at(j).code_length;
            item.next_state = (number_of_states + j) >> len;
            item.symbol = enc_table->table.at(i).at(j).symbol;
            
            item.min_num_bits = tab[prev].min_num_bits == 0 && len > 0 ?
                len : tab[prev].min_num_bits;
            
            tab[prev] = item;
        }
    }

    return std::make_shared<CUHDCodetable> (table);
}

std::shared_ptr<ANSEncoderTable> ANSTableGenerator::generate_encoder_table(
    std::shared_ptr<std::vector<std::vector<Encoder_Table_Entry>>> tab) {
    
    ANSEncoderTable enc_table;
    const size_t number_of_symbols = tab->size();
    enc_table.number_of_states = tab->at(0).size();
    
    enc_table.max_number_of_symbols = 1 << (sizeof(SYMBOL_TYPE) * 8);
    enc_table.number_of_symbols = number_of_symbols;
    
    enc_table.table.resize(number_of_symbols);
    
    for(size_t i = 0; i < enc_table.number_of_symbols; ++i) {
        enc_table.table.at(i).resize(enc_table.number_of_states);
    
        for(size_t j = 0; j < enc_table.number_of_states; ++j) {
            Encoder_Table_Entry entry = tab->at(i).at(j);

            enc_table.table.at(i).at(j)
                = {entry.x, entry.rem, entry.shift, entry.symbol};
        }
    }

    return std::make_shared<ANSEncoderTable>(enc_table);
}

size_t ANSTableGenerator::get_max_compressed_size(
    std::shared_ptr<ANSEncoderTable> encoder_table, size_t input_size) {
    
    const size_t num_symbols = encoder_table->table.size();
    const size_t num_states = encoder_table->table.at(0).size();
    
    // find maximum codeword length
    const size_t max = encoder_table->table.at(num_symbols - 1)
        .at(num_states - 1).code_length;
        
    size_t size = ((max * input_size) / 8) + 1;
    
	// calculate number of units
	size_t compressed_size_units = size % sizeof(UNIT_TYPE) == 0 ?
		size / sizeof(UNIT_TYPE) : size / sizeof(UNIT_TYPE) + 1;
    
    return compressed_size_units;
}

