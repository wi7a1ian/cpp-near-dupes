#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <span>
#include <tuple>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <execution>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>
#include <cctype>
#include "csv.h"
#include "lmdb++.h"
#include "MurMurHash3.h"

#ifdef _DEBUG
constexpr bool debug_mode = true;
#else
constexpr bool debug_mode = false;
#endif

const size_t random_seed = 71; // vert sensitive, causes hash collisions, hence increasing cp amount
const size_t signature_size = 100;
const size_t shingle_size = 5;

using hash_list = std::vector<uint32_t>;
using minhash_sig = std::array<uint32_t, signature_size>;
using bucket = std::vector<uint32_t>;
using bucket_list = std::unordered_map<uint32_t, bucket>;
using band_list = std::vector<bucket_list>;
using idx_bucket_xref = std::vector<uint32_t>;
using doc_pair = std::pair<uint32_t, uint32_t>;
using idx_docid_xref = std::vector<std::string>;
using bucketing_result = std::tuple<band_list, idx_bucket_xref>;
using cache_record_func = std::function<bool(std::string_view, const minhash_sig&)>;
using parse_record_action = std::function<void(std::string_view, const std::span<const uint32_t>)>;
using iterate_records_action = std::function<void(parse_record_action)>;
using get_record_func = std::function<const std::span<const uint32_t>(std::string_view)>;

auto pair_compare = [](const doc_pair& a, const doc_pair& b) {
    return (a.first < b.first && a.second < b.second)
        || (a.first < b.second && a.second < b.first);
};

auto pair_equal = [](const doc_pair& a, const doc_pair& b) {
    return (a.first == b.first && a.second == b.second)
        || (a.first == b.second && a.second == b.first);
};

template <typename T, typename... Rest>
inline size_t hash_combine(size_t seed, const T& v, Rest... rest) {
    if constexpr (sizeof...(Rest) > 0)
    {
        const auto hash = seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        return hash_combine(hash, rest...);
    }
    else
    {
        return seed;
    }
}

auto candidate_pair_hasher = [](const doc_pair& a) {
    return hash_combine(random_seed, a.first, a.second);
};
using candidate_pair_set = std::unordered_set<doc_pair, decltype(candidate_pair_hasher), decltype(pair_equal)>; // about 27% slower than array
using candidate_pair_array = std::vector<doc_pair>;
using pair_set = std::set<doc_pair>;

inline auto to_val(const std::string_view& str)
{
    return lmdb::val(str.data(), str.size());
}

template<typename T>
inline auto to_val(const std::array<T, signature_size>& vec)
{
    return lmdb::val(vec.data(), vec.size() * sizeof(T));
}

inline auto from_val(const lmdb::val& val)
{
    return std::string_view(val.data(), val.size());
}

template<typename T>
inline auto to_span(const lmdb::val& val)
{
    return std::span<const T>(reinterpret_cast<const T*>(val.data()), val.size() / sizeof(T));
}

inline auto lsh_threshold(int bands, int rows)
{
    return pow(1.0f / bands, 1.0f / rows);
}

inline auto lsh_cp_probability(int bands, int rows, float expected_similarity)
{
    return 1.0f - pow(1.0f - pow(expected_similarity, rows), bands);
}

inline auto lsh_false_negatives_prob(int bands, int rows, float expected_similarity)
{
    return pow(pow(expected_similarity, rows), bands);
}

inline std::tuple<int, int> lsh_bands_n_rows(int sig_size, float similarity_threshold = 0.8f)
{
    // Choose bands and rows so they produce similarity threshold just below expected one.
    // Thanks to that we lower chance of false negatives.
    float min_sim = 1;
    int bands = sig_size;
    for (int rows = 1; rows <= sig_size; ++rows)
    {
        if (sig_size % rows) continue;
        auto sim = lsh_threshold((sig_size / rows), rows);
        if (sim >= similarity_threshold) break;
        bands = sig_size / rows;
    }
    return { bands, sig_size / bands };
}

template<typename Cont>
float calculate_similarity(const Cont a, const Cont b)
{
    int intersect_cnt{}, union_cnt = b.size();
    for (const auto& item_a : a)
    {
        if (std::find(b.cbegin(), b.cend(), item_a) != b.cend())
        {
            ++intersect_cnt;
        }
        else
        {
            ++union_cnt;
        }
    }
    return static_cast<float>(intersect_cnt) / union_cnt;
}

void index_documents(cache_record_func cache_record, csv::CSVReader& reader);

std::string normalize_text(const std::string_view data);

minhash_sig get_signature(const std::string_view data);

const minhash_sig minhash(hash_list& hashes);

idx_docid_xref get_doc_ids(iterate_records_action iterate_records, size_t record_count);

pair_set find_pairs(lmdb::txn& t, lmdb::dbi& d, iterate_records_action iterate_records, size_t record_count, get_record_func lookup, const idx_docid_xref docid_xref, const float similarity_threshold);

bucketing_result do_lsh(iterate_records_action iterate_records, size_t record_count, const int band_cnt, const int row_cnt);

float calculate_cp_ratio(const band_list& bands);
