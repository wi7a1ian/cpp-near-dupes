#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <span>
#include <tuple>
#include <unordered_set>
#include <string>
#include <string_view>
#include <sstream>
#include <algorithm>
#include <execution>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>
#include <cctype>
#include <filesystem>
#include "csv.h"
#include "lmdb++.h"
#include "utils.h"
#include "MurMurHash3.h"

const size_t random_seed = 71; // vert sensitive, causes hash collisions, hence increasing cp amount
const size_t signature_size = 256;
const size_t shingle_size = 3;

namespace similarity
{
    using std::vector;
    using std::array;
    using std::unordered_map;
    using std::span;
    using std::string_view;
    using std::function;
    using std::pair;
    using std::tuple;

    using doc_id = uint32_t;
    using shingle_set = vector<uint32_t>;
    using shingle_view = span<const uint32_t>;
    using minhash_sig = array<uint32_t, signature_size>;
    using bucket = vector<doc_id>;
    using bucket_list = unordered_map<size_t, bucket>;
    using band_list = vector<bucket_list>;
    using idx_docid_xref = vector<std::string>;
    using put_record_func = function<bool(doc_id, const shingle_set&)>;
    using seek_record_func = function<const shingle_view(doc_id)>;
    using parse_record_action = function<void(doc_id, const shingle_view)>;
    using iterate_records_action = function<void(parse_record_action)>;
    using parse_input_action = function<void(string_view, string_view)>;
    using iterate_input_action = function<void(parse_input_action)>;
    using nd_groups = unordered_map<doc_id, vector<pair<doc_id, float>>>;


    struct doc_cacher
    {
        idx_docid_xref xref;

        inline string_view get_id_for(doc_id idx) const
        {
            return xref.at(idx);
        }

        void add_documents(iterate_input_action iterate_input, put_record_func put_record)
        {
            doc_id doc_idx{};
            iterate_input([&](auto docid, auto doctext) {
                if constexpr (debug_mode)
                {
                    if ((doc_idx % 1000) == 0) { std::cout << "Done reading " << doc_idx << "\n"; }
                }

                assert((docid.size() > 0 || doctext.size() > 0) || "doc data is corrupted");
                if (docid.size() == 0)
                {
                    throw std::runtime_error{ "Incorrect doc data." };
                }

                if (doctext.size() == 0)
                {
                    return;
                }

                auto added = put_record(doc_idx++, generate_shingles(normalize_text(doctext)));
                assert(added || "could not insert entry");

                if (!added)
                {
                    std::cout << "Failed adding item " << (doc_idx - 1) << std::endl;
                    throw std::runtime_error{ "Lmdb database ran out of space." };
                }

                xref.emplace_back(docid);
            });
        }

        std::string normalize_text(const string_view data) const
        {
            std::string norm; norm.reserve(data.size());
            std::unique_copy(data.cbegin(), data.cend(), std::back_inserter(norm),
                [](unsigned char a, unsigned char b) { return isspace(a) && isspace(b); });
            std::transform(norm.begin(), norm.end(), norm.begin(),
                [](unsigned char c) { return isspace(c) ? ' ' : std::tolower(c); }); // todo: fix for other langs
            return norm;
        }

        shingle_set generate_shingles(const string_view data) const
        {
            vector<int> whitespaces({ 0 });
            for (int i = 1; i < data.size(); ++i)
            {
                if (isspace(static_cast<unsigned char>(data.at(i))))
                {
                    whitespaces.push_back(i);
                }
            }

            shingle_set shingles(std::max(1, static_cast<int>(whitespaces.size() - shingle_size)));
            const auto s_size = std::min(shingle_size, whitespaces.size());
            for (size_t i = s_size; i < whitespaces.size(); ++i)
            {
                auto start = whitespaces[i - s_size];
                auto length = whitespaces[i] - start;

                if (isspace(data[start]))
                {
                    ++start; --length;
                }

                const auto shingle = data.substr(start, length);
                MurmurHash3_x86_32(shingle.data(), shingle.length(), random_seed, shingles[i - s_size]);
            }

            std::sort(shingles.begin(), shingles.end());
            shingles.erase(std::unique(shingles.begin(), shingles.end()), shingles.end());
            return shingles;
        }
    };

    struct lsh_index
    {
        band_list bands;
        int band_cnt{};
        int row_cnt{};

        lsh_index(const int band_count, const int row_count)
        {
            band_cnt = band_count;
            row_cnt = row_count;
            bands = band_list(band_cnt);
        }

        vector<doc_id> get_candidates(const minhash_sig& signature) const
        {
            uint32_t bucket_id{};
            vector<uint32_t> candidates;
            auto docsig = span(signature);
            for (int band_id = 0; band_id < bands.size(); ++band_id)
            {
                const auto row = docsig.subspan(band_id * row_cnt, row_cnt);
                MurmurHash3_x86_32(row.data(), row.size_bytes(), band_id, bucket_id);

                auto& band = bands.at(band_id);
                if (band.find(bucket_id) != band.end())
                {
                    auto& bucket = band.at(bucket_id);
                    candidates.insert(candidates.end(), bucket.begin(), bucket.end());
                }
            }

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            return candidates;
        }

        void add(doc_id id, const minhash_sig& signature)
        {
            auto docsig = span(signature);
            uint32_t bucket_id{};
            for (int band_id = 0; band_id < bands.size(); ++band_id)
            {
                const auto row = docsig.subspan(band_id * row_cnt, row_cnt);
                MurmurHash3_x86_32(row.data(), row.size_bytes(), band_id, bucket_id);

                auto& band = bands.at(band_id);
                if (band.find(bucket_id) == band.end())
                {
                    band.emplace(bucket_id, bucket({ id }));
                }
                else
                {
                    band.at(bucket_id).push_back(id);
                }
            }
        }
    };

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

    inline tuple<int, int> lsh_bands_n_rows(int sig_size, float similarity_threshold = 0.8f)
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
    inline float calculate_similarity(const Cont a, const Cont b)
    {
        int intersect_cnt{};
        for (const auto& item_a : a)
        {
            if (std::find(b.begin(), b.end(), item_a) != b.end())
            {
                ++intersect_cnt;
            }
        }
        const int union_cnt = a.size() + b.size() - intersect_cnt;
        return static_cast<float>(intersect_cnt) / union_cnt;
    }


    const minhash_sig minhash(shingle_view shingles)
    {
        static const auto coeffs = []()
        {
            std::mt19937 rng{ random_seed };
            std::uniform_int_distribution<uint32_t> uint_dist;
            array<uint32_t, signature_size> coeffs;
            std::generate(coeffs.begin(), coeffs.end(), [&]() { return uint_dist(rng); });
            return coeffs;
        }();

        size_t int_ctr{};
        minhash_sig sig;

        if constexpr (is_using_msvc)
        {
            for (int i = 0; i < sig.size(); ++i)
            {
                uint32_t min = std::numeric_limits<uint32_t>::max();
                for (int j = 0; j < shingles.size(); ++j)
                {
                    uint32_t a = shingles[j] ^ coeffs[i];
                    if (a < min) min = a;
                }
                sig[i] = min;
            }
        }
        else
        {
            std::generate(sig.begin(), sig.end(), [&int_ctr, &shingles]() {
                const auto& xor_coeff = coeffs[int_ctr++];
                return std::reduce(shingles.begin(), shingles.end(), std::numeric_limits<uint32_t>::max(), [&xor_coeff](auto min, auto hash) {
                    return std::min(min, hash ^ xor_coeff);
                    });
                });
        }

        return sig;
    }

    template<typename TLshIndex = lsh_index>
    nd_groups find_near_dupes(iterate_records_action iterate_records, size_t record_count, seek_record_func seek_record, const float similarity_threshold)
    {
        auto [band_cnt, row_cnt] = lsh_bands_n_rows(signature_size, similarity_threshold);
        TLshIndex lsh(band_cnt, row_cnt);

        if constexpr (debug_mode)
        {
            std::cout << "Using " << band_cnt << " and " << row_cnt << " rows" << std::endl;
            std::cout << "About " << std::setprecision(2) << (lsh_false_negatives_prob(band_cnt, row_cnt, similarity_threshold) * 100) << "% of the " << similarity_threshold * 100 << "%-similar pairs will be false negatives " << std::endl;
            std::cout << "We should find " << std::setprecision(2) << (lsh_cp_probability(band_cnt, row_cnt, similarity_threshold) * 100) << "% pairs of truly similar documents" << std::endl;
        }

        vector<size_t> doc_size_xref(record_count);
        iterate_records([&, i = 0](auto idx, auto shingles) mutable { doc_size_xref.at(i++) = static_cast<uint32_t>(shingles.size()); });

        vector<doc_id> docs_by_size_desc(record_count);
        std::generate(docs_by_size_desc.begin(), docs_by_size_desc.end(), [i = 0]() mutable { return i++; });
        auto sort_by_size_desc = [&doc_size_xref](const doc_id& a, const doc_id& b) -> bool { return doc_size_xref[a] > doc_size_xref[b]; };
        std::sort(docs_by_size_desc.begin(), docs_by_size_desc.end(), sort_by_size_desc);

        nd_groups groups;
        for (const auto& doc_a : docs_by_size_desc)
        {
            float best_score{};
            doc_id best_match{};

            const auto shingles_a = seek_record(doc_a);
            assert(!shingles_a.empty());
            const auto signature = minhash(shingles_a);
            auto candidates = lsh.get_candidates(signature);
            std::sort(candidates.begin(), candidates.end(), sort_by_size_desc);

            for (const auto& doc_b : candidates)
            {
                if ((static_cast<float>(doc_size_xref[doc_a]) / doc_size_xref[doc_b]) < similarity_threshold)
                {
                    break; // this one and next candidates are too small
                }

                const auto shingles_b = seek_record(doc_b);
                assert(!shingles_b.empty());

                const auto score = calculate_similarity(shingles_a, shingles_b);
                if (score > best_score)
                {
                    best_score = score;
                    best_match = doc_b;
                }
            }

            if (best_score >= similarity_threshold)
            {
                groups[best_match].emplace_back(doc_a, best_score);
            }
            else
            {
                lsh.add(doc_a, signature);
                groups.emplace(doc_a, vector<pair<doc_id, float>>());
            }
        }

        return groups;
    }
}