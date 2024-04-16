#include "Similarity.h"

int old_main()
{
    const float similarity_threshold = 0.80f;
    csv::CSVReader reader(R"|(C:\Temp\ndd-enron-100k\enron100k.csv)|");

    // todo: add sampling that enable setting up shingle size, i.e choose 1k docs and compare against various shingle sizes 

    auto env = lmdb::env::create();
    env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);
    env.open(R"|(C:\Temp\ndd-enron-100k)|", MDB_FIXEDMAP, 0664);

    using namespace std::chrono;
    auto start = steady_clock::now();
    auto wtxn = lmdb::txn::begin(env);
    auto dbi = lmdb::dbi::open(wtxn, nullptr);
    cache_record_func cache_record = [&](auto key, const auto& value) { return dbi.put(wtxn, to_val(key), to_val(value)); };
    index_documents(cache_record, reader);
    wtxn.commit();
    std::cout << "min-hash time in seconds : " << duration_cast<seconds>(steady_clock::now() - start).count() << " sec" << std::endl;

    start = steady_clock::now();
    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    dbi = lmdb::dbi::open(rtxn, nullptr);

    iterate_records_action iterate_records = [&](parse_record_action parse) {
        auto cursor = lmdb::cursor::open(rtxn, dbi);
        lmdb::val key{}, value{};
        while (cursor.get(key, value, MDB_NEXT))
        {
            parse(from_val(key), to_span<uint32_t>(value));
        }
        cursor.close();
    };
    get_record_func lookup = [&](auto k) { lmdb::val v; dbi.get(rtxn, to_val(k), v); return to_span<uint32_t>(v); };
    auto docid_xref = get_doc_ids(iterate_records, dbi.size(rtxn));
    auto similar_docs = find_pairs(rtxn, dbi, iterate_records, dbi.size(rtxn), lookup, docid_xref, similarity_threshold);
    std::cout << "lsh time in seconds : " << duration_cast<seconds>(steady_clock::now() - start).count() << " sec" << std::endl;
    rtxn.abort();

    std::ofstream myfile;
    myfile.open(R"|(C:\Temp\ndd-enron-100k\cpp-out.csv)|");
    myfile << "DocA, DocB, Similarity" << std::endl;
    int falsePositives{};
    for (const auto& pair : similar_docs)
    {
        myfile << docid_xref.at(pair.first) << ", " << docid_xref.at(pair.second) << ", " << 0 /* todo */ << "\n";
    }

    myfile.close();
}

void index_documents(cache_record_func cache_record, csv::CSVReader& reader)
{
    for (csv::CSVRow& row : reader)
    {
        auto docid = row[0].get_sv();
        const auto doctext = row[1].get_sv();
        assert((docid.size() > 0 && doctext.size()) || "doc data is corrupted");

        const auto signature = get_signature(normalize_text(doctext));
        assert((signature.size() == signature_size) || "could not insert entry");

        auto added = cache_record(docid, signature);
        assert(added || "could not insert entry");
    }
}

std::string normalize_text(const std::string_view data)
{
    // todo:
    //- replace whitespaces with spaces
    //- NonAlphaNumericFilter->Char.IsLetterOrDigit(c)
    //- remove stopwords->StopWords.txt
    //- tolowercase
    //- replace dupe spaces

    std::string norm; norm.reserve(data.size());
    std::unique_copy(data.cbegin(), data.cend(), std::back_inserter(norm),
        [](unsigned char a, unsigned char b) { return isspace(a) && isspace(b); });
    std::transform(norm.begin(), norm.end(), norm.begin(),
        [](unsigned char c) { return isspace(c) ? ' ' : std::tolower(c); }); // todo: fix for other langs
    return norm;
}

minhash_sig get_signature(const std::string_view data)
{
    std::vector<int> whitespaces({ 0 });
    for (int i = 1; i < data.size(); ++i)
    {
        if (isspace(static_cast<unsigned char>(data.at(i))))
        {
            whitespaces.push_back(i);
        }
    }

    hash_list hashes(std::max(1, static_cast<int>(whitespaces.size() - shingle_size)));
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
        MurmurHash3_x86_32(shingle.data(), shingle.length(), random_seed, hashes[i - s_size]);
    }
    
    return minhash(hashes);
}

const minhash_sig minhash(hash_list& hashes)
{
    static const auto coeffs = []()
    {
        std::mt19937 rng{ random_seed };
        std::uniform_int_distribution<uint32_t> uint_dist;
        std::array<uint32_t, signature_size> coeffs;
        std::generate(coeffs.begin(), coeffs.end(), [&]() { return uint_dist(rng); });
        return coeffs;
    }();

    size_t int_ctr{};
    minhash_sig sig;
    std::generate(sig.begin(), sig.end(), [&int_ctr, &hashes]() {
        const auto& xor_coeff = coeffs[int_ctr++];
        return std::reduce(hashes.cbegin(), hashes.cend(), std::numeric_limits<uint32_t>::max(), [&xor_coeff](auto min, auto hash) {
            return std::min(min, hash ^ xor_coeff);
            });
        });

    return sig;
}

pair_set find_pairs(lmdb::txn& t, lmdb::dbi& d, iterate_records_action iterate_records, size_t record_count, get_record_func lookup, const idx_docid_xref docid_xref, const float similarity_threshold)
{
    auto [band_cnt, row_cnt] = lsh_bands_n_rows(signature_size, similarity_threshold);

    std::cout << "Using " << band_cnt << " and " << row_cnt << " rows" << std::endl;
    std::cout << "About " << std::setprecision(2) << (lsh_false_negatives_prob(band_cnt, row_cnt, similarity_threshold) * 100) << "% of the " << similarity_threshold*100 << "%-similar pairs will be false negatives " << std::endl;
    std::cout << "We should find " << std::setprecision(2) << (lsh_cp_probability(band_cnt, row_cnt, similarity_threshold) * 100) << "% pairs of truly similar documents" << std::endl;

    auto [bands, doc_bucket_xref] = do_lsh(iterate_records, record_count, band_cnt, row_cnt);

    std::cout << "Amount of pairs to compare dropped to " << calculate_cp_ratio(bands) << "%" << std::endl;

    pair_set similar_docs;
    /*for (int doc_idx = 0; doc_idx < docid_xref.size(); ++doc_idx)
    {
        for (int band_id = 0; band_id < band_cnt; ++band_id)
        {
            const auto buctek_id = doc_bucket_xref[doc_idx * band_cnt + band_id];
            const auto& bucket = bands.at(band_id).at(buctek_id);

            for (const auto peer_idx : bucket)
            {
                if (doc_idx < peer_idx)
                {
                    dbi.get(rtxn, to_val(docid_xref.at(doc_idx)), sig_a);
                    dbi.get(rtxn, to_val(docid_xref.at(peer_idx)), sig_b);
                    if (!similar_docs.contains(std::make_pair(doc_idx, peer_idx))
                        && calculate_similarity(to_span<uint32_t>(sig_a), to_span<uint32_t>(sig_b)) >= similarity_threshold)
                    {
                        similar_docs.emplace(doc_idx, peer_idx);
                    }
                }
            }
        }
    }*/

    lmdb::val sig_a;

    for (auto& buckets : bands)
    {
        for (auto& [bucket_id, bucket] : buckets)
        {
            if (bucket.size() < 2)
            {
                continue;
            }
            for (size_t doc_idx = 0; doc_idx < bucket.size(); ++doc_idx)
            {
                for (size_t peer_idx = 0; peer_idx < bucket.size(); ++peer_idx)
                {
                    if (doc_idx < peer_idx)
                    {
                        auto pair = std::make_pair(doc_idx, peer_idx);
                        if (similar_docs.contains(pair))
                        {
                            continue;
                        }

                        d.get(t, to_val(docid_xref.at(doc_idx)), sig_a);
                        //auto a = to_span<uint32_t>(sig_a);

                        //const auto sig_a = lookup(docid_xref.at(doc_idx));
                        //const auto sig_b = lookup(docid_xref.at(peer_idx));

                        /*if (calculate_similarity(sig_a, sig_b) >= similarity_threshold)
                        {
                            similar_docs.emplace(doc_idx, peer_idx);
                        }*/
                    }
                }
            }
            bucket.clear();
        }
        buckets.clear();
    }

    std::cout << "Found " << similar_docs.size() << " similar doc pairs" << std::endl;

    return similar_docs;
}

idx_docid_xref get_doc_ids(iterate_records_action iterate_records, size_t record_count)
{
    idx_docid_xref xref;
    xref.reserve(record_count);
    iterate_records([&](std::string_view key, std::span<const uint32_t> value) 
    {
        xref.emplace_back(key);
    });
    return xref;
}

bucketing_result do_lsh(iterate_records_action iterate_records, size_t record_count, const int band_cnt, const int row_cnt)
{
    uint32_t doc_idx{}, bucket_id{};
    band_list bands(band_cnt);
    std::vector<uint32_t> idx_to_bucket(band_cnt * record_count, 0U);
    iterate_records([&](std::string_view docid, std::span<const uint32_t> docsig)
    {
        assert(docsig.size() == signature_size);

        for (int band_id = 0; band_id < band_cnt; ++band_id)
        {
            const auto row = docsig.subspan(band_id * row_cnt, row_cnt);
            MurmurHash3_x86_32(row.data(), row.size_bytes(), band_id, bucket_id);

            auto& band = bands.at(band_id);
            if (band.find(bucket_id) == band.end())
            {
                band.emplace(bucket_id, bucket({ doc_idx }));
            }
            else
            {
                band.at(bucket_id).push_back(doc_idx);
            }
            idx_to_bucket.at(doc_idx * band_cnt + band_id) = bucket_id;
        }
        ++doc_idx;
    });

    return { bands, idx_to_bucket };
}

float calculate_cp_ratio(const band_list& bands)
{
    size_t cp_cnt{}, total_cnt{};
    for (const auto& band : bands) {
        for (const auto& [bucket_id, bucket] : band) {
            total_cnt += bucket.size();
            if (bucket.size() < 2) continue;
            for (size_t doc_idx = 0; doc_idx < bucket.size(); ++doc_idx)
            {
                for (size_t peer_idx = 0; peer_idx < bucket.size(); ++peer_idx)
                {
                    if (doc_idx < peer_idx) ++cp_cnt;
                }
            }
        }
    }
    return static_cast<float>(cp_cnt) / pow(total_cnt / bands.size(), 2);
}