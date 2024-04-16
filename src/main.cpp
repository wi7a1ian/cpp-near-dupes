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
#include <MurMurHash3.h>
#include <csv.hpp>
#include <lmdb++.h>
#include "core.h"
#include "utils.h"

using namespace std;

int main()
{
	const float similarity_threshold = 0.80f;
    csv::CSVReader reader(R"|(/workspaces/cpp-near-dupes/data/enron100k.csv)|");

    // TODO: add sampling that enable setting up shingle size, i.e choose 1k docs and compare against various shingle sizes 

    auto env = lmdb::env::create();
    env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);
    env.open(R"|(/tmp/ndd-cache)|", MDB_FIXEDMAP, 0664);

    using namespace std::chrono;
    auto start = steady_clock::now();
    auto wtxn = lmdb::txn::begin(env);
    auto dbi = lmdb::dbi::open(wtxn, nullptr);

    similarity::iterate_input_action iterate_csv_records = [&](similarity::parse_input_action parse) {
        for (csv::CSVRow& row : reader)
        {
            auto docid = row[0].get_sv();
            const auto doctext = row[1].get_sv();
            assert((docid.size() > 0 && doctext.size()) || "doc data is corrupted");
            parse(docid, doctext);
        }
    };

    // TODO: ensure records are appended, not inserted
    // docs: http://www.lmdb.tech/doc/group__internal.html#ga4fa8573d9236d54687c61827ebf8cac0
    similarity::put_record_func put_record = [&](auto key, const auto& value) { return dbi.put(wtxn, to_key(key), to_val(value)); };

    similarity::doc_cacher cache;
    cache.add_documents(iterate_csv_records, put_record);
    std::cout << "Min-hash took " << duration_cast<seconds>(steady_clock::now() - start).count() << " sec" << std::endl;
    std::cout << "Processed " << dbi.size(wtxn) << " records" << std::endl;
    wtxn.commit();

    start = steady_clock::now();
    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    dbi = lmdb::dbi::open(rtxn, nullptr);

    similarity::iterate_records_action iterate_cache_records = [&](similarity::parse_record_action parse) {
        auto cursor = lmdb::cursor::open(rtxn, dbi);
        lmdb::val key{}, value{};
        while (cursor.get(key, value, MDB_NEXT))
        {
            parse(from_key(key), to_span<uint32_t>(value));
        }
        cursor.close();
    };

    similarity::seek_record_func seek_record = [&](auto key) {
        lmdb::val v; dbi.get(rtxn, to_key(key), v); return to_span<uint32_t>(v);
    };
    
    auto groups = similarity::find_near_dupes(iterate_cache_records, dbi.size(rtxn), seek_record, similarity_threshold);

    std::cout << "LSH took " << duration_cast<seconds>(steady_clock::now() - start).count() << " sec" << std::endl;
    std::cout << "Produced " << groups.size() << " records" << std::endl;
    rtxn.abort();

    std::ofstream myfile;
    myfile.open(R"|(/tmp/ndd-groups.out.csv)|");
    myfile << "DocA, DocB, Similarity" << std::endl;
    int falsePositives{};
    for (const auto& group : groups) {
        myfile << cache.get_id_for(group.first) << ", " << cache.get_id_for(group.first) << ", 1 \n";
        for (const auto& near_dupes : group.second)
        {
            myfile << cache.get_id_for(group.first) << ", " << cache.get_id_for(near_dupes.first) << ", " << near_dupes.second << "\n";
        }
    }

    myfile.close();
}

