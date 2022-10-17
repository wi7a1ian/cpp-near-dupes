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
#include <include/csv.hpp> // TODO
#include "lmdb++.h" // why ""  and not <> ?

using namespace std;

int main()
{
	const float similarity_threshold = 0.80f;
    csv::CSVReader reader(R"|(enron100k.csv)|");

    // todo: add sampling that enable setting up shingle size, i.e choose 1k docs and compare against various shingle sizes 

    auto env = lmdb::env::create();
    env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);
    env.open(R"|(ndd-enron-100k)|", MDB_FIXEDMAP, 0664);

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
    myfile.open(R"|(cpp-out.csv)|");
    myfile << "DocA, DocB, Similarity" << std::endl;
    int falsePositives{};
    for (const auto& pair : similar_docs)
    {
        myfile << docid_xref.at(pair.first) << ", " << docid_xref.at(pair.second) << ", " << 0 /* todo */ << "\n";
    }

    myfile.close();
}
