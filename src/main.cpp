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
    csv::CSVReader reader(R"|(enron100k.csv)|");

    // todo: add sampling that enable setting up shingle size, i.e choose 1k docs and compare against various shingle sizes 

    auto env = lmdb::env::create();
    env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);
    env.open(R"|(ndd-enron-100k)|", MDB_FIXEDMAP, 0664);

    using namespace std::chrono;
    auto start = steady_clock::now();
    auto wtxn = lmdb::txn::begin(env);
    auto dbi = lmdb::dbi::open(wtxn, nullptr);

    similarity::iterate_input_action iterate_records = [&](similarity::parse_input_action parse) {
        for (csv::CSVRow& row : reader)
        {
            auto docid = row[0].get_sv();
            const auto doctext = row[1].get_sv();
            assert((docid.size() > 0 && doctext.size()) || "doc data is corrupted");
            parse(docid, doctext);
        }
    };
    similarity::put_record_func put_record = [&](auto key, const auto& value) { return dbi.put(wtxn, key, to_val(value)); };

    similarity::doc_cacher cacher;
    cacher.add_documents(iterate_records, put_record);
    wtxn.commit();
    std::cout << "min-hash time in seconds : " << duration_cast<seconds>(steady_clock::now() - start).count() << " sec" << std::endl;
}

