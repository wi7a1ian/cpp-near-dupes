cmake_minimum_required (VERSION 3.8)

# csv-parser
# https://github.com/vincentlaucsb/csv-parser
# git submodule add https://github.com/vincentlaucsb/csv-parser ./third_party/csv-parser
#add_subdirectory(csv-parser)
add_library(csv INTERFACE)
target_include_directories(csv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/csv-parser/single_include)
target_link_libraries(csv INTERFACE global_pch_n_options)

# MurMurHash3
# https://github.com/aappleby/smhasher
add_library(murmurhash STATIC ${CMAKE_CURRENT_SOURCE_DIR}/MurmurHash3/MurMurHash3.cpp)
target_include_directories(murmurhash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/MurmurHash3)
target_link_libraries(murmurhash PRIVATE global_pch_n_options)

# Lmdb/Lmdb++
# https://github.com/LMDB/lmdb
# https://github.com/drycpp/lmdbxx
# git submodule add https://github.com/LMDB/lmdb ./third_party/lmdb
set(LMDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lmdb/libraries/liblmdb)
add_library(lmdb STATIC 
    ${LMDB_INCLUDE_DIR}/mdb.c
    ${LMDB_INCLUDE_DIR}/midl.c)
target_include_directories(lmdb PUBLIC ${LMDB_INCLUDE_DIR})
add_library(lmdbxx INTERFACE)
target_include_directories(lmdbxx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lmdbxx)
target_link_libraries(lmdbxx INTERFACE lmdb) # Link lmdbxx with lmdb, establishing the dependency
target_link_libraries(lmdbxx INTERFACE global_pch_n_options)
