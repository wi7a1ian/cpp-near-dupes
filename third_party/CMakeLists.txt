cmake_minimum_required (VERSION 3.8)

# csv-parser
# https://github.com/mafintosh/csv-parser
add_library(csv INTERFACE)
target_include_directories(csv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/csv-parser)

# MurMurHash3
# https://github.com/aappleby/smhasher
add_library(murmurhash STATIC ${CMAKE_CURRENT_SOURCE_DIR}/MurmurHash3/MurMurHash3.cpp)
target_include_directories(murmurhash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/MurmurHash3)
target_link_libraries(murmurhash PRIVATE project_options)

# Lmdb/Lmdb++
# https://github.com/LMDB/lmdb
# https://github.com/drycpp/lmdbxx
# git submodule add https://github.com/LMDB/lmdb ./third_party/lmdb
set(LMDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lmdb/libraries/liblmdb)
add_library(lmdb STATIC 
    ${LMDB_INCLUDE_DIR}/mdb_copy.c
    ${LMDB_INCLUDE_DIR}/mdb_drop.c
    ${LMDB_INCLUDE_DIR}/mdb_dump.c
    ${LMDB_INCLUDE_DIR}/mdb_load.c
    ${LMDB_INCLUDE_DIR}/mdb_stat.c
    ${LMDB_INCLUDE_DIR}/mdb.c
    ${LMDB_INCLUDE_DIR}/midl.c)
set_target_properties(lmdb PROPERTIES 
    LINKER_LANGUAGE CXX)
add_library(lmdbxx INTERFACE)
target_include_directories(lmdbxx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lmdbxx)