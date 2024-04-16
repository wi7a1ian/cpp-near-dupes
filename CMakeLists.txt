cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("NearDupes") # LANGUAGES CXX) We will not only use C++ because LMDB is C
#set(CMAKE_CXX_STANDARD 20) # Force C++20 stardard for all targets
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2") # Enable vector extensions for all targets

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(global_pch_n_options INTERFACE)
target_precompile_headers( global_pch_n_options INTERFACE
    <cassert>
    <cmath>
    <vector>
    <span>
    <tuple>
    <unordered_set>
    <string>
    <string_view>
    <sstream>
    <algorithm>
    <execution>
    <iostream>
    <iomanip>
    <fstream>
    <random>
    <chrono>
    <cctype>
    <filesystem>)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #add_compile_options(-mavx2)
    target_compile_options(global_pch_n_options INTERFACE -mavx2)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #add_compile_options(/arch:AVX2)
    target_compile_options(global_pch_n_options INTERFACE /arch:AVX2)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")

add_subdirectory(third_party)
add_subdirectory (src)
