cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("NearDupes" LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)
target_precompile_headers( project_options INTERFACE
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

# Include sub-projects.
add_subdirectory(third_party)
add_subdirectory (src)
