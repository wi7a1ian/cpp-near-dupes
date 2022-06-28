#pragma once

#include <string_view>
#include <array>
#include <vector>
#include <span>
#include "lmdb++.h"


#ifdef _DEBUG
constexpr bool debug_mode = true;
#else
constexpr bool debug_mode = false;
#endif

#if _MSC_VER && !__INTEL_COMPILER
constexpr bool is_using_msvc = true;
#else
constexpr bool is_using_msvc = false;
#endif

inline auto to_val(const std::string_view& str)
{
    return lmdb::val(str.data(), str.size());
}

//template<typename T, typename std::enable_if<std::is_integral_v<T>, int>::type = 0>
template<typename T, typename = std::enable_if<std::is_integral_v<T>>>
inline auto to_key(T& id)
{
    std::reverse(reinterpret_cast<uint8_t*>(&id), reinterpret_cast<uint8_t*>(&id) + sizeof(T));
    return lmdb::val(&id, sizeof(T));
}

template<typename T>
inline auto to_val(const std::vector<T>& vec)
{
    return lmdb::val(vec.data(), vec.size() * sizeof(T));
}

template<typename T = uint32_t>
inline T from_key(const lmdb::val& val)
{
    T id = *reinterpret_cast<const T*>(val.data());
    std::reverse(reinterpret_cast<uint8_t*>(&id), reinterpret_cast<uint8_t*>(&id) + sizeof(T));
    return id;
}

template<typename T>
inline auto to_span(const lmdb::val& val)
{
    return std::span<const T>(reinterpret_cast<const T*>(val.data()), val.size() / sizeof(T));
}
