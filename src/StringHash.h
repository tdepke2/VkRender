#pragma once

#include <cstdint>
#include <string_view>

struct StringHash {
public:
    // With consteval (instead of constexpr), we enforce that the hash will be computed at compile-time.
    consteval StringHash(const char* str) :
        str(str, constexpr_strlen(str)),
        hash(fnv1a_32(str, constexpr_strlen(str))) {
    }
    consteval StringHash(std::string_view str) :
        str(str),
        hash(fnv1a_32(str.data(), str.size())) {
    }

    const std::string_view str;
    const uint32_t hash;

private:
    // FNV-1a 32-bit hashing algorithm.
    // From: https://github.com/vblanco20-1/vulkan-guide/blob/engine/extra-engine/string_utils.h
    static constexpr uint32_t fnv1a_32(const char* str, size_t count) {
        return ((count ? fnv1a_32(str, count - 1) : 2166136261u) ^ str[count]) * 16777619u;
    }

    // Could also use std::char_traits<char>::length() in C++17.
    static constexpr size_t constexpr_strlen(const char* str) {
        return (str && str[0]) ? (constexpr_strlen(&str[1]) + 1) : 0;
    }
};
