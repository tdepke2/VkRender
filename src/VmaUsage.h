#pragma once

#include <spdlog/fmt/bundled/base.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_to_string.hpp>

#define VK_CHECK(x)                                                           \
    do {                                                                      \
        auto result = static_cast<vk::Result>(x);                             \
        if (result != vk::Result::eSuccess) {                                 \
            fmt::println("Detected Vulkan error: {}", vk::to_string(result)); \
            abort();                                                          \
        }                                                                     \
    } while (0)
