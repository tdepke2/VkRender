#pragma once

// FIXME: we should look into precompiled headers, see vk_types.h and corresponding cmake files.
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>    // FIXME: we want vk::to_string() instead
#include <vk_mem_alloc.h>

#include <spdlog/fmt/bundled/base.h>

struct AllocatedImage {
    vk::Image image;
    vk::raii::ImageView imageView = nullptr;
    VmaAllocation allocation;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;

    // FIXME: this isn't great. do we want an raii wrapper for vma, or just use std::unique_ptr with custom deletor?
    void clear(VmaAllocator allocator) {
        imageView.clear();
        vmaDestroyImage(allocator, image, allocation);
    }
};

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)
