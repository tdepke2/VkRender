#pragma once

// FIXME: we should look into precompiled headers, see vk_types.h and corresponding cmake files.
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>    // FIXME: we want vk::to_string() instead

// FIXME: may want a VmaUsage.h file instead, like vma example.
#include <vk_mem_alloc.h>

#include <spdlog/fmt/bundled/base.h>

// FIXME: I don't like this, why is it needed? I had to set this once we introduced perspective projection with 10000 near and 0.1 far planes.
// Setting it here isn't good either. It could go in cmake scripts.
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

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

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};

/*struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};*/

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)
