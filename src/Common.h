#pragma once

// FIXME: we should look into precompiled headers, see vk_types.h and corresponding cmake files.
#include <vulkan/vulkan_raii.hpp>

#include <VmaUsage.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

struct AllocatedImage {
    vk::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
    VmaAllocation allocation = nullptr;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;

    // FIXME: this isn't great. do we want an raii wrapper for vma, or just use std::unique_ptr with custom deletor?
    void clear(VmaAllocator allocator) {
        imageView.clear();
        if (image) {
            vmaDestroyImage(allocator, image, allocation);
        }
    }
};

struct AllocatedBuffer {
    vk::Buffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info;

    // FIXME: same issue here
    void clear(VmaAllocator allocator) {
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
    }
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// Holds the resources needed for a mesh.
struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    vk::DeviceAddress vertexBufferAddress;
};

// Push constants for our mesh object draws.
struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    vk::DeviceAddress vertexBuffer;
};

/*struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};*/
