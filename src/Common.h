#pragma once

// FIXME: we should look into precompiled headers, see vk_types.h and corresponding cmake files.
#include <vulkan/vulkan_raii.hpp>

#include <VmaUsage.h>

#include <glm/mat4x4.hpp>

#include <IndexBuffer.h>
#include <VertexBuffer.h>


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





// Holds the resources needed for a mesh.
struct GPUMeshBuffers {
    IndexBuffer indexBuffer;
    VertexBuffer vertexBuffer;
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
