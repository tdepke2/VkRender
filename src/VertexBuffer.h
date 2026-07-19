#pragma once

#include <AllocatedBuffer.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};
static_assert(sizeof(Vertex) == 16 * 3);

class VertexBuffer : public AllocatedBuffer {
public:
    VertexBuffer() = default;
    VertexBuffer(const vk::Device& device, size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    const vk::DeviceAddress& getBufferAddress() const;

private:
    vk::DeviceAddress bufferAddress_ = 0;
};
