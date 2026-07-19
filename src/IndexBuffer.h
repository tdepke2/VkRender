#pragma once

#include <AllocatedBuffer.h>

class IndexBuffer : public AllocatedBuffer {
public:
    IndexBuffer() = default;
    IndexBuffer(size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);
};
