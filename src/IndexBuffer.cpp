#include <IndexBuffer.h>

IndexBuffer::IndexBuffer(size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage) :
    AllocatedBuffer(size, usage, memoryUsage) {
}
