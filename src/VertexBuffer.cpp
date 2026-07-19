#include <VertexBuffer.h>

VertexBuffer::VertexBuffer(const vk::Device& device, size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage) :
    AllocatedBuffer(size, usage, memoryUsage) {

    bufferAddress_ = device.getBufferAddress({ .buffer = getBuffer() });
}

const vk::DeviceAddress& VertexBuffer::getBufferAddress() const {
    return bufferAddress_;
}
