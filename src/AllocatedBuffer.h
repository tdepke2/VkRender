#pragma once

#include <VmaUsage.h>

#include <vulkan/vulkan_raii.hpp>

class AllocatedBuffer {
public:
    static void setAllocator(VmaAllocator allocator);

    AllocatedBuffer() = default;
    AllocatedBuffer(size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    ~AllocatedBuffer();
    AllocatedBuffer(const AllocatedBuffer& rhs) = delete;
    AllocatedBuffer(AllocatedBuffer&& rhs) noexcept;
    AllocatedBuffer& operator=(const AllocatedBuffer& rhs) = delete;
    AllocatedBuffer& operator=(AllocatedBuffer&& rhs) noexcept;

    const vk::Buffer& getBuffer() const;
    const VmaAllocationInfo& getInfo() const;

    void clear() noexcept;

private:
    static VmaAllocator allocator_;

    vk::Buffer buffer_ = nullptr;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo info_;
};
