#include <AllocatedBuffer.h>

VmaAllocator AllocatedBuffer::allocator_ = nullptr;

void AllocatedBuffer::setAllocator(VmaAllocator allocator) {
    allocator_ = allocator;
}

AllocatedBuffer::AllocatedBuffer(size_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    vk::BufferCreateInfo bufferInfo = {
        .size = size,
        .usage = usage
    };

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = {};
    VK_CHECK(vmaCreateBuffer(allocator_, &*bufferInfo, &allocInfo, &buffer, &allocation_, &info_));
    buffer_ = buffer;
}

AllocatedBuffer::~AllocatedBuffer() {
    clear();
}

AllocatedBuffer::AllocatedBuffer(AllocatedBuffer&& rhs) noexcept :
    buffer_(std::exchange(rhs.buffer_, nullptr)),
    allocation_(std::move(rhs.allocation_)),
    info_(std::move(rhs.info_)) {
}

AllocatedBuffer& AllocatedBuffer::operator=(AllocatedBuffer&& rhs) noexcept {
    if (this != &rhs) {
        std::swap(buffer_, rhs.buffer_);
        std::swap(allocation_, rhs.allocation_);
        std::swap(info_, rhs.info_);
    }
    return *this;
}

const vk::Buffer& AllocatedBuffer::getBuffer() const {
    return buffer_;
}

const VmaAllocationInfo& AllocatedBuffer::getInfo() const {
    return info_;
}

void AllocatedBuffer::clear() noexcept {
    if (buffer_) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
    buffer_ = nullptr;
}
