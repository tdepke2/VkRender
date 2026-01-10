#pragma once

#include <deque>
#include <Common.h>

class DescriptorLayoutBuilder {
public:
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

    void addBinding(uint32_t binding, vk::DescriptorType type);
    void clear();
    vk::raii::DescriptorSetLayout build(const vk::raii::Device& device, vk::ShaderStageFlags shaderStages, void* pNext = nullptr, vk::DescriptorSetLayoutCreateFlags flags = {});
};

class DescriptorWriter {
public:
    std::deque<vk::DescriptorImageInfo> imageInfos;
    std::deque<vk::DescriptorBufferInfo> bufferInfos;
    std::vector<vk::WriteDescriptorSet> writes;

    void writeImage(uint32_t binding, vk::ImageView image, vk::Sampler sampler, vk::ImageLayout layout, vk::DescriptorType type);
    void writeBuffer(uint32_t binding, vk::Buffer buffer, size_t size, size_t offset, vk::DescriptorType type);

    void clear();
    void updateSet(const vk::raii::Device& device, vk::DescriptorSet set);
};

class DescriptorAllocator {
public:
    struct PoolSizeRatio {
        vk::DescriptorType type;
        float ratio;
    };

    vk::raii::DescriptorPool pool = nullptr;

    void initPool(const vk::raii::Device& device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clearDescriptors();
    void destroyPool();

    vk::raii::DescriptorSet allocate(const vk::raii::Device& device, vk::DescriptorSetLayout layout);
};

class DescriptorAllocatorGrowable {
public:
    struct PoolSizeRatio {
        vk::DescriptorType type;
        float ratio;
    };

    void init(const vk::raii::Device& device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios);
    void clearPools();
    void destroyPools();

    vk::raii::DescriptorSet allocate(const vk::raii::Device& device, vk::DescriptorSetLayout layout, void* pNext = nullptr);

private:
    vk::raii::DescriptorPool getPool(const vk::raii::Device& device);
    vk::raii::DescriptorPool createPool(const vk::raii::Device& device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios);

    std::vector<PoolSizeRatio> ratios_;
    std::vector<vk::raii::DescriptorPool> fullPools_;
    std::vector<vk::raii::DescriptorPool> readyPools_;
    uint32_t setsPerPool_;
};
