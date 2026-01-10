#include <Descriptors.h>

void DescriptorLayoutBuilder::addBinding(uint32_t binding, vk::DescriptorType type) {
    vk::DescriptorSetLayoutBinding newBinding = {
        .binding = binding,
        .descriptorType = type,
        .descriptorCount = 1
    };

    bindings.push_back(newBinding);
}

void DescriptorLayoutBuilder::clear() {
    bindings.clear();
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::build(const vk::raii::Device& device, vk::ShaderStageFlags shaderStages, void* pNext, vk::DescriptorSetLayoutCreateFlags flags) {
    for (auto& b : bindings) {
        b.stageFlags |= shaderStages;
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {
        .pNext = pNext,
        .flags = flags,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return {device, layoutInfo};
}

void DescriptorWriter::writeImage(uint32_t binding, vk::ImageView image, vk::Sampler sampler, vk::ImageLayout layout, vk::DescriptorType type) {
    imageInfos.push_back({
        .sampler = sampler,
        .imageView = image,
        .imageLayout = layout
    });

    vk::WriteDescriptorSet write = {
        .dstSet = nullptr,    // Left empty for now until we need to write it.
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = &imageInfos.back()
    };

    writes.push_back(write);
}

void DescriptorWriter::writeBuffer(uint32_t binding, vk::Buffer buffer, size_t size, size_t offset, vk::DescriptorType type) {
    bufferInfos.push_back({
        .buffer = buffer,
        .offset = offset,
        .range = size
    });

    vk::WriteDescriptorSet write = {
        .dstSet = nullptr,    // Left empty for now until we need to write it.
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &bufferInfos.back()
    };

    writes.push_back(write);
}

void DescriptorWriter::clear() {
    imageInfos.clear();
    writes.clear();
    bufferInfos.clear();
}

void DescriptorWriter::updateSet(const vk::raii::Device& device, vk::DescriptorSet set) {
    for (auto& write : writes) {
        write.dstSet = set;
    }

    device.updateDescriptorSets({ static_cast<uint32_t>(writes.size()), writes.data() }, {});
}

void DescriptorAllocator::initPool(const vk::raii::Device& device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (const auto& ratio : poolRatios) {
        poolSizes.push_back({
            .type = ratio.type,
            .descriptorCount = static_cast<uint32_t>(ratio.ratio * maxSets)
        });
    }

    vk::DescriptorPoolCreateInfo poolInfo = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    pool = vk::raii::DescriptorPool(device, poolInfo);
}

void DescriptorAllocator::clearDescriptors() {
    pool.reset();
}

void DescriptorAllocator::destroyPool() {
    pool.clear();
}

vk::raii::DescriptorSet DescriptorAllocator::allocate(const vk::raii::Device& device, vk::DescriptorSetLayout layout) {
    vk::DescriptorSetAllocateInfo allocInfo = {
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };

    return std::move(device.allocateDescriptorSets(allocInfo).front());    // FIXME: we don't necessarily need to use the raii type here (the memory is managed by the pool). The vk tutorial does this though, what is the benefit? Note the eFreeDescriptorSet flag is needed for the pool if we do use the raii type.
}

void DescriptorAllocatorGrowable::init(const vk::raii::Device& device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios) {
    ratios_.clear();

    for (const auto& r : poolRatios) {
        ratios_.push_back(r);
    }

    readyPools_.emplace_back(createPool(device, initialSets, poolRatios));

    setsPerPool_ = static_cast<uint32_t>(initialSets * 1.5);
}

void DescriptorAllocatorGrowable::clearPools() {
    for (const auto& p : readyPools_) {
        p.reset();
    }
    for (auto& p : fullPools_) {
        p.reset();
        readyPools_.emplace_back(nullptr);
        p.swap(readyPools_.back());
    }
    fullPools_.clear();
}

void DescriptorAllocatorGrowable::destroyPools() {
    readyPools_.clear();
    fullPools_.clear();
}

vk::raii::DescriptorSet DescriptorAllocatorGrowable::allocate(const vk::raii::Device& device, vk::DescriptorSetLayout layout, void* pNext) {
    vk::raii::DescriptorPool poolToUse = getPool(device);

    vk::DescriptorSetAllocateInfo allocInfo = {
        .pNext = pNext,
        .descriptorPool = poolToUse,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };

    vk::raii::DescriptorSet newDescriptor(nullptr);
    try {
        newDescriptor = std::move(device.allocateDescriptorSets(allocInfo).front());
    } catch(const vk::SystemError& /*e*/) {
        // Allocation failed (pool is probably full), so try again with a new one.
        fullPools_.emplace_back(std::move(poolToUse));

        poolToUse = getPool(device);
        allocInfo.descriptorPool = poolToUse;

        newDescriptor = std::move(device.allocateDescriptorSets(allocInfo).front());
    }

    readyPools_.emplace_back(std::move(poolToUse));

    return newDescriptor;
}

vk::raii::DescriptorPool DescriptorAllocatorGrowable::getPool(const vk::raii::Device& device) {
    vk::raii::DescriptorPool newPool(nullptr);
    if (!readyPools_.empty()) {
        newPool.swap(readyPools_.back());
        readyPools_.pop_back();
    } else {
        newPool = createPool(device, setsPerPool_, ratios_);

        setsPerPool_ = static_cast<uint32_t>(setsPerPool_ * 1.5);
        if (setsPerPool_ > 4092) {
            setsPerPool_ = 4092;
        }
    }

    return newPool;
}

vk::raii::DescriptorPool DescriptorAllocatorGrowable::createPool(const vk::raii::Device& device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (const auto& ratio : poolRatios) {
        poolSizes.push_back({
            .type = ratio.type,
            .descriptorCount = static_cast<uint32_t>(ratio.ratio * setCount)
        });
    }

    vk::DescriptorPoolCreateInfo poolInfo = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = setCount,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    return {device, poolInfo};
}
