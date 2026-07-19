#pragma once

#include <Pipelines.h>

#include <memory>
#include <string>
#include <string_view>
#include <vulkan/vulkan_raii.hpp>

class MaterialInstance {
public:
    MaterialInstance(std::string_view name, vk::raii::Pipeline&& pipeline);
    const std::string& getName() const;
    const vk::Pipeline& getPipeline() const;

private:
    std::string name_;
    vk::raii::Pipeline pipeline_ = nullptr;
};

class Material {
public:
    Material(std::string_view name, const vk::PipelineLayout& layout);
    virtual ~Material() = default;

    std::unique_ptr<MaterialInstance> createInstance(std::string_view name = "") const;
protected:
    PipelineBuilder builder_;

private:
    std::string name_;
};
