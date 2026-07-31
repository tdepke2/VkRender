#pragma once

#include <components/Renderable.h>
#include <Pipelines.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vulkan/vulkan_raii.hpp>

class Material;

class MaterialInstance {
public:
    const std::string& getName() const;
    const Material& getMaterial() const;

private:
    MaterialInstance(std::string_view name, const Material& material);

    std::string name_;
    const Material* material_;

    friend Material;
};

class Material {
public:
    Material(std::string_view name, const vk::raii::Device& device, const vk::raii::PipelineLayout& layout);

    const std::string& getName() const;
    const vk::raii::Pipeline& getPipeline(components::Renderable::PrimitiveType primitiveType) const;

    void buildPipeline(components::Renderable::PrimitiveType primitiveType) const;
    std::unique_ptr<MaterialInstance> createInstance(std::string_view name = "") const;

protected:
    mutable PipelineBuilder builder_;

private:
    std::string name_;
    const vk::raii::Device* device_;
    mutable std::array<vk::raii::Pipeline, components::Renderable::count> pipelines_ = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};



class GltfMetallicRoughness : public Material {    // FIXME: store gltf mats in the gltf loader class?
    GltfMetallicRoughness(std::string_view name, const vk::raii::Device& device, const vk::raii::PipelineLayout& layout, vk::Format colorAttachmentFormat, vk::Format depthFormat);

private:
    vk::raii::ShaderModule triangleVertexShader_ = nullptr;
    vk::raii::ShaderModule triangleFragShader_ = nullptr;
};
