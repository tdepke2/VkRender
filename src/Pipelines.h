#pragma once

#include <Common.h>

vk::raii::ShaderModule createShaderModule(const std::string& filename, const vk::raii::Device& device);

class PipelineBuilder {
public:
    PipelineBuilder();

    void clear();

    vk::raii::Pipeline buildPipeline(const vk::raii::Device& device);

    void setPipelineLayout(vk::PipelineLayout pipelineLayout);

    void setShaders(vk::ShaderModule vertexShader, vk::ShaderModule fragmentShader);

    void setInputTopology(vk::PrimitiveTopology topology);

    void setPolygonMode(vk::PolygonMode mode);

    void setCullMode(vk::CullModeFlags cullMode, vk::FrontFace frontFace);

    void setMultisamplingNone();

    void disableBlending();

    void enableBlendingAdditive();

    void enableBlendingAlphablend();

    void setColorAttachmentFormat(vk::Format format);

    void setDepthFormat(vk::Format format);

    void disableDepthtest();

    void enableDepthtest(bool depthWriteEnable, vk::CompareOp op);

private:
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages_;

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly_;
    vk::PipelineRasterizationStateCreateInfo rasterizer_;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment_;
    vk::PipelineMultisampleStateCreateInfo multisampling_;
    vk::PipelineLayout pipelineLayout_;
    vk::PipelineDepthStencilStateCreateInfo depthStencil_;
    vk::PipelineRenderingCreateInfo renderInfo_;
    vk::Format colorAttachmentformat_;
};
