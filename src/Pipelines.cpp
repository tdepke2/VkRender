#include <Pipelines.h>

#include <fstream>

vk::raii::ShaderModule createShaderModule(const std::string& filename, const vk::raii::Device& device) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("\"" + filename + "\": unable to open shader file.");
    }

    auto fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0);

    std::vector<uint32_t> buffer((fileSize + sizeof(uint32_t) - 1) / sizeof(uint32_t));

    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    vk::ShaderModuleCreateInfo createInfo = {
        .codeSize = buffer.size() * sizeof(uint32_t),
        .pCode = buffer.data(),
    };

    return {device, createInfo};
}

PipelineBuilder::PipelineBuilder() {
    clear();
}

void PipelineBuilder::clear() {
    shaderStages_.clear();

    inputAssembly_ = {};

    rasterizer_ = {};

    colorBlendAttachment_ = {};

    multisampling_ = {};

    pipelineLayout_ = nullptr;

    depthStencil_ = {};

    renderInfo_ = {};

    colorAttachmentformat_ = vk::Format::eR8G8B8A8Unorm;
}

vk::raii::Pipeline PipelineBuilder::buildPipeline(const vk::raii::Device& device) {
    vk::PipelineViewportStateCreateInfo viewportState = {
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending = {
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment_
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};

    vk::DynamicState state[] = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicInfo = {
        .dynamicStateCount = 2,
        .pDynamicStates = &state[0],
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo = {
        .pNext = &renderInfo_,
        .stageCount = static_cast<uint32_t>(shaderStages_.size()),
        .pStages = shaderStages_.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly_,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer_,
        .pMultisampleState = &multisampling_,
        .pDepthStencilState = &depthStencil_,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicInfo,
        .layout = pipelineLayout_,
    };

    return {device, nullptr, pipelineInfo};
}

void PipelineBuilder::setPipelineLayout(vk::PipelineLayout pipelineLayout) {
    pipelineLayout_ = pipelineLayout;
}

void PipelineBuilder::setShaders(vk::ShaderModule vertexShader, vk::ShaderModule fragmentShader) {
    shaderStages_.clear();

    shaderStages_.push_back(vk::PipelineShaderStageCreateInfo {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertexShader,
        .pName = "main",
    });

    shaderStages_.push_back(vk::PipelineShaderStageCreateInfo {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragmentShader,
        .pName = "main",
    });
}

void PipelineBuilder::setInputTopology(vk::PrimitiveTopology topology) {
    inputAssembly_.topology = topology;
    inputAssembly_.primitiveRestartEnable = vk::False;
}

void PipelineBuilder::setPolygonMode(vk::PolygonMode mode) {
    rasterizer_.polygonMode = mode;
    rasterizer_.lineWidth = 1.0f;
}

void PipelineBuilder::setCullMode(vk::CullModeFlags cullMode, vk::FrontFace frontFace) {
    rasterizer_.cullMode = cullMode;
    rasterizer_.frontFace = frontFace;
}

void PipelineBuilder::setMultisamplingNone() {
    multisampling_.sampleShadingEnable = vk::False;
    multisampling_.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling_.minSampleShading = 1.0f;
    multisampling_.pSampleMask = nullptr;
    multisampling_.alphaToCoverageEnable = vk::False;
    multisampling_.alphaToOneEnable = vk::False;
}

void PipelineBuilder::disableBlending() {
    colorBlendAttachment_.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    colorBlendAttachment_.blendEnable = vk::False;
}

void PipelineBuilder::enableBlendingAdditive() {
    colorBlendAttachment_.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    colorBlendAttachment_.blendEnable = vk::True;
    colorBlendAttachment_.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment_.dstColorBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment_.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment_.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment_.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment_.alphaBlendOp = vk::BlendOp::eAdd;
}

void PipelineBuilder::enableBlendingAlphablend() {
    colorBlendAttachment_.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    colorBlendAttachment_.blendEnable = vk::True;
    colorBlendAttachment_.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment_.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment_.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment_.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment_.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment_.alphaBlendOp = vk::BlendOp::eAdd;
}

void PipelineBuilder::setColorAttachmentFormat(vk::Format format) {
    colorAttachmentformat_ = format;
    renderInfo_.colorAttachmentCount = 1;
    renderInfo_.pColorAttachmentFormats = &colorAttachmentformat_;
}

void PipelineBuilder::setDepthFormat(vk::Format format) {
    renderInfo_.depthAttachmentFormat = format;
}

void PipelineBuilder::disableDepthtest() {
    depthStencil_.depthTestEnable = vk::False;
    depthStencil_.depthWriteEnable = vk::False;
    depthStencil_.depthCompareOp = vk::CompareOp::eNever;
    depthStencil_.depthBoundsTestEnable = vk::False;
    depthStencil_.stencilTestEnable = vk::False;
    depthStencil_.front = {};
    depthStencil_.back = {};
    depthStencil_.minDepthBounds = 0.0f;
    depthStencil_.maxDepthBounds = 1.0f;
}

void PipelineBuilder::enableDepthtest(bool depthWriteEnable, vk::CompareOp op) {
    depthStencil_.depthTestEnable = vk::True;
    depthStencil_.depthWriteEnable = depthWriteEnable;
    depthStencil_.depthCompareOp = op;
    depthStencil_.depthBoundsTestEnable = vk::False;
    depthStencil_.stencilTestEnable = vk::False;
    depthStencil_.front = {};
    depthStencil_.back = {};
    depthStencil_.minDepthBounds = 0.0f;
    depthStencil_.maxDepthBounds = 1.0f;
}
