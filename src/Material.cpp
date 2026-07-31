#include <Material.h>

#include <cassert>

const std::string& MaterialInstance::getName() const {
    return name_;
}

const Material& MaterialInstance::getMaterial() const {
    return *material_;
}

MaterialInstance::MaterialInstance(std::string_view name, const Material& material) :
    name_(name),
    material_(&material) {
}

Material::Material(std::string_view name, const vk::raii::Device& device, const vk::raii::PipelineLayout& layout) :
    name_(name),
    device_(&device) {

    builder_.setPipelineLayout(layout);
}

const std::string& Material::getName() const {
    return name_;
}

const vk::raii::Pipeline& Material::getPipeline(components::Renderable::PrimitiveType primitiveType) const {
    buildPipeline(primitiveType);
    return pipelines_[primitiveType];
}

void Material::buildPipeline(components::Renderable::PrimitiveType primitiveType) const {
    if (pipelines_[primitiveType] != nullptr) {
        return;
    }

    switch(primitiveType) {
    case components::Renderable::points:
        builder_.setInputTopology(vk::PrimitiveTopology::ePointList);
        break;
    case components::Renderable::lines:
        builder_.setInputTopology(vk::PrimitiveTopology::eLineList);
        break;
    case components::Renderable::lineStrip:
        builder_.setInputTopology(vk::PrimitiveTopology::eLineStrip);
        break;
    case components::Renderable::triangles:
        builder_.setInputTopology(vk::PrimitiveTopology::eTriangleList);
        break;
    case components::Renderable::triangleStrip:
        builder_.setInputTopology(vk::PrimitiveTopology::eTriangleStrip);
        break;
    case components::Renderable::count:
    default:
        assert(false);
        break;
    }
    pipelines_[primitiveType] = builder_.buildPipeline(*device_);
}

std::unique_ptr<MaterialInstance> Material::createInstance(std::string_view name) const {
    return std::unique_ptr<MaterialInstance>(new MaterialInstance(name == "" ? name_ : name, *this));
}



GltfMetallicRoughness::GltfMetallicRoughness(std::string_view name, const vk::raii::Device& device, const vk::raii::PipelineLayout& layout, vk::Format colorAttachmentFormat, vk::Format depthFormat) :
    Material(name, device, layout) {

    triangleVertexShader_ = createShaderModule("shaders/colored_triangle_mesh.vert.spv", device);

    triangleFragShader_ = createShaderModule("shaders/tex_image.frag.spv", device);

    builder_.setShaders(triangleVertexShader_, triangleFragShader_);

    //builder_.setInputTopology(vk::PrimitiveTopology::eTriangleList);    // FIXME: switching pipelines is expensive, is it easier to put topology into dynamic state? then we don't need to store a bunch of pipelines per material

    builder_.setPolygonMode(vk::PolygonMode::eFill);

    builder_.setCullMode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);

    builder_.setMultisamplingNone();

    builder_.disableBlending();

    builder_.enableDepthtest(true, vk::CompareOp::eGreaterOrEqual);

    builder_.setColorAttachmentFormat(colorAttachmentFormat);
    builder_.setDepthFormat(depthFormat);
}
