#include <Material.h>

MaterialInstance::MaterialInstance(std::string_view name, vk::raii::Pipeline&& pipeline) :
    name_(name),
    pipeline_(std::move(pipeline)) {
}

const std::string& MaterialInstance::getName() const {
    return name_;
}

const vk::Pipeline& MaterialInstance::getPipeline() const {
    return *pipeline_;
}

Material::Material(std::string_view name, const vk::PipelineLayout& layout) {

}

std::unique_ptr<MaterialInstance> Material::createInstance(std::string_view name) const {
    return std::make_unique(name == "" ? name_ : name, builder_.buildPipeline( ));    // FIXME: we should always work with references to the raii types (api can be different, some functions requires the raii version), maybe the spec has some hints to confirm this?
}
