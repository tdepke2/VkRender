#include <components/Renderable.h>
#include <Scene.h>

namespace components {

Renderable::Renderable(const MeshAsset& mesh) :
    mesh_(&mesh) {
}

const MeshAsset& Renderable::getMesh() const {
    return *mesh_;
}

} // namespace components
