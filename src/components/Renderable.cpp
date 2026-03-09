#include <components/Renderable.h>
#include <Scene.h>

#include <cassert>

namespace components {

Renderable* Renderable::addToScene(EntityId id, const MeshAsset& mesh) {
    assert(Scene::instance().access<Renderable>(id) == nullptr);
    return Scene::instance().assign<Renderable>(id, Private(), mesh);
}

Renderable::Renderable(Private, const MeshAsset& mesh) :
    mesh_(&mesh) {
}

const MeshAsset& Renderable::getMesh() const {
    return *mesh_;
}

} // namespace components
