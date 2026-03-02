#include <Scene.h>

#include <limits>

namespace priv {
    unsigned int componentIdCounter = 0;
}

Scene& Scene::instance() {
    static std::unique_ptr<Scene> scene = std::make_unique<Scene>(Private());
    return *scene;
}

Scene::Scene(Private) {}

EntityId Scene::createEntity() {
    if (!freeEntityIndices_.empty()) {
        auto index = freeEntityIndices_.back();
        freeEntityIndices_.pop_back();
        entities_[index].id = makeEntityId(index, getEntitySerial(entities_[index].id));
        return entities_[index].id;
    }
    entities_.emplace_back(makeEntityId(static_cast<uint32_t>(entities_.size()), 0), 0);
    return entities_.back().id;
}

void Scene::destroyEntity(EntityId id) {
    // Ensure we're not using an entity that has been deleted.
    auto& entityInfo = entities_[getEntityIndex(id)];
    assert(entityInfo.id == id);

    for (size_t i = 0; i < priv::componentIdCounter; ++i) {
        if (entityInfo.mask.test(i)) {
            componentArrays_[i]->entityDestroyed(getEntityIndex(id));
        }
    }

    // Mark the entity id as invalid, and increment serial.
    entityInfo.id = makeEntityId(std::numeric_limits<uint32_t>::max(), getEntitySerial(id) + 1);
    entityInfo.mask.reset();
    freeEntityIndices_.push_back(getEntityIndex(id));
}

void Scene::destroyAllEntities() {
    // Resetting the component arrays is tricky, as some components may access others in their destructor.
    // To make things safe, pretend that all components are gone before actually deleting them.
    for (auto& entityInfo : entities_) {
        entityInfo.mask.reset();
    }
    for (auto& componentArray : componentArrays_) {
        componentArray.reset();
    }
    freeEntityIndices_.clear();
    entities_.clear();
}

bool Scene::isEntityAlive(EntityId id) {
    return entities_[getEntityIndex(id)].id == id;
}
