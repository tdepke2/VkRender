#include <Scene.h>

#include <algorithm>



#include <iostream>




namespace priv {
    unsigned int componentIdCounter = 0;
}

Scene& Scene::instance() {
    static std::unique_ptr<Scene> scene = std::make_unique<Scene>(Private());
    return *scene;
}

Scene::Scene(Private) {}

Scene::~Scene() {
    destroyAllEntities();
}

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

EntityId Scene::createEntityChild(EntityId parent) {
    // Ensure we're not using an entity that has been deleted.
    auto& parentInfo = entities_[getEntityIndex(parent)];
    assert(parentInfo.id == parent);

    EntityId id = createEntity();
    hierarchy_.assign(getEntityIndex(id), parent, std::vector<EntityId>{});

    if (hierarchy_.hasEntity(getEntityIndex(parent))) {
        hierarchy_[getEntityIndex(parent)].children.push_back(id);
    } else {
        hierarchy_.assign(getEntityIndex(parent), makeEntityId(INVALID_ENTITY_INDEX, 0), std::vector<EntityId>{id});
    }

    return id;
}

void Scene::destroyEntity(EntityId id) {
    std::cout << "Scene::destroyEntity() for " << id << "\n";
    // Ensure we're not using an entity that has been deleted.
    auto& entityInfo = entities_[getEntityIndex(id)];
    assert(entityInfo.id == id);

    // Destroy all of the children (if any) in a depth-last order.
    if (hierarchy_.hasEntity(getEntityIndex(id))) {
        // Steal the ParentInfo as the child will make changes to `hierarchy_` and we can't hold on to a reference to it after that happens.
        auto parentInfo = std::move(hierarchy_[getEntityIndex(id)]);
        for (auto child : parentInfo.children) {
            hierarchy_[getEntityIndex(child)].parent = INVALID_ENTITY_INDEX;
            destroyEntity(child);
        }

        if (parentInfo.parent != INVALID_ENTITY_INDEX) {
            std::cout << "Informing parent " << parentInfo.parent << " that child " << id << " is gone.\n";
            auto& parentsChildren = hierarchy_[getEntityIndex(parentInfo.parent)].children;
            parentsChildren.erase(std::find(parentsChildren.begin(), parentsChildren.end(), id));
        }
        hierarchy_.remove(getEntityIndex(id));
    }

    for (size_t i = 0; i < priv::componentIdCounter; ++i) {
        if (entityInfo.mask.test(i)) {
            componentArrays_[i]->entityDestroyed(getEntityIndex(id));
        }
    }

    // Mark the entity id as invalid, and increment serial.
    entityInfo.id = makeEntityId(INVALID_ENTITY_INDEX, getEntitySerial(id) + 1);
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
    // FIXME: need to reset hierarchy_
}

bool Scene::isEntityAlive(EntityId id) {
    return entities_[getEntityIndex(id)].id == id;
}

uint32_t Scene::getEntitiesCount() const {
    return entities_.size() - freeEntityIndices_.size();
}
