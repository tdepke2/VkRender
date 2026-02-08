#pragma once

#include <ComponentArray.h>

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace priv {
    extern unsigned int componentIdCounter;
}

using EntityId = uint64_t;

/**
 * Storage for entities and components (entity component system).
 * 
 * An entity is just a simple identifier. Different types of components can be
 * assigned to the entity depending on what kind of data it needs. The
 * components are stored in contiguous arrays so that iterating them is very
 * fast. See `ComponentArray` for more details. Also see `SceneView` for
 * iterating entities within the scene.
 * 
 * Based on the following implementations:
 * https://www.david-colson.com/2020/02/09/making-a-simple-ecs.html
 * https://austinmorlan.com/posts/entity_component_system/
 */
class Scene {
private:
    struct Private {
        explicit Private() = default;
    };

public:
    static constexpr uint32_t MAX_ENTITIES = IComponentArray::MAX_SIZE;    // FIXME: throw exception if we create and entity and this is exceeded?
    static constexpr uint32_t MAX_COMPONENT_TYPES = 32;

    static Scene& instance();
    Scene(Private);
    ~Scene() = default;
    Scene(const Scene& rhs) = delete;
    Scene(Scene&& rhs) noexcept = delete;
    Scene& operator=(const Scene& rhs) = delete;
    Scene& operator=(Scene&& rhs) noexcept = delete;

    EntityId createEntity();
    void destroyEntity(EntityId id);
    void destroyAllEntities();

    // The pointer may become invalid when removing any component of the same type (or destroying an entity with the component type).
    template<typename T, typename... Args>
    T* assignComponent(EntityId id, Args&&... args);

    // Does nothing if component already removed.
    template<typename T>
    void removeComponent(EntityId id);

    // See assignComponent() notes about pointer validity.
    // Returns nullptr if component not found.
    template<typename T>
    T* accessComponent(EntityId id);

private:
    struct EntityInfo {
        EntityId id;
        std::bitset<MAX_COMPONENT_TYPES> mask;
    };

    // The entity id is composed of an index (into the vector) and serial. When
    // an entity is destroyed, the index can be reused to make a new entity but
    // the serial will increment. This ensures unique ids for entities until we
    // wrap around the 32-bit integer.
    static inline EntityId makeEntityId(uint32_t index, uint32_t serial) {
        return (static_cast<EntityId>(index + 1) << 32) | serial;
    }

    static inline uint32_t getEntityIndex(EntityId id) {
        return static_cast<uint32_t>((id >> 32) - 1);
    }

    static inline uint32_t getEntitySerial(EntityId id) {
        return static_cast<uint32_t>(id);
    }

    static inline bool isEntityValid(EntityId id) {
        return (id >> 32) != 0;
    }

    template<typename T>
    static unsigned int getComponentId() {
        static unsigned int componentId = priv::componentIdCounter++;
        return componentId;
    }

    template<typename T>
    inline ComponentArray<T>* getComponentArray() {
        return static_cast<ComponentArray<T>*>(componentArrays_[getComponentId<T>()].get());
    }

    std::vector<EntityInfo> entities_;
    std::vector<uint32_t> freeEntityIndices_;
    std::array<std::unique_ptr<IComponentArray>, MAX_COMPONENT_TYPES> componentArrays_;

    template<typename... ComponentTypes>
    friend class SceneView;
};

template<typename T, typename... Args>
T* Scene::assignComponent(EntityId id, Args&&... args) {
    // Ensure we're not using an entity that has been deleted.
    auto& entityInfo = entities_[getEntityIndex(id)];
    assert(entityInfo.id == id);

    if (componentArrays_[getComponentId<T>()] == nullptr) {
        componentArrays_[getComponentId<T>()] = std::make_unique<ComponentArray<T>>();
    }

    entityInfo.mask.set(getComponentId<T>());
    return &*(getComponentArray<T>()->assign(getEntityIndex(id), std::forward<Args>(args)...));
}

template<typename T>
void Scene::removeComponent(EntityId id) {
    // Ensure we're not using an entity that has been deleted.
    auto& entityInfo = entities_[getEntityIndex(id)];
    assert(entityInfo.id == id);

    if (entityInfo.mask.test(getComponentId<T>())) {
        entityInfo.mask.reset(getComponentId<T>());
        getComponentArray<T>()->remove(getEntityIndex(id));
    }
}

template<typename T>
T* Scene::accessComponent(EntityId id) {
    // Ensure we're not using an entity that has been deleted.
    auto& entityInfo = entities_[getEntityIndex(id)];
    assert(entityInfo.id == id);

    if (entityInfo.mask.test(getComponentId<T>())) {
        return &(*getComponentArray<T>())[getEntityIndex(id)];
    } else {
        return nullptr;
    }
}
