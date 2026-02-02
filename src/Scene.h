#pragma once

// https://www.david-colson.com/2020/02/09/making-a-simple-ecs.html

// https://austinmorlan.com/posts/entity_component_system/

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <ComponentArray.h>

namespace priv {
    extern unsigned int componentIdCounter;
}

using EntityId = uint64_t;

class Scene {
public:
    static constexpr uint32_t MAX_ENTITIES = 40;
    static constexpr uint32_t MAX_COMPONENT_TYPES = 32;

    Scene() = default;
    Scene(const Scene& rhs) = delete;
    Scene& operator=(const Scene& rhs) = delete;

    EntityId createEntity() {
        if (!freeEntityIndices_.empty()) {
            auto index = freeEntityIndices_.back();
            freeEntityIndices_.pop_back();
            entities_[index].id = makeEntityId(index, getEntitySerial(entities_[index].id));
            return entities_[index].id;
        }
        entities_.emplace_back(makeEntityId(static_cast<uint32_t>(entities_.size()), 0), 0);
        return entities_.back().id;
    }

    void destroyEntity(EntityId id) {
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

    // The pointer may become invalid when removing any component of the same type (or destroying an entity with the component type).
    // Returns nullptr if entity not found.
    template<typename T, typename... Args>
    T* assignComponent(EntityId id, Args&&... args) {
        // Ensure we're not using an entity that has been deleted.
        auto& entityInfo = entities_[getEntityIndex(id)];
        assert(entityInfo.id == id);

        if (componentArrays_[getComponentId<T>()] == nullptr) {
            componentArrays_[getComponentId<T>()] = std::make_unique<ComponentArray<T>>(MAX_ENTITIES);
        }

        entityInfo.mask.set(getComponentId<T>());
        return &*(getComponentArray<T>()->assign(getEntityIndex(id), std::forward<Args>(args)...));
    }

    // Does nothing if entity not found or component already removed.
    template<typename T>
    void removeComponent(EntityId id) {
        // Ensure we're not using an entity that has been deleted.
        auto& entityInfo = entities_[getEntityIndex(id)];
        assert(entityInfo.id == id);

        if (entityInfo.mask.test(getComponentId<T>())) {
            entityInfo.mask.reset(getComponentId<T>());
            getComponentArray<T>()->remove(getEntityIndex(id));
        }
    }

    // See assignComponent() notes about pointer validity.
    // Returns nullptr if entity or component not found.
    template<typename T>
    T* accessComponent(EntityId id) {
        // Ensure we're not using an entity that has been deleted.
        auto& entityInfo = entities_[getEntityIndex(id)];
        assert(entityInfo.id == id);

        if (entityInfo.mask.test(getComponentId<T>())) {
            return &(*getComponentArray<T>())[getEntityIndex(id)];
        } else {
            return nullptr;
        }
    }

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
