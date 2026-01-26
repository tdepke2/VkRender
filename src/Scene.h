#pragma once

// https://www.david-colson.com/2020/02/09/making-a-simple-ecs.html

// https://austinmorlan.com/posts/entity_component_system/

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include <ComponentArray.h>

namespace priv {
    extern unsigned int componentIdCounter;
}

using EntityId = uint64_t;

class Scene {
public:
    static constexpr size_t MAX_ENTITIES = 10;

    EntityId createEntity() {
        if (!freeEntityIndices_.empty()) {
            size_t index = freeEntityIndices_.front();
            freeEntityIndices_.pop();
            entities_[index] = makeEntityId(index, getEntitySerial(entities_[index]));
            return entities_[index];
        }
        entities_.push_back(makeEntityId(entities_.size(), 0));
        return entities_.back();
    }

    void destroyEntity(EntityId id) {
        // Mark the entity id as invalid, and increment serial.
        entities_[getEntityIndex(id)] = makeEntityId(0, getEntitySerial(id) + 1);
        freeEntityIndices_.push(getEntityIndex(id));


    }

    template<typename T>
    static unsigned int getComponentId() {
        static unsigned int componentId = priv::componentIdCounter++;
        return componentId;
    }

    template<typename T, typename... Args>
    T* addComponent(EntityId id, Args&&... args) {
        // Ensure we're not using an entity that has been deleted.
        if (entities_[getEntityIndex(id)] != id) {
            return nullptr;
        }

        unsigned int componentId = getComponentId<T>();
        if (componentArrays_.size() <= componentId) {
            componentArrays_.resize(componentId + 1);
        }
        if (componentArrays_[componentId] == nullptr) {
            componentArrays_[componentId] = std::make_unique<ComponentArray<T>>(MAX_ENTITIES);
        }



        static_cast<ComponentArray<T>*>(componentArrays_[componentId].get())->emplace(getEntityIndex(id), std::forward<Args>(args)...);
    }

    template<typename T>
    void removeComponent(EntityId id) {
        // Ensure we're not using an entity that has been deleted.
        if (entities_[getEntityIndex(id)] != id) {
            return;
        }

        unsigned int componentId = getComponentId<T>();


    }

    template<typename T>
    T* getComponent(EntityId id) {
        // Ensure we're not using an entity that has been deleted.
        if (entities_[getEntityIndex(id)] != id) {
            return nullptr;
        }


    }

private:
    static inline EntityId makeEntityId(uint32_t index, uint32_t serial) {
        return (static_cast<EntityId>(index + 1) << 32) | serial;
    }

    static inline uint32_t getEntityIndex(EntityId id) {
        return (id >> 32) - 1;
    }

    static inline uint32_t getEntitySerial(EntityId id) {
        return id;
    }

    static inline bool isEntityValid(EntityId id) {
        return (id >> 32) != 0;
    }

    std::vector<EntityId> entities_;
    std::queue<uint32_t> freeEntityIndices_;
    std::vector<std::unique_ptr<IComponentArray>> componentArrays_;
};
