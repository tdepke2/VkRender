#pragma once

#include <ComponentArray.h>
#include <Scene.h>

#include <algorithm>
#include <bitset>
#include <utility>

/**
 * Provides iteration of the entities in a scene.
 * 
 * A `SceneView` can be constructed with the component types that all entities
 * must have, or no component types (which will iterate all entities). This
 * object and its iterators are valid until the `Scene` is modified in some way
 * (adding/removing components or entities). Think of it like the `std::vector`
 * iterator rules.
 */
template<typename... ComponentTypes>
class SceneView {
public:
    struct Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        //using difference_type   = std::ptrdiff_t;    // FIXME: not sure what this would be, it depends on anyComponents_?
        using value_type        = EntityId;
        using pointer           = EntityId*;
        using reference         = EntityId&;

        Iterator(Scene* scene, const std::bitset<Scene::MAX_COMPONENT_TYPES>& componentMask, const Scene::EntityInfo* start, const Scene::EntityInfo* end) :
            scene_(scene),
            componentMask_(componentMask),
            allEntities_(true) {

            entitiesPtr_ = start;
            entitiesEnd_ = end;
        }

        Iterator(Scene* scene, const std::bitset<Scene::MAX_COMPONENT_TYPES>& componentMask, const uint32_t* start, const uint32_t* end) :
            scene_(scene),
            componentMask_(componentMask),
            allEntities_(false) {

            entityIndexPtr_ = start;
            entityIndexEnd_ = end;
        }

        EntityId operator*() const {
            if (allEntities_) {
                return entitiesPtr_->id;
            } else {
                return scene_->entities_[*entityIndexPtr_].id;
            }
        }

        // Prefix increment.
        Iterator& operator++() {
            if (allEntities_) {
                do {
                    ++entitiesPtr_;
                } while (entitiesPtr_ < entitiesEnd_ && !Scene::isEntityValid(entitiesPtr_->id));
            } else {
                do {
                    ++entityIndexPtr_;
                } while (entityIndexPtr_ < entityIndexEnd_ && (componentMask_ != (componentMask_ & scene_->entities_[*entityIndexPtr_].mask)));
            }
            return *this;
        }

        // Postfix increment.
        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.entitiesPtr_ == b.entitiesPtr_;
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return a.entitiesPtr_ != b.entitiesPtr_;
        }

    private:
        Scene* scene_ = nullptr;
        std::bitset<Scene::MAX_COMPONENT_TYPES> componentMask_;
        bool allEntities_ = false;
        union {
            // These pointers will be the same size in memory, so we don't need to pick the active one when comparing pointers with another iterator.
            const Scene::EntityInfo* entitiesPtr_;
            const uint32_t* entityIndexPtr_;
        };
        union {
            const Scene::EntityInfo* entitiesEnd_;
            const uint32_t* entityIndexEnd_;
        };
    };

    SceneView(Scene& scene) :
        scene_(&scene) {

        if constexpr (sizeof...(ComponentTypes) == 0) {
            allEntities_ = true;
        } else {
            allEntities_ = false;

            // The first element will be unused, we need it so that this still compiles when no ComponentTypes are provided.
            std::pair<uint32_t, unsigned int> countAndId[] = {{0, 0}, getComponentCountAndId<ComponentTypes>()...};
            for (size_t i = 1; i < std::size(countAndId); ++i) {
                componentMask_.set(countAndId[i].second);
            }

            // Find the component id of the array with the least number of items.
            unsigned int smallestComponentArrayId = std::min_element(countAndId + 1, countAndId + std::size(countAndId))->second;

            (setEntityIndexRange<ComponentTypes>(smallestComponentArrayId), ...);
        }
    }

    Iterator begin() {
        if (allEntities_) {
            // Find the first entity that has a valid id.
            const Scene::EntityInfo* firstEntity = scene_->entities_.data();
            const Scene::EntityInfo* pastLastEntity = firstEntity + scene_->entities_.size();
            while (firstEntity != pastLastEntity && !Scene::isEntityValid(firstEntity->id)) {
                ++firstEntity;
            }
            return {scene_, componentMask_, firstEntity, pastLastEntity};
        } else {
            // Find the first entity that matches the component mask (it will have a valid id).
            const uint32_t* firstEntityIndex = entityIndexBegin_;
            while (firstEntityIndex != entityIndexEnd_ && (componentMask_ != (componentMask_ & scene_->entities_[*firstEntityIndex].mask))) {
                ++firstEntityIndex;
            }
            return {scene_, componentMask_, firstEntityIndex, entityIndexEnd_};
        }
    }

    Iterator end() {
        if (allEntities_) {
            const Scene::EntityInfo* pastLastEntity = scene_->entities_.data() + scene_->entities_.size();
            return {scene_, componentMask_, pastLastEntity, pastLastEntity};
        } else {
            return {scene_, componentMask_, entityIndexEnd_, entityIndexEnd_};
        }
    }

private:
    template<typename T>
    std::pair<uint32_t, unsigned int> getComponentCountAndId() {
        uint32_t count = 0;
        // There may not yet be an array entry that exists.
        if (scene_->componentArrays_[Scene::getComponentId<T>()] != nullptr) {
            count = scene_->getComponentArray<T>()->size();
        }
        return {count, Scene::getComponentId<T>()};
    }

    template<typename T>
    void setEntityIndexRange(unsigned int smallestComponentArrayId) {
        //std::cout << "setEntityIndexRange() called for " << typeid(T).name() << "\n";
        if (Scene::getComponentId<T>() == smallestComponentArrayId) {
            if (scene_->componentArrays_[Scene::getComponentId<T>()] != nullptr) {
                entityIndexBegin_ = scene_->getComponentArray<T>()->begin().getEntityIndexPtr();
                entityIndexEnd_ = scene_->getComponentArray<T>()->end().getEntityIndexPtr();
            } else {
                entityIndexBegin_ = nullptr;
                entityIndexEnd_ = nullptr;
            }
        }
    }

    Scene* scene_ = nullptr;
    std::bitset<Scene::MAX_COMPONENT_TYPES> componentMask_;
    bool allEntities_ = false;
    const uint32_t* entityIndexBegin_ = nullptr;
    const uint32_t* entityIndexEnd_ = nullptr;
};
