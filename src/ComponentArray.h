#pragma once

#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class IComponentArray {
public:
    virtual ~IComponentArray() = default;
    virtual void entityDestroyed(uint32_t entityIndex) = 0;
};

template<typename T>
class ComponentArray : public IComponentArray {
public:
    // https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp
    struct Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = T;
        using pointer           = T*;
        using reference         = T&;

        Iterator(const uint32_t* keyBase, pointer valueBase, uint32_t offset) :
            keyPtr_(keyBase + offset),
            valuePtr_(valueBase + offset) {
        }

        reference operator*() const {
            return *valuePtr_;
        }
        pointer operator->() const {
            return valuePtr_;
        }

        // Prefix increment.
        Iterator& operator++() {
            ++keyPtr_;
            ++valuePtr_;
            return *this;
        }

        // Postfix increment.
        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        // The components and array to match them to entity index are stored in
        // separate containers, so there isn't a nice way for this iterator to
        // return a std::pair with both. This getter function should be simple
        // enough to solve this.
        uint32_t getEntityIndex() const {
            return *keyPtr_;
        }

        // Provided for the SceneView so that it can iterate the ComponentArray
        // using its own methods.
        const uint32_t* getEntityIndexPtr() const {
            return keyPtr_;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.valuePtr_ == b.valuePtr_;
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return a.valuePtr_ != b.valuePtr_;
        }

    private:
        const uint32_t* keyPtr_;
        pointer valuePtr_;
    };

    ComponentArray(uint32_t capacity) :
        components_(std::make_unique_for_overwrite<T[]>(capacity)),
        entityIndexToComponent_(capacity, std::numeric_limits<uint32_t>::max()),
        componentToEntityIndex_(),
        capacity_(capacity),
        size_(0) {
    }
    virtual ~ComponentArray() = default;
    ComponentArray(const ComponentArray& rhs) = delete;
    ComponentArray& operator=(const ComponentArray& rhs) = delete;

    void entityDestroyed(uint32_t entityIndex) override {
        remove(entityIndex);
    }

    template<typename... Args>
    Iterator assign(uint32_t entityIndex, Args&&... args) {
        static_assert(
            std::is_constructible<T, Args...>::value,
            "Cannot construct T from the given arguments."
        );

        auto existingComponentIndex = entityIndexToComponent_[entityIndex];
        if (existingComponentIndex != std::numeric_limits<uint32_t>::max()) {
            components_[existingComponentIndex] = {std::forward<Args>(args)...};
            return {componentToEntityIndex_.data(), components_.get(), existingComponentIndex};
        }

        components_[size_] = {std::forward<Args>(args)...};
        componentToEntityIndex_.push_back(entityIndex);
        entityIndexToComponent_[entityIndex] = size_;
        ++size_;
        return {componentToEntityIndex_.data(), components_.get(), size_ - 1};
    }

    bool remove(uint32_t entityIndex) {
        auto componentIndex = entityIndexToComponent_[entityIndex];
        if (componentIndex == std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        entityIndexToComponent_[entityIndex] = std::numeric_limits<uint32_t>::max();
        if (componentIndex + 1 < size_) {
            // If we are removing a middle component, move the last component into the empty space.
            entityIndexToComponent_[componentToEntityIndex_.back()] = componentIndex;
            componentToEntityIndex_[componentIndex] = componentToEntityIndex_.back();
            components_[componentIndex] = components_[size_ - 1];
        }

        componentToEntityIndex_.pop_back();
        --size_;
        return true;
    }

    T& at(uint32_t entityIndex) {
        auto componentIndex = entityIndexToComponent_.at(entityIndex);
        if (componentIndex == std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range("ComponentArray::at(): component does not exist for entity at index " + std::to_string(entityIndex) + ".");
        }
        return components_[componentIndex];
    }

    T& operator[](uint32_t entityIndex) {
        return components_[entityIndexToComponent_[entityIndex]];
    }

    bool hasEntity(uint32_t entityIndex) const {
        return entityIndexToComponent_[entityIndex] != std::numeric_limits<uint32_t>::max();
    }

    uint32_t size() const {
        return size_;
    }

    uint32_t capacity() const {
        return capacity_;
    }

    Iterator begin() {
        return {componentToEntityIndex_.data(), components_.get(), 0};
    }

    Iterator end() {
        return {componentToEntityIndex_.data(), components_.get(), size_};
    }

private:
    std::unique_ptr<T[]> components_;
    std::vector<uint32_t> entityIndexToComponent_;
    std::vector<uint32_t> componentToEntityIndex_;
    uint32_t capacity_;
    uint32_t size_;
};
