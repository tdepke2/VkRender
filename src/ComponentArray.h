#pragma once

#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class IComponentArray {
public:
    static constexpr uint32_t MAX_SIZE = 100;

    virtual ~IComponentArray() = default;
    virtual void entityDestroyed(uint32_t entityIndex) = 0;
};

/**
 * Contiguous array of component data.
 * 
 * The array is initialized with a fixed capacity so that we don't have to
 * allocate more memory and move stuff around if space runs out. The downside is
 * there is a limit on the number of entities we have in the scene. Note that
 * the component type should be move assignable, but this isn't required.
 * 
 * An alternative implementation I found of this called it a `ComponentPool`, it
 * dynamically allocated the bytes for component data and was not a template
 * class. It's a less type safe way to do things but does avoid some template
 * annoyances and virtual calls. I also could not figure out how to allocate
 * memory like this and still conform to memory alignment. Maybe it's as simple
 * as adding an offset when addressing the bytes of the component data so the
 * memory address modulus the alignment equals zero?
 */
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

    ComponentArray() :
        entityIndexToComponent_(MAX_SIZE, std::numeric_limits<uint32_t>::max()),
        componentToEntityIndex_() {
    }
    virtual ~ComponentArray() {
        for (uint32_t i = 0; i < size_; ++i) {
            std::destroy_at(getComponent(i));
        }
    }
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
            *getComponent(existingComponentIndex) = T(std::forward<Args>(args)...);
            return {componentToEntityIndex_.data(), getComponent(0), existingComponentIndex};
        }

        ::new (data_ + size_ * sizeof(T)) T(std::forward<Args>(args)...);
        componentToEntityIndex_.push_back(entityIndex);
        entityIndexToComponent_[entityIndex] = size_;
        ++size_;
        return {componentToEntityIndex_.data(), getComponent(0), size_ - 1};
    }

    bool remove(uint32_t entityIndex) {
        auto componentIndex = entityIndexToComponent_[entityIndex];
        if (componentIndex == std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        entityIndexToComponent_[entityIndex] = std::numeric_limits<uint32_t>::max();
        if (componentIndex + 1 < size_) {
            // If we are removing a middle component, move the last component into the old space.
            entityIndexToComponent_[componentToEntityIndex_.back()] = componentIndex;
            componentToEntityIndex_[componentIndex] = componentToEntityIndex_.back();
            *getComponent(componentIndex) = std::move(*getComponent(size_ - 1));
        }

        std::destroy_at(getComponent(size_ - 1));

        componentToEntityIndex_.pop_back();
        --size_;
        return true;
    }

    T& at(uint32_t entityIndex) {
        auto componentIndex = entityIndexToComponent_.at(entityIndex);
        if (componentIndex == std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range("ComponentArray::at(): component does not exist for entity at index " + std::to_string(entityIndex) + ".");
        }
        return *getComponent(componentIndex);
    }

    T& operator[](uint32_t entityIndex) {
        return *getComponent(entityIndexToComponent_[entityIndex]);
    }

    bool hasEntity(uint32_t entityIndex) const {
        return entityIndexToComponent_[entityIndex] != std::numeric_limits<uint32_t>::max();
    }

    uint32_t size() const {
        return size_;
    }

    Iterator begin() {
        return {componentToEntityIndex_.data(), getComponent(0), 0};
    }

    Iterator end() {
        return {componentToEntityIndex_.data(), getComponent(0), size_};
    }

private:
    inline T* getComponent(uint32_t componentIndex) {
        return std::launder(reinterpret_cast<T*>(&data_) + componentIndex);
    }

    // Placement new and aligned memory is tricky, see following for details:
    // https://en.cppreference.com/w/cpp/language/new.html
    // https://en.cppreference.com/w/cpp/types/aligned_storage.html
    // https://en.cppreference.com/w/cpp/utility/launder.html

    alignas(T) std::byte data_[sizeof(T) * MAX_SIZE];
    std::vector<uint32_t> entityIndexToComponent_;
    std::vector<uint32_t> componentToEntityIndex_;
    uint32_t size_ = 0;
};
