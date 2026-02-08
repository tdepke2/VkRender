#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <optional>
#include <vector>

using EntityId = uint64_t;

class Transform {
private:
    struct Private {
        explicit Private() = default;
    };

public:
    static Transform* addToScene(EntityId id, std::optional<EntityId> parent = std::nullopt);
    Transform(Private, EntityId id, std::optional<EntityId> parent);
    ~Transform();
    Transform(const Transform& rhs) = delete;
    Transform(Transform&& rhs) noexcept = delete;
    Transform& operator=(const Transform& rhs) = delete;
    Transform& operator=(Transform&& rhs) noexcept;

    //void setPosition(); // maybe do translate(), rotate(), scale() instead?
    //void setScale();
    //void lookAt();
    //std::optional<EntityId> getParent(); // optional return val?
    // no function to set parent probably makes sense
    // children iterator?

    const glm::mat4& getTransform() const;
    glm::mat4 getWorldTransform() const; // composition of this local transform and parent's world transform.
    void setTransform(const glm::mat4& mat);
private:
    // we likely only want to store the mat4 and parent/child data
    glm::mat4 mat_;
    mutable bool matDirty_ = false;
    std::optional<EntityId> parent_;
    std::vector<EntityId> children_;
};

// When transform changed and dirty flag not set, set dirty flag for this one and all descendants.
// When getting transform, if dirty flag set then compute transform for this one and ancestors (up until we see dirty flag not set) and unset the flag for each.
// Any new child will have dirty flag set.
