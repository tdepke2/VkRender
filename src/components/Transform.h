#pragma once

#include <cstdint>
#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <vector>

using EntityId = uint64_t;

namespace components {

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
    Transform& operator=(Transform&& rhs) noexcept = default;

    //void setPosition(); // maybe do translate(), rotate(), scale() instead?
    //void setScale();
    //void lookAt();
    //std::optional<EntityId> getParent(); // optional return val?
    // no function to set parent probably makes sense
    // children iterator?

    const glm::vec3& getPosition() const;
    const glm::quat& getOrientation() const;
    const glm::vec3& getScale() const;
    const glm::vec3& getOrigin() const;
    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);
    void setScale(const glm::vec3& scale);
    void setOrigin(const glm::vec3& origin);

    void move(const glm::vec3& offset);
    void rotate(const glm::quat& angle);
    void scale(const glm::vec3& factor);

    const glm::mat4& getLocalTransform() const;
    const glm::mat4& getWorldTransform() const; // composition of this local transform and parent's world transform.
    void setLocalTransform(const glm::mat4& local);

private:
    void localTransformChanged();
    void worldTransformChanged();

    glm::vec3 position_ {0.0f};
    glm::quat orientation_ {1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale_ {1.0f};
    glm::vec3 origin_ {0.0f};
    mutable glm::mat4 local_;
    mutable glm::mat4 world_;
    mutable bool localDirty_ = true;
    mutable bool worldDirty_ = true;
    std::optional<EntityId> parent_;
    std::vector<EntityId> children_;
};

} // namespace components
