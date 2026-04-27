#pragma once

#include <cstdint>
#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

using EntityId = uint64_t;

namespace components {

// The Transform is unique as it combines data and behavior within a component, which is not typical of ECS.
// This decision was made as the transform uses dirty bits to track pending updates in the matrices.
// As such, there is no `TransformSystem`, the `Transform` can be directly assigned to an entity in the scene.
struct Transform {
public:
    Transform(EntityId id);

    //void lookAt();

    void printDebug();

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

    EntityId id_;
    glm::vec3 position_ {0.0f};
    glm::quat orientation_ {1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale_ {1.0f};
    glm::vec3 origin_ {0.0f};
    mutable glm::mat4 local_;
    mutable glm::mat4 world_;
    mutable bool localDirty_ = true;    // FIXME: these need to be updated on copy/move operation?
    mutable bool worldDirty_ = true;
};

} // namespace components
