#pragma once

#include <cstdint>
#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

using EntityId = uint64_t;

struct TransformComponent {
    glm::vec3 position {0.0f};
    glm::quat orientation {1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale {1.0f};
    glm::vec3 origin {0.0f};
    glm::mat4 local;
    glm::mat4 world;
    bool localDirty = true;    // FIXME: these need to be updated on copy/move operation?
    bool worldDirty = true;
};

class Transforms {
public:
    struct Instance {
        EntityId id;
        TransformComponent* t;
    };

    ~Transforms() = default;
    Transforms(const Transforms& rhs) = delete;
    Transforms(Transforms&& rhs) noexcept = delete;
    Transforms& operator=(const Transforms& rhs) = delete;
    Transforms& operator=(Transforms&& rhs) noexcept = delete;

    //void lookAt();

    Instance create(EntityId id) const;
    void destroy(EntityId id) const;

    Instance getInstance(EntityId id) const;

    void printDebug(Instance inst);

    const glm::vec3& getPosition(Instance inst) const;
    const glm::quat& getOrientation(Instance inst) const;
    const glm::vec3& getScale(Instance inst) const;
    const glm::vec3& getOrigin(Instance inst) const;
    void setPosition(Instance inst, const glm::vec3& position) const;
    void setOrientation(Instance inst, const glm::quat& orientation) const;
    void setScale(Instance inst, const glm::vec3& scale) const;
    void setOrigin(Instance inst, const glm::vec3& origin) const;

    void move(Instance inst, const glm::vec3& offset) const;
    void rotate(Instance inst, const glm::quat& angle) const;
    void scale(Instance inst, const glm::vec3& factor) const;

    const glm::mat4& getLocalTransform(Instance inst) const;
    const glm::mat4& getWorldTransform(Instance inst) const; // composition of this local transform and parent's world transform.
    void setLocalTransform(Instance inst, const glm::mat4& local) const;

private:
    Transforms() = default;

    void localTransformChanged();
    void worldTransformChanged();

    friend class Engine;
};

// FIXME: need to think about this design more, it seems like the wrong way to structure the code for ECS.
// maybe have only the basic component data in ./components and the "manager" here in ./systems?
// see some other implementations:
// https://github.com/vblanco20-1/entt-breakout
// https://github.com/Daivuk/tddod
// https://gamedev.stackexchange.com/questions/174319/dealing-with-more-complex-entities-in-an-ecs-architecture
