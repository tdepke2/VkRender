#pragma once

#include <cstdint>
#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace components {
    struct Transform;
}

using EntityId = uint64_t;
class Scene;

class TransformInstance {
public:
    static TransformInstance create(Scene& scene, EntityId id);
    static void destroy(Scene& scene, EntityId id);
    static TransformInstance get(Scene& scene, EntityId id);

    inline bool isValid() const {
        return t_ != nullptr;
    }

    //void lookAt();

    void printDebug() const;

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
    TransformInstance(Scene& scene, EntityId id, components::Transform* t);

    void localTransformChanged();
    void worldTransformChanged();

    Scene* scene_;
    EntityId id_;
    components::Transform* t_;
};

// FIXME: need to think about this design more, it seems like the wrong way to structure the code for ECS.
// maybe have only the basic component data in ./components and the "manager" here in ./systems?
// see some other implementations:
// https://github.com/vblanco20-1/entt-breakout
// https://github.com/Daivuk/tddod
// https://gamedev.stackexchange.com/questions/174319/dealing-with-more-complex-entities-in-an-ecs-architecture
