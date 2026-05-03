#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>

namespace components {
    struct Camera;
    struct Transform;
}

using EntityId = uint64_t;
class Scene;

class CameraInstance {
public:
    static CameraInstance create(Scene& scene, EntityId id);
    static void destroy(Scene& scene, EntityId id);
    static CameraInstance get(Scene& scene, EntityId id);

    inline bool isValid() const {
        return c_ != nullptr && t_ != nullptr;
    }

    const glm::mat4& getProjection() const;
    const glm::mat4& getViewProjection() const;
    void setProjection(float fovYRadians, float aspect, float near, float far);

private:
    CameraInstance(Scene& scene, EntityId id, components::Camera* c, components::Transform* t);

    Scene* scene_;
    EntityId id_;
    components::Camera* c_;
    components::Transform* t_;
};
