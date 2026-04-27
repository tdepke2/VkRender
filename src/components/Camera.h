#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>

using EntityId = uint64_t;

namespace components {

struct Transform;

struct Camera {
    glm::mat4 projection;
    bool ownsTransform = true;
};

class CameraSystem {
public:
    struct Instance {
        EntityId id;
        Camera* c;
        Transform* t;
    };

    ~CameraSystem() = default;
    CameraSystem(const CameraSystem& rhs) = delete;
    CameraSystem(CameraSystem&& rhs) noexcept = delete;
    CameraSystem& operator=(const CameraSystem& rhs) = delete;
    CameraSystem& operator=(CameraSystem&& rhs) noexcept = delete;

    Instance create(EntityId id) const;
    void destroy(EntityId id) const;
    Instance getInstance(EntityId id) const;

    const glm::mat4& getProjection(Instance inst) const;
    const glm::mat4& getViewProjection(Instance inst) const;
    void setProjection(Instance inst, float fovYRadians, float aspect, float near, float far) const;

private:
    CameraSystem() = default;

    friend class Engine;
};

} // namespace components
