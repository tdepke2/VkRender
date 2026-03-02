#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>

using EntityId = uint64_t;

namespace components {

class Camera {
private:
    struct Private {
        explicit Private() = default;
    };

public:
    static Camera* addToScene(EntityId id);
    Camera(Private);
    ~Camera() = default;
    Camera(const Camera& rhs) = delete;
    Camera(Camera&& rhs) noexcept = delete;
    Camera& operator=(const Camera& rhs) = delete;
    Camera& operator=(Camera&& rhs) noexcept = default;

    const glm::mat4& getProjection() const;
    const glm::mat4& getViewProjection() const;
    void setProjection(float fovYRadians, float aspect, float near, float far);

private:
    // FIXME: always store the entity id with the component?
    glm::mat4 projection_;
};

} // namespace components
