#include <components/Camera.h>
#include <Scene.h>

#include <cassert>
#include <glm/ext/matrix_clip_space.hpp>

namespace components {

Camera* Camera::addToScene(EntityId id) {
    assert(Scene::instance().access<Camera>(id) == nullptr);
    return Scene::instance().assign<Camera>(id, Private());
}

Camera::Camera(Private) {
}

const glm::mat4& Camera::getProjection() const {
    return projection_;
}

const glm::mat4& Camera::getViewProjection() const {
    // FIXME: NYI
    return {};
}

void Camera::setProjection(float fovYRadians, float aspect, float near, float far) {
    projection_ = glm::perspective(fovYRadians, aspect, near, far);

    // Invert the Y direction on projection matrix so that we are more similar to OpenGL and gltf axis.
    // FIXME: will need to verify this is the right approach
    projection_[1][1] *= -1.0f;
}

} // namespace components
