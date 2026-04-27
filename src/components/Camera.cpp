#include <components/Camera.h>
#include <components/Transform.h>
#include <Scene.h>

#include <cassert>
#include <glm/ext/matrix_clip_space.hpp>

namespace components {

CameraSystem::Instance CameraSystem::create(EntityId id) const {
    auto& scene = Scene::instance();
    auto transform = scene.access<Transform>(id);
    auto camera = scene.assign<Camera>(id);
    if (transform == nullptr) {
        return {id, camera, scene.assign<Transform>(id, id)};
    } else {
        camera->ownsTransform = false;
        return {id, camera, transform};
    }
}

void CameraSystem::destroy(EntityId id) const {
    auto& scene = Scene::instance();
    if (scene.access<Camera>(id)->ownsTransform) {
        scene.remove<Transform>(id);
    }
    scene.remove<Camera>(id);
}

CameraSystem::Instance CameraSystem::getInstance(EntityId id) const {
    auto& scene = Scene::instance();
    return {id, scene.access<Camera>(id), scene.access<Transform>(id)};
}

const glm::mat4& CameraSystem::getProjection(Instance inst) const {
    return inst.c->projection;
}

const glm::mat4& CameraSystem::getViewProjection(Instance inst) const {
    // FIXME: NYI
    return {};
}

void CameraSystem::setProjection(Instance inst, float fovYRadians, float aspect, float near, float far) const {
    inst.c->projection = glm::perspective(fovYRadians, aspect, near, far);

    // Invert the Y direction on projection matrix so that we are more similar to OpenGL and gltf axis.
    // FIXME: will need to verify this is the right approach
    inst.c->projection[1][1] *= -1.0f;
}

} // namespace components
