#include <CameraInstance.h>
#include <components/Camera.h>
#include <components/Transform.h>
#include <Scene.h>
#include <TransformInstance.h>

#include <glm/ext/matrix_clip_space.hpp>

CameraInstance CameraInstance::create(Scene& scene, EntityId id) {
    auto c = scene.assign<components::Camera>(id);
    auto t = scene.access<components::Transform>(id);
    if (t == nullptr) {
        TransformInstance::create(scene, id);
        return {scene, id, c, scene.access<components::Transform>(id)};
    } else {
        c->ownsTransform = false;
        return {scene, id, c, t};
    }
}

void CameraInstance::destroy(Scene& scene, EntityId id) {
    auto c = scene.access<components::Camera>(id);
    if (c != nullptr && c->ownsTransform) {
        TransformInstance::destroy(scene, id);
    }
    scene.remove<components::Camera>(id);
}

CameraInstance CameraInstance::get(Scene& scene, EntityId id) {
    return {scene, id, scene.access<components::Camera>(id), scene.access<components::Transform>(id)};
}

const glm::mat4& CameraInstance::getProjection() const {
    return c_->projection;
}

glm::mat4 CameraInstance::getViewProjection() const {
    if (t_ != nullptr) {
        return c_->projection * glm::inverse(getTransform().getWorldTransform());
    } else {
        return c_->projection;
    }
}

void CameraInstance::setProjection(float fovYRadians, float aspect, float near, float far) {
    c_->projection = glm::perspective(fovYRadians, aspect, near, far);

    // Invert the Y direction on projection matrix so that we are more similar to OpenGL and gltf axis.
    // FIXME: will need to verify this is the right approach
    c_->projection[1][1] *= -1.0f;
}

TransformInstance CameraInstance::getTransform() const {
    return TransformInstance::get(*scene_, id_, t_);
}

CameraInstance::CameraInstance(Scene& scene, EntityId id, components::Camera* c, components::Transform* t) :
    scene_(&scene),
    id_(id),
    c_(c),
    t_(t) {
}
