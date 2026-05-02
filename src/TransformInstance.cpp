#include <components/Transform.h>
#include <Scene.h>
#include <TransformInstance.h>

#include <cassert>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



#include <iostream>




TransformInstance TransformInstance::create(Scene& scene, EntityId id) {
    return {scene, id, scene.assign<components::Transform>(id)};
}

void TransformInstance::destroy(Scene& scene, EntityId id) {
    scene.remove<components::Transform>(id);
}

TransformInstance TransformInstance::get(Scene& scene, EntityId id) {
    return {scene, id, scene.access<components::Transform>(id)};
}

void TransformInstance::printDebug() const {
    std::cout << "Transform id " << id_ << ", parent = " << (scene_->getParent(id_) ? std::to_string(*scene_->getParent(id_)) : "null") << "\n";
    std::cout << "  children = { ";
    for (auto child : scene_->getChildren(id_)) {
        std::cout << child << " ";
    }
    std::cout << "}\n";
}

const glm::vec3& TransformInstance::getPosition() const {
    return t_->position;
}
const glm::quat& TransformInstance::getOrientation() const {
    return t_->orientation;
}
const glm::vec3& TransformInstance::getScale() const {
    return t_->scale;
}
const glm::vec3& TransformInstance::getOrigin() const {
    return t_->origin;
}
void TransformInstance::setPosition(const glm::vec3& position) {
    t_->position = position;
    localTransformChanged();
}
void TransformInstance::setOrientation(const glm::quat& orientation) {
    t_->orientation = orientation;
    localTransformChanged();
}
void TransformInstance::setScale(const glm::vec3& scale) {
    t_->scale = scale;
    localTransformChanged();
}
void TransformInstance::setOrigin(const glm::vec3& origin) {
    t_->origin = origin;
    localTransformChanged();
}

void TransformInstance::move(const glm::vec3& offset) {
    t_->position += offset;
    localTransformChanged();
}
void TransformInstance::rotate(const glm::quat& angle) {
    t_->orientation *= angle;    // FIXME: this may be the wrong order.
    localTransformChanged();
}
void TransformInstance::scale(const glm::vec3& factor) {
    t_->scale += factor;
    localTransformChanged();
}

const glm::mat4& TransformInstance::getLocalTransform() const {
    if (t_->localDirty) {
        //std::cout << "computing local transform\n";
        t_->localRaw = glm::translate(glm::mat4(1.0f), t_->position - t_->origin);    // FIXME: this isn't going to be the most efficient way to calculate this.
        t_->localRaw *= glm::mat4_cast(t_->orientation);
        t_->localRaw = glm::scale(t_->localRaw, t_->scale);
        t_->localDirty = false;
    }
    return t_->localRaw;
}

const glm::mat4& TransformInstance::getWorldTransform() const {
    // When transform changed and dirty flag not set, set dirty flag for this one and all descendants.
    // When getting transform, if dirty flag set then compute transform for this one and ancestors (up until we see dirty flag not set) and unset the flag for each.
    // Any new child will have dirty flag set.

    auto parent = scene_->getParent(id_);    // FIXME: logic is wrong? check worldDirty first, then parent.
    if (parent) {
        if (t_->worldDirty) {
            //std::cout << "computing world transform\n";
            auto parentTransform = TransformInstance::get(*scene_, *parent);
            if (parentTransform.isValid()) {
                t_->worldRaw = parentTransform.getWorldTransform() * getLocalTransform();
            } else {
                t_->worldRaw = getLocalTransform();
            }
            t_->worldDirty = false;
        }
        return t_->worldRaw;
    } else {
        //std::cout << "world transform up to date (no parent)\n";
        t_->worldDirty = false;
        return getLocalTransform();
    }
}

void TransformInstance::setLocalTransform(const glm::mat4& local) {
    t_->localRaw = local;
    t_->localDirty = false;
    worldTransformChanged();
}

TransformInstance::TransformInstance(Scene& scene, EntityId id, components::Transform* t) :
    scene_(&scene),
    id_(id),
    t_(t) {
}

void TransformInstance::localTransformChanged() {
    if (t_->localDirty) {
        return;
    }

    //std::cout << "local transform needs update\n";
    t_->localDirty = true;
    worldTransformChanged();    // FIXME: we can't skip this call if localDirty_
}

void TransformInstance::worldTransformChanged() {
    if (t_->worldDirty) {
        return;
    }

    //std::cout << "world transform needs update\n";
    t_->worldDirty = true;
    for (auto child : scene_->getChildren(id_)) {
        auto childTransform = TransformInstance::get(*scene_, child);
        if (childTransform.isValid()) {
            childTransform.worldTransformChanged();
        }
    }
}
