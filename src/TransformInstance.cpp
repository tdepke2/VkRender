#include <components/Transform.h>
#include <Scene.h>
#include <TransformInstance.h>

#include <cassert>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



#include <iostream>




TransformInstance TransformInstance::create(Scene& scene, EntityId id) {
    std::cout << "TransformInstance::create() for id " << id << "\n";
    return {scene, id, scene.assign<components::Transform>(id)};
}

void TransformInstance::destroy(Scene& scene, EntityId id) {
    std::cout << "TransformInstance::destroy() for id " << id << "\n";
    auto transform = TransformInstance::get(scene, id);
    if (transform.isValid()) {
        transform.worldTransformChanged();
    }
    scene.remove<components::Transform>(id);
}

TransformInstance TransformInstance::get(Scene& scene, EntityId id) {
    return {scene, id, scene.access<components::Transform>(id)};
}

TransformInstance TransformInstance::get(Scene& scene, EntityId id, components::Transform* t) {
    return {scene, id, t};
}

void TransformInstance::printDebug() const {
    std::cout << "Transform id " << id_ << ", localDirty = " << t_->localDirty << ", worldDirty = " << t_->worldDirty << ", parent = " << (scene_->getParent(id_) ? std::to_string(*scene_->getParent(id_)) : "null") << "\n";
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

void TransformInstance::move(const glm::vec3& offset) {
    t_->position += offset;
    localTransformChanged();
}
void TransformInstance::rotate(const glm::quat& angle) {
    t_->orientation *= angle;
    localTransformChanged();
}
void TransformInstance::scale(const glm::vec3& factor) {
    t_->scale += factor;
    localTransformChanged();
}

const glm::mat4& TransformInstance::getLocalTransform() const {
    if (t_->localDirty) {
        //std::cout << "TransformInstance::getLocalTransform() for id " << id_ << " recomputing...\n";
        t_->localRaw = glm::translate(glm::mat4(1.0f), t_->position);    // FIXME: this isn't going to be the most efficient way to calculate this.
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

    if (t_->worldDirty) {
        auto parent = scene_->getParent(id_);
        if (parent) {
            //std::cout << "TransformInstance::getWorldTransform() for id " << id_ << " recomputing...\n";
            auto parentTransform = TransformInstance::get(*scene_, *parent);
            if (parentTransform.isValid()) {
                t_->worldRaw = parentTransform.getWorldTransform() * getLocalTransform();
            } else {
                t_->worldRaw = getLocalTransform();
            }
        } else {
            //std::cout << "TransformInstance::getWorldTransform() for id " << id_ << " up to date (no parent)\n";
            t_->worldRaw = getLocalTransform();
        }
        t_->worldDirty = false;
    }
    return t_->worldRaw;
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
        //std::cout << "TransformInstance::localTransformChanged() for id " << id_ << " but already dirty.\n";
        // We can skip call to worldTransformChanged() as localDirty implies worldDirty.
        return;
    }

    //std::cout << "TransformInstance::localTransformChanged() for id " << id_ << "\n";
    t_->localDirty = true;
    worldTransformChanged();
}

void TransformInstance::worldTransformChanged() {
    if (t_->worldDirty) {
        //std::cout << "TransformInstance::worldTransformChanged() for id " << id_ << " but already dirty.\n";
        return;
    }

    //std::cout << "TransformInstance::worldTransformChanged() for id " << id_ << "\n";
    t_->worldDirty = true;
    for (auto child : scene_->getChildren(id_)) {
        auto childTransform = TransformInstance::get(*scene_, child);
        if (childTransform.isValid()) {
            childTransform.worldTransformChanged();
        }
    }
}
