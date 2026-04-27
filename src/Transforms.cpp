#include <Scene.h>
#include <Transforms.h>

#include <cassert>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



#include <iostream>




Transforms::Instance Transforms::create(EntityId id) const {
    return {id, Scene::instance().assign<TransformComponent>(id)};
}

void Transforms::destroy(EntityId id) const {
    Scene::instance().remove<TransformComponent>(id);
}

Transforms::Instance Transforms::getInstance(EntityId id) const {
    return {id, Scene::instance().access<TransformComponent>(id)};
}

void Transforms::printDebug(Instance inst) {
    auto& scene = Scene::instance();
    std::cout << "Transform id " << inst.id << ", parent = " << (scene.getParent(inst.id) ? std::to_string(*scene.getParent(inst.id)) : "null") << "\n";
    std::cout << "  children = { ";
    for (auto child : scene.getChildren(inst.id)) {
        std::cout << child << " ";
    }
    std::cout << "}\n";
}

const glm::vec3& Transforms::getPosition(Instance inst) const {
    return inst.t->position;
}
const glm::quat& Transforms::getOrientation(Instance inst) const {
    return inst.t->orientation;
}
const glm::vec3& Transforms::getScale(Instance inst) const {
    return scale_;
}
const glm::vec3& Transforms::getOrigin(Instance inst) const {
    return origin_;
}
void Transforms::setPosition(const glm::vec3& position) {
    position_ = position;
    localTransformChanged();
}
void Transforms::setOrientation(const glm::quat& orientation) {
    orientation_ = orientation;
    localTransformChanged();
}
void Transforms::setScale(const glm::vec3& scale) {
    scale_ = scale;
    localTransformChanged();
}
void Transforms::setOrigin(const glm::vec3& origin) {
    origin_ = origin;
    localTransformChanged();
}

void Transforms::move(const glm::vec3& offset) {
    position_ += offset;
    localTransformChanged();
}
void Transforms::rotate(const glm::quat& angle) {
    orientation_ *= angle;    // FIXME: this may be the wrong order.
    localTransformChanged();
}
void Transforms::scale(const glm::vec3& factor) {
    scale_ += factor;
    localTransformChanged();
}

const glm::mat4& Transforms::getLocalTransform() const {
    if (localDirty_) {
        //std::cout << "computing local transform\n";
        local_ = glm::translate(glm::mat4(1.0f), position_ - origin_);    // FIXME: this isn't going to be the most efficient way to calculate this.
        local_ *= glm::mat4_cast(orientation_);
        local_ = glm::scale(local_, scale_);
        localDirty_ = false;
    }
    return local_;
}

const glm::mat4& Transforms::getWorldTransform() const {
    // When transform changed and dirty flag not set, set dirty flag for this one and all descendants.
    // When getting transform, if dirty flag set then compute transform for this one and ancestors (up until we see dirty flag not set) and unset the flag for each.
    // Any new child will have dirty flag set.

    auto parent = Scene::instance().getParent(id_);
    if (parent) {
        if (worldDirty_) {
            //std::cout << "computing world transform\n";
            auto parentTransform = Scene::instance().access<Transform>(*parent);
            if (parentTransform != nullptr) {
                world_ = parentTransform->getWorldTransform() * getLocalTransform();
            } else {
                world_ = getLocalTransform();
            }
            worldDirty_ = false;
        }
        return world_;
    } else {
        //std::cout << "world transform up to date (no parent)\n";
        worldDirty_ = false;
        return getLocalTransform();
    }
}

void Transforms::setLocalTransform(const glm::mat4& local) {
    local_ = local;
    worldTransformChanged();
}

void Transforms::localTransformChanged() {
    if (localDirty_) {
        return;
    }

    //std::cout << "local transform needs update\n";
    localDirty_ = true;
    worldTransformChanged();    // FIXME: we can't skip this call if localDirty_
}

void Transforms::worldTransformChanged() {
    if (worldDirty_) {
        return;
    }

    //std::cout << "world transform needs update\n";
    worldDirty_ = true;
    Scene& scene = Scene::instance();
    for (auto child : scene.getChildren(id_)) {
        auto childTransform = scene.access<Transform>(child);
        if (childTransform != nullptr) {
            childTransform->worldTransformChanged();
        }
    }
}
