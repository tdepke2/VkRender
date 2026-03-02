#include <components/Transform.h>
#include <Scene.h>

#include <cassert>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



#include <iostream>



namespace components {

Transform* Transform::addToScene(EntityId id, std::optional<EntityId> parent) {
    assert(Scene::instance().accessComponent<Transform>(id) == nullptr);
    return Scene::instance().assignComponent<Transform>(id, Private(), id, parent);
}

Transform::Transform(Private, EntityId id, std::optional<EntityId> parent) :
    parent_(parent) {

    std::cout << "new transform, id = " << id << ", parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    if (parent_) {
        assert(id != *parent_);
        Scene::instance().accessComponent<Transform>(*parent_)->children_.push_back(id);
    }
}

Transform::~Transform() {
    std::cout << "transform dtor, parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    if (parent_) {
        assert(Scene::instance().isEntityAlive(*parent_));    // FIXME: just temp assert this
    }

    // Inform all of the children that their parent is no more.
    // We could also inform our parent (and then checks for child entity alive
    // could be skipped), but that would require a linear search through vector.
    Scene& scene = Scene::instance();
    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            std::cout << "set child " << child << " parent to null\n";
            // The child may not have a transform if components are being deleted.
            Transform* childTransform = scene.accessComponent<Transform>(child);
            if (childTransform != nullptr) {
                childTransform->parent_ = std::nullopt;
            }
        }
    }
}

const glm::vec3& Transform::getPosition() const {
    return position_;
}
const glm::quat& Transform::getOrientation() const {
    return orientation_;
}
const glm::vec3& Transform::getScale() const {
    return scale_;
}
const glm::vec3& Transform::getOrigin() const {
    return origin_;
}
void Transform::setPosition(const glm::vec3& position) {
    position_ = position;
    localTransformChanged();
}
void Transform::setOrientation(const glm::quat& orientation) {
    orientation_ = orientation;
    localTransformChanged();
}
void Transform::setScale(const glm::vec3& scale) {
    scale_ = scale;
    localTransformChanged();
}
void Transform::setOrigin(const glm::vec3& origin) {
    origin_ = origin;
    localTransformChanged();
}

void Transform::move(const glm::vec3& offset) {
    position_ += offset;
    localTransformChanged();
}
void Transform::rotate(const glm::quat& angle) {
    orientation_ *= angle;    // FIXME: this may be the wrong order.
    localTransformChanged();
}
void Transform::scale(const glm::vec3& factor) {
    scale_ += factor;
    localTransformChanged();
}

const glm::mat4& Transform::getLocalTransform() const {
    if (localDirty_) {
        std::cout << "computing local transform\n";
        local_ = glm::translate(glm::mat4(1.0f), position_ - origin_);    // FIXME: this isn't going to be the most efficient way to calculate this.
        local_ *= glm::mat4_cast(orientation_);
        local_ = glm::scale(local_, scale_);
        localDirty_ = false;
    } else {
        std::cout << "local transform up to date, return it\n";
    }
    return local_;
}

const glm::mat4& Transform::getWorldTransform() const {
    // When transform changed and dirty flag not set, set dirty flag for this one and all descendants.
    // When getting transform, if dirty flag set then compute transform for this one and ancestors (up until we see dirty flag not set) and unset the flag for each.
    // Any new child will have dirty flag set.

    if (parent_) {
        if (worldDirty_) {
            std::cout << "computing world transform\n";
            world_ = Scene::instance().accessComponent<Transform>(*parent_)->getWorldTransform() * getLocalTransform();
            worldDirty_ = false;
        } else {
            std::cout << "world transform up to date, return it\n";
        }
        return world_;
    } else {
        std::cout << "world transform up to date (no parent)\n";
        worldDirty_ = false;
        return getLocalTransform();
    }
}

void Transform::setLocalTransform(const glm::mat4& local) {
    local_ = local;
    worldTransformChanged();
}

void Transform::localTransformChanged() {
    if (localDirty_) {
        return;
    }

    std::cout << "local transform needs update\n";
    localDirty_ = true;
    worldTransformChanged();
}

void Transform::worldTransformChanged() {
    if (worldDirty_) {
        return;
    }

    std::cout << "world transform needs update\n";
    worldDirty_ = true;
    Scene& scene = Scene::instance();
    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            scene.accessComponent<Transform>(child)->worldTransformChanged();
        }
    }
}

} // namespace components
