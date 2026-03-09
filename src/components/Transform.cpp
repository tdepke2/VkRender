#include <components/Transform.h>
#include <Scene.h>

#include <cassert>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>



#include <iostream>


namespace components {

Transform* Transform::addToScene(EntityId id, std::optional<EntityId> parent) {
    assert(Scene::instance().access<Transform>(id) == nullptr);
    return Scene::instance().assign<Transform>(id, Private(), id, parent);
}

void Transform::destroyEntityRecursive(EntityId id) {
    // Note that we can't call this within the Transform dtor because we must be
    // careful not to let a component destroy other components of the same type.
    // If this happened, the component would be mid destruction in the
    // ComponentArray while another change is being made to the array, the
    // memory could get moved out from underneath us.

    /*Scene& scene = Scene::instance();
    auto transform = scene.access<Transform>(id);
    if (transform == nullptr) {
        scene.destroyEntity(id);
        return;
    }

    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            Transform* childTransform = scene.access<Transform>(child);
            if (childTransform != nullptr) {
                childTransform->destroyChildren();
            }
            scene.destroyEntity(child);
        }
    }
    children_.clear();*/
}

Transform::Transform(Private, EntityId id, std::optional<EntityId> parent) :
    id_(id),
    parent_(parent) {

    std::cout << "new transform, id = " << id << ", parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    if (parent_) {
        assert(id != *parent_);
        Scene::instance().access<Transform>(*parent_)->children_.push_back(id);
        std::cout << "added id to parents children\n";
    }
}

Transform::~Transform() {
    std::cout << "transform dtor for " << id_ << ", parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    std::cout << "num children = " << children_.size() << "\n";

    // Inform all of the children that their parent is no more.
    // We could also inform our parent (and then checks for child entity alive
    // could be skipped), but that would require a linear search through vector.
    Scene& scene = Scene::instance();
    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            // The child may not have a transform if components are being deleted.
            Transform* childTransform = scene.access<Transform>(child);
            if (childTransform != nullptr) {
                std::cout << "set childTransform with parent " << *childTransform->parent_ << " to null\n";
                childTransform->parent_ = std::nullopt;
                childTransform->worldTransformChanged();
            }
        }
    }
}

Transform& Transform::operator=(Transform&& rhs) noexcept {
    std::cout << "transform move assign for " << id_ << ", parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    std::cout << "num children = " << children_.size() << "\n";

    Scene& scene = Scene::instance();
    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            // The child may not have a transform if components are being deleted.
            Transform* childTransform = scene.access<Transform>(child);
            if (childTransform != nullptr) {
                std::cout << "set childTransform with parent " << *childTransform->parent_ << " to null\n";
                childTransform->parent_ = std::nullopt;
                childTransform->worldTransformChanged();
            }
        }
    }

    position_ = std::move(rhs.position_);
    orientation_ = std::move(rhs.orientation_);
    scale_ = std::move(rhs.scale_);
    origin_ = std::move(rhs.origin_);
    local_ = std::move(rhs.local_);
    world_ = std::move(rhs.world_);
    localDirty_ = std::move(rhs.localDirty_);
    worldDirty_ = std::move(rhs.worldDirty_);
    id_ = std::move(rhs.id_);
    parent_ = std::move(rhs.parent_);
    children_ = std::move(rhs.children_);

    return *this;
}

void Transform::destroyChildren() {
    
}

void Transform::printDebug(EntityId id) {
    assert(id == id_);
    std::cout << "Transform id " << id << ", parent = " << (parent_ ? std::to_string(*parent_) : "no parent") << "\n";
    std::cout << "  num children = " << children_.size() << "\n";
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
        //std::cout << "computing local transform\n";
        local_ = glm::translate(glm::mat4(1.0f), position_ - origin_);    // FIXME: this isn't going to be the most efficient way to calculate this.
        local_ *= glm::mat4_cast(orientation_);
        local_ = glm::scale(local_, scale_);
        localDirty_ = false;
    }
    return local_;
}

const glm::mat4& Transform::getWorldTransform() const {
    // When transform changed and dirty flag not set, set dirty flag for this one and all descendants.
    // When getting transform, if dirty flag set then compute transform for this one and ancestors (up until we see dirty flag not set) and unset the flag for each.
    // Any new child will have dirty flag set.

    if (parent_) {
        if (worldDirty_) {
            //std::cout << "computing world transform\n";
            world_ = Scene::instance().access<Transform>(*parent_)->getWorldTransform() * getLocalTransform();
            worldDirty_ = false;
        }
        return world_;
    } else {
        //std::cout << "world transform up to date (no parent)\n";
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

    //std::cout << "local transform needs update\n";
    localDirty_ = true;
    worldTransformChanged();
}

void Transform::worldTransformChanged() {
    if (worldDirty_) {
        return;
    }

    //std::cout << "world transform needs update\n";
    worldDirty_ = true;
    Scene& scene = Scene::instance();
    for (auto child : children_) {
        if (scene.isEntityAlive(child)) {
            scene.access<Transform>(child)->worldTransformChanged();
        }
    }
}

} // namespace components
