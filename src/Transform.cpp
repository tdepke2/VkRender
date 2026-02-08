#include <Scene.h>
#include <Transform.h>



#include <iostream>



Transform* Transform::addToScene(EntityId id, std::optional<EntityId> parent) {
    return Scene::instance().assignComponent<Transform>(id, Private(), id, parent);
}

Transform::Transform(Private, EntityId id, std::optional<EntityId> parent) :
    parent_(parent) {

    if (parent_) {
        Scene::instance().accessComponent<Transform>(*parent_)->children_.push_back(id);
    }
}

Transform::~Transform() {
    std::cout << "transform dtor\n";
}

Transform& Transform::operator=(Transform&& rhs) noexcept {
    mat_ = std::move(rhs.mat_);
    matDirty_ = std::move(rhs.matDirty_);
    parent_ = std::move(rhs.parent_);
    children_ = std::move(rhs.children_);
    return *this;
}

const glm::mat4& Transform::getTransform() const {
    return mat_;
}

glm::mat4 Transform::getWorldTransform() const {
    if (parent_) {
        return Scene::instance().accessComponent<Transform>(*parent_)->getWorldTransform() * mat_;    // FIXME: bad, we have to cache this val.
    } else {
        return mat_;
    }
}

void Transform::setTransform(const glm::mat4& mat) {
    mat_ = mat;
}
