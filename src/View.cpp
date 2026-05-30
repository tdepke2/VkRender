#include <View.h>

View::View(Scene& scene) :
    scene_(&scene) {
}

Scene& View::getScene() const {
    return *scene_;
}

EntityId View::getCamera() const {
    return camera_;
}

void View::setScene(Scene& scene) {
    scene_ = &scene;
}

void View::setCamera(EntityId camera) {
    camera_ = camera;
}
