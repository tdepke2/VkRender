#pragma once

#include <cstdint>

using EntityId = uint64_t;
class Scene;

class View {
public:
    View(Scene& scene);

    Scene& getScene() const;
    EntityId getCamera() const;
    void setScene(Scene& scene);
    void setCamera(EntityId camera);

private:
    Scene* scene_;
    EntityId camera_ = 0;
};
