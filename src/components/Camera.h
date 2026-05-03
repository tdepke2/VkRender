#pragma once

#include <glm/mat4x4.hpp>

namespace components {

struct Camera {
    glm::mat4 projection;
    bool ownsTransform = true;
};

} // namespace components
