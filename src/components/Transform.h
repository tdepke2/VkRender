#pragma once

#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace components {

struct Transform {
    glm::vec3 position {0.0f};
    glm::quat orientation {1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale {1.0f};
    glm::vec3 origin {0.0f};
    glm::mat4 localRaw;
    glm::mat4 worldRaw;
    bool localDirty = true;
    bool worldDirty = true;
};

} // namespace components
