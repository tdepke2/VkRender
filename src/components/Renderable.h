#pragma once

#include <cstdint>

using EntityId = uint64_t;
struct MeshAsset;

namespace components {

class Renderable {
public:
    Renderable(const MeshAsset& mesh);

    const MeshAsset& getMesh() const;

private:
    const MeshAsset* mesh_;
};

} // namespace components
