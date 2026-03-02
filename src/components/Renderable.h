#pragma once

#include <cstdint>

using EntityId = uint64_t;
struct MeshAsset;

namespace components {

class Renderable {
private:
    struct Private {
        explicit Private() = default;
    };

public:
    static Renderable* addToScene(EntityId id, const MeshAsset& mesh);
    Renderable(Private, const MeshAsset& mesh);
    ~Renderable() = default;
    Renderable(const Renderable& rhs) = delete;
    Renderable(Renderable&& rhs) noexcept = delete;
    Renderable& operator=(const Renderable& rhs) = delete;
    Renderable& operator=(Renderable&& rhs) noexcept = default;

    const MeshAsset& getMesh() const;

private:
    const MeshAsset* mesh_;
};

} // namespace components
