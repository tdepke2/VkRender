#pragma once

#include <vector>

class IndexBuffer;
class MaterialInstance;
struct MeshAsset;
class VertexBuffer;

namespace components {

enum class PrimitiveType {
    points,
    lines,
    lineStrip,
    triangles,
    triangleStrip
};

struct Renderable {
    struct Primitive {
        PrimitiveType type;
        VertexBuffer* vertices = nullptr;
        IndexBuffer* indices = nullptr;
        size_t offset = 0;
        size_t count = 0;
        MaterialInstance* material = nullptr;
    };

    MeshAsset* mesh;    // FIXME: remove once primitives are working
    std::vector<Primitive> primitives;    // FIXME: another case for small vec type
};

} // namespace components
