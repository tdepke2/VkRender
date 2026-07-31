#pragma once

#include <cstdint>
#include <vector>

class IndexBuffer;
class MaterialInstance;
struct MeshAsset;
class VertexBuffer;

namespace components {

struct Renderable {
    // Follows glTF-2.0 mesh.primitive.mode types.
    enum PrimitiveType {
        points = 0,
        lines = 1,
        //lineLoop = 2,
        lineStrip = 3,
        triangles = 4,
        triangleStrip = 5,
        //triangleFan = 6,
        count = 7
    };

    struct Primitive {
        PrimitiveType type;
        VertexBuffer* vertices = nullptr;
        IndexBuffer* indices = nullptr;
        uint32_t offset = 0;
        uint32_t count = 0;
        MaterialInstance* material = nullptr;
    };

    MeshAsset* mesh;    // FIXME: remove once primitives are working
    std::vector<Primitive> primitives;    // FIXME: another case for small vec type
};

} // namespace components
