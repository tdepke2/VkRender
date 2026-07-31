#pragma once

#include <Common.h>
#include <Descriptors.h>
#include <Material.h>
#include <Scene.h>
#include <unordered_map>
#include <filesystem>
#include <deque>

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

//forward declaration
class Engine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(Engine* engine, std::filesystem::path filePath);

// FIXME: all of above is now dead code

struct LoadedGltf {
    std::deque<VertexBuffer> vertexBuffers;
    std::deque<IndexBuffer> indexBuffers;
    std::unordered_map<std::string, EntityId> renderables;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, MaterialInstance*> materials;

    std::vector<EntityId> topRenderables;

    std::vector<vk::Sampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;
};

// FIXME: new gltf loading function is needed
