#pragma once

#include <functional>
#include <vector>
#include <span>

#include <Common.h>
#include <Loader.h>
#include <Descriptors.h>

struct SDL_Window;

struct FrameData {
    vk::raii::Semaphore swapchainSemaphore = nullptr, renderSemaphore = nullptr;
    vk::raii::Fence renderFence = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffer mainCommandBuffer = nullptr;

    //DescriptorAllocatorGrowable _frameDescriptors;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

class Engine {
public:
    void init();
    void run();
    void cleanup();

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

private:
    void initVulkan();
    void initSwapchain();
    void createSwapchain(uint32_t width, uint32_t height);
    void resizeSwapchain();
    void destroySwapchain();
    void initCommands();
    void initSyncStructures();
    void initDescriptors();
    void initPipelines();
    void initMeshPipeline();
    void initImGui();
    void initDefaultData();

    FrameData& getCurrentFrame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
    void draw();
    void drawBackground(vk::CommandBuffer cmd);
    void drawGeometry(vk::CommandBuffer cmd);
    void drawImGui(vk::CommandBuffer cmd, vk::ImageView targetImageView);
    void immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function);
    AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroyBuffer(const AllocatedBuffer& buffer);
    AllocatedImage createImage(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage createImage(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);
    void destroyImage(AllocatedImage& img);

    int _frameNumber = 0;

    vk::Extent2D windowExtent_ = { 17 * 40 , 9 * 40 };
    SDL_Window* window_ = nullptr;

    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Device device_ = nullptr;

    vk::raii::Queue graphicsQueue_ = nullptr;
    uint32_t graphicsQueueFamily_;

    VmaAllocator allocator_;

    vk::raii::SwapchainKHR swapchain_ = nullptr;
    vk::Format swapchainImageFormat_ = vk::Format::eUndefined;
    vk::Extent2D swapchainExtent_;

    std::vector<vk::Image> swapchainImages_;
    std::vector<vk::raii::ImageView> swapchainImageViews_;

    FrameData _frames[FRAME_OVERLAP];

    AllocatedImage drawImage_;
    AllocatedImage depthImage_;
    vk::Extent2D drawExtent_;

    DescriptorAllocator globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    vk::raii::Pipeline gradientPipeline_ = nullptr;
    vk::raii::PipelineLayout gradientPipelineLayout_ = nullptr;
    ComputePushConstants gradientConstants_ = {
        {1, 0, 0, 1},
        {0, 0, 1, 1},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };

    // Immediate submit structures.
    vk::raii::Fence immFence_ = nullptr;
    vk::raii::CommandPool immCommandPool_ = nullptr;
    vk::raii::CommandBuffer immCommandBuffer_ = nullptr;

    VkDescriptorPool imguiPool;

    vk::raii::PipelineLayout meshPipelineLayout_ = nullptr;
    vk::raii::Pipeline meshPipeline_ = nullptr;

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    //GPUSceneData sceneData;
    //VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    vk::raii::Sampler _defaultSamplerLinear = nullptr;
    vk::raii::Sampler _defaultSamplerNearest = nullptr;

    VkDescriptorSet _singleImageDescriptors;
    VkDescriptorSetLayout _singleImageDescriptorLayout;

    bool resizeRequested = false;
    bool freeze_rendering = false;
    float renderScale = 1.0f;
};
