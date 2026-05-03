#pragma once

#include <functional>
#include <vector>
#include <span>

#include <Common.h>
#include <Descriptors.h>

class Scene;
union SDL_Event;
struct SDL_Window;

struct FrameData {
    vk::raii::Semaphore swapchainSemaphore = nullptr;
    vk::raii::Fence renderFence = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffer mainCommandBuffer = nullptr;

    //DescriptorAllocatorGrowable _frameDescriptors;
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

class Engine {
public:
    static constexpr unsigned int FRAME_OVERLAP = 2;

    void init();
    void processEvent(const SDL_Event* event);
    void render(Scene& scene);
    void cleanup();

    const vk::raii::Device& getDevice() const;
    VmaAllocator getAllocator() const;

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

private:
    void initVulkan();
    void initSwapchain();
    void createSwapchain(uint32_t width, uint32_t height);
    bool resizeSwapchain();
    void destroySwapchain();
    void initCommands();
    void initSyncStructures();
    void initDescriptors();
    void initPipelines();
    void initMeshPipeline();
    void initImGui();
    void initDefaultData();

    FrameData& getCurrentFrame() { return frames_[frameNumber_ % FRAME_OVERLAP]; };
    void draw(Scene& scene);
    void drawBackground(vk::CommandBuffer cmd);
    void drawGeometry(vk::CommandBuffer cmd, Scene& scene);
    void drawImGui(vk::CommandBuffer cmd, vk::ImageView targetImageView);
    void immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function);
    AllocatedBuffer createBuffer(size_t allocSize, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    AllocatedImage createImage(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage createImage(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);

    uint64_t frameNumber_ = 0;

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

    FrameData frames_[FRAME_OVERLAP];
    std::vector<vk::raii::Semaphore> renderSemaphores_;

    AllocatedImage drawImage_;
    AllocatedImage depthImage_;
    vk::Extent2D drawExtent_;

    DescriptorAllocator globalDescriptorAllocator;

    vk::raii::DescriptorSet _drawImageDescriptors = nullptr;
    vk::raii::DescriptorSetLayout _drawImageDescriptorLayout = nullptr;

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

    vk::raii::DescriptorPool imguiPool = nullptr;

    vk::raii::PipelineLayout meshPipelineLayout_ = nullptr;
    vk::raii::Pipeline meshPipeline_ = nullptr;

    //GPUSceneData sceneData;
    //VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    vk::raii::Sampler _defaultSamplerLinear = nullptr;
    vk::raii::Sampler _defaultSamplerNearest = nullptr;

    vk::raii::DescriptorSet _singleImageDescriptors = nullptr;
    vk::raii::DescriptorSetLayout _singleImageDescriptorLayout = nullptr;

    bool resizeRequested = false;
    bool freeze_rendering = false;
    float renderScale = 1.0f;
};
