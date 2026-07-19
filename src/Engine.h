#pragma once

#include <Common.h>
#include <Descriptors.h>
#include <EngineSettings.h>

#include <functional>
#include <span>
#include <vector>

union SDL_Event;
struct SDL_Window;
class View;

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

    Engine();

    void init();
    void processEvent(const SDL_Event* event);
    bool beginImGuiFrame();
    void endImGuiFrame();
    bool render(View& view);
    void cleanup();

    const vk::raii::Device& getDevice() const;
    VmaAllocator getAllocator() const;
    vk::Extent2D getWindowExtent() const;

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

private:
    void initVulkan();
    void initSwapchain();
    void createSwapchain(uint32_t width, uint32_t height);
    bool resizeSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    void initCommands();
    void initSyncStructures();
    void initDescriptors();
    void updateDescriptors();
    void initPipelines();
    void initMeshPipeline();
    void initImGui();
    void initDefaultData();

    FrameData& getCurrentFrame() { return frames_[frameNumber_ % FRAME_OVERLAP]; };
    void draw(View& view);
    void drawBackground(vk::CommandBuffer cmd);
    void drawGeometry(vk::CommandBuffer cmd, View& view);
    void drawImGui(vk::CommandBuffer cmd, vk::ImageView targetImageView);
    void immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function);
    AllocatedBuffer createBuffer(size_t allocSize, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    AllocatedImage createImage(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage createImage(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped = false);

    EngineSettings settings_;
    uint64_t frameNumber_ = 0;

    vk::Extent2D windowExtent_ = { 0, 0 };
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

    DescriptorAllocator globalDescriptorAllocator_;

    vk::raii::DescriptorSet drawImageDescriptors_ = nullptr;
    vk::raii::DescriptorSetLayout drawImageDescriptorLayout_ = nullptr;

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

    vk::raii::DescriptorPool imguiPool_ = nullptr;

    vk::raii::PipelineLayout meshPipelineLayout_ = nullptr;
    vk::raii::Pipeline meshPipeline_ = nullptr;

    //GPUSceneData sceneData;
    //VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage whiteImage_;
    AllocatedImage blackImage_;
    AllocatedImage greyImage_;
    AllocatedImage errorCheckerboardImage_;

    vk::raii::Sampler defaultSamplerLinear_ = nullptr;
    vk::raii::Sampler defaultSamplerNearest_ = nullptr;

    vk::raii::DescriptorSet singleImageDescriptors_ = nullptr;
    vk::raii::DescriptorSetLayout singleImageDescriptorLayout_ = nullptr;

    bool resizeRequested_ = false;
    bool freezeRendering_ = false;
    float renderScale_ = 1.0f;

    friend class EngineSettings;
};
