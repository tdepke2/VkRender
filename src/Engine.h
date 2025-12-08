#pragma once

#include <functional>
#include <vector>

#include <Common.h>
#include <Descriptors.h>

struct SDL_Window;

struct FrameData {
    VkSemaphore _swapchainSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class Engine {
public:
    void init();
    void run();
    void cleanup();

private:
    void initVulkan();
    void initSwapchain();
    void createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    void initCommands();
    void initSyncStructures();
    void initDescriptors();
    void initPipelines();
    void initImGui();

    FrameData& getCurrentFrame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
    void draw();
    void drawBackground(vk::CommandBuffer cmd);
    void drawImGui(VkCommandBuffer cmd, VkImageView targetImageView);
    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

    int _frameNumber {0};

    vk::Extent2D windowExtent_ = { 1700 , 900 };
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
    VkExtent2D _drawExtent;

    DescriptorAllocator globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    VkDescriptorPool imguiPool;

    bool resize_requested{ false };
    bool freeze_rendering{ false };
};
