#pragma once

#include <vector>

// FIXME: we should look into precompiled headers instead of doing this, see vk_types.h and corresponding cmake files.
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>    // FIXME: we want vk::to_string() instead
#include <vk_mem_alloc.h>
#include <spdlog/fmt/bundled/base.h>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

struct AllocatedImage {
    vk::Image image;
    vk::raii::ImageView imageView = nullptr;
    VmaAllocation allocation;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;

    // FIXME: this isn't great. do we want an raii wrapper for vma, or just use std::unique_ptr with custom deletor?
    void clear(VmaAllocator allocator) {
        imageView.clear();
        vmaDestroyImage(allocator, image, allocation);
    }
};

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

    FrameData& getCurrentFrame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
    void draw();
    void drawBackground(vk::CommandBuffer cmd);

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

    bool resize_requested{ false };
    bool freeze_rendering{ false };
};
