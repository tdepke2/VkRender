#pragma once

#include <vector>

// FIXME: we should look into precompiled headers instead of doing this, see vk_types.h and corresponding cmake files.
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <spdlog/fmt/bundled/base.h>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

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

    int _frameNumber {0};

    VkExtent2D _windowExtent{ 1700 , 900 };
    SDL_Window* _window{ nullptr };

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    FrameData _frames[FRAME_OVERLAP];

    VkSurfaceKHR _surface;
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    VkExtent2D _swapchainExtent;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    bool resize_requested{ false };
    bool freeze_rendering{ false };
};
