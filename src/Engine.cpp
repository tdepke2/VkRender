#include <Engine.h>
#include <Pipelines.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>

#include <spdlog/spdlog.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <thread>

#include <ConsoleVars.h>

namespace {

constexpr uint32_t vulkanApiVersion = vk::ApiVersion13;    // FIXME: move to vulkan 1.4? see: https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/01_Instance.html

constexpr bool enableValidationLayers = true;

void transitionImage(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout currentLayout, vk::ImageLayout newLayout) {
    vk::ImageMemoryBarrier2 imageBarrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,

        .oldLayout = currentLayout,
        .newLayout = newLayout,

        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = (newLayout == vk::ImageLayout::eDepthAttachmentOptimal ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor),
            .baseMipLevel = 0,
            .levelCount = vk::RemainingMipLevels,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };

    vk::DependencyInfo depInfo = {
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageBarrier,
    };

    cmd.pipelineBarrier2(depInfo);
}

void copyImageToImage(vk::CommandBuffer cmd, vk::Image source, vk::Image destination, vk::Extent2D sourceSize, vk::Extent2D destinationSize) {
    std::array<vk::Offset3D, 2> srcOffsets = {
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<int32_t>(sourceSize.width), static_cast<int32_t>(sourceSize.height), 1}
    };
    std::array<vk::Offset3D, 2> dstOffsets = {
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<int32_t>(destinationSize.width), static_cast<int32_t>(destinationSize.height), 1}
    };

    vk::ImageBlit2 blitRegion = {
        .srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .srcOffsets = srcOffsets,
        .dstSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .dstOffsets = dstOffsets
    };

    vk::BlitImageInfo2 blitInfo = {
        .srcImage = source,
        .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
        .dstImage = destination,
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount = 1,
        .pRegions = &blitRegion,
        .filter = vk::Filter::eLinear
    };

    cmd.blitImage2(blitInfo);
}

}

void Engine::init() {
    SDL_Init(SDL_INIT_VIDEO);

    // FIXME: recommended to use SDL_SetAppMetadata() after startup, see SDL api reference
    // FIXME: need to call SDL_Quit() at end?

    SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    window_ = SDL_CreateWindow("Vulkan Engine", windowExtent_.width, windowExtent_.height, windowFlags);
    if (window_ == nullptr) {
        spdlog::error("SDL failed to create window: {}", SDL_GetError());
        abort();
    }

    initVulkan();

    initSwapchain();

    initCommands();

    initSyncStructures();

    initDescriptors();

    initPipelines();

    initImGui();

    initDefaultData();
}

void Engine::run() {
    SDL_Event event;
    bool closeWindow = false;

    while (!closeWindow) {
        // Handle events from the queue.
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);

            if (event.type == SDL_EVENT_QUIT) {
                closeWindow = true;
            }

            if (event.type == SDL_EVENT_WINDOW_MINIMIZED) {
                freeze_rendering = true;
            } else if (event.type == SDL_EVENT_WINDOW_RESTORED) {
                freeze_rendering = false;
            }
        }

        if (freeze_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (resizeRequested && !resizeSwapchain()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // ImGui new frame.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // Some ImGui UI to test.
        //ImGui::ShowDemoWindow();

        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.0f);
            ImGui::InputFloat4("data1", reinterpret_cast<float*>(&gradientConstants_.data1));
            ImGui::InputFloat4("data2", reinterpret_cast<float*>(&gradientConstants_.data2));
            ImGui::InputFloat4("data3", reinterpret_cast<float*>(&gradientConstants_.data3));
            ImGui::InputFloat4("data4", reinterpret_cast<float*>(&gradientConstants_.data4));

            ImGui::SeparatorText("Console Vars");
            ConsoleVars::drawWithImGui();
        }
        ImGui::End();

        // Tell ImGui to calculate internal draw structures.
        ImGui::Render();

        draw();
    }
}

void Engine::cleanup() {
    // Ensure the GPU has stopped doing its things.
    vkDeviceWaitIdle(*device_);

    for (auto& mesh : testMeshes) {
        mesh->meshBuffers.indexBuffer.clear(allocator_);
        mesh->meshBuffers.vertexBuffer.clear(allocator_);
    }

    _defaultSamplerNearest.clear();
    _defaultSamplerLinear.clear();

    _whiteImage.clear(allocator_);
    _greyImage.clear(allocator_);
    _blackImage.clear(allocator_);
    _errorCheckerboardImage.clear(allocator_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    imguiPool.clear();//vkDestroyDescriptorPool(*device_, imguiPool, nullptr);

    meshPipelineLayout_.clear();//vkDestroyPipelineLayout(*device_, _meshPipelineLayout, nullptr);
    meshPipeline_.clear();//vkDestroyPipeline(*device_, _meshPipeline, nullptr);

    gradientPipelineLayout_.clear();//vkDestroyPipelineLayout(*device_, _gradientPipelineLayout, nullptr);
    gradientPipeline_.clear();//vkDestroyPipeline(*device_, _gradientPipeline, nullptr);

    /*for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        _frames[i]._frameDescriptors.destroy_pools(*device_);
    }*/

    _singleImageDescriptors.clear();
    _singleImageDescriptorLayout.clear();//vkDestroyDescriptorSetLayout(*device_, _singleImageDescriptorLayout, nullptr);
    _drawImageDescriptors.clear();
    _drawImageDescriptorLayout.clear();//vkDestroyDescriptorSetLayout(*device_, _drawImageDescriptorLayout, nullptr);

    globalDescriptorAllocator.destroyPool();

    immFence_.clear();//vkDestroyFence(*device_, _immFence, nullptr);

    immCommandBuffer_.clear();
    immCommandPool_.clear();//vkDestroyCommandPool(*device_, _immCommandPool, nullptr);

    renderSemaphores_.clear();

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        frames_[i].mainCommandBuffer.clear();
        frames_[i].commandPool.clear();//vkDestroyCommandPool(*device_, _frames[i]._commandPool, nullptr);

        frames_[i].renderFence.clear();//vkDestroyFence(*device_, _frames[i]._renderFence, nullptr);
        frames_[i].swapchainSemaphore.clear();//vkDestroySemaphore(*device_ ,_frames[i]._swapchainSemaphore, nullptr);
    }

    depthImage_.clear(allocator_);
    drawImage_.clear(allocator_);
    destroySwapchain();

    surface_.clear();
    vmaDestroyAllocator(allocator_);
    device_.clear();//vkDestroyDevice(_device, nullptr);
    physicalDevice_.clear();
    debugMessenger_.clear();//vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    instance_.clear();//vkDestroyInstance(_instance, nullptr);
    SDL_DestroyWindow(window_);
}

GPUMeshBuffers Engine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer = createBuffer(vertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        VMA_MEMORY_USAGE_GPU_ONLY);

    newSurface.vertexBufferAddress = device_.getBufferAddress({ .buffer = newSurface.vertexBuffer.buffer });

    newSurface.indexBuffer = createBuffer(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_ONLY);    // FIXME: vma flag is deprecated.

    AllocatedBuffer staging = createBuffer(vertexBufferSize + indexBufferSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.info.pMappedData;

    memcpy(data, vertices.data(), vertexBufferSize);

    memcpy(static_cast<char*>(data) + vertexBufferSize, indices.data(), indexBufferSize);

    immediateSubmit([vertexBufferSize,indexBufferSize,&staging,&newSurface](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertexBufferSize
        };
        cmd.copyBuffer(staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        vk::BufferCopy indexCopy = {
            .srcOffset = vertexBufferSize,
            .dstOffset = 0,
            .size = indexBufferSize
        };
        cmd.copyBuffer(staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    staging.clear(allocator_);

    return newSurface;
}

void Engine::initVulkan() {
    vkb::InstanceBuilder builder;

    // Make the Vulkan instance, with basic debug features.
    auto instRet = builder.set_app_name("VkRender")
        .request_validation_layers(enableValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(vulkanApiVersion)
        .build();

    vkb::Instance vkbInst = instRet.value();

    // Grab the instance.
    instance_ = vk::raii::Instance(context_, vkbInst.instance);
    debugMessenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, vkbInst.debug_messenger);

    VkSurfaceKHR surfaceHandle;
    if (!SDL_Vulkan_CreateSurface(window_, *instance_, nullptr, &surfaceHandle)) {
        spdlog::error("SDL failed to create surface: {}", SDL_GetError());
        abort();
    }
    surface_ = vk::raii::SurfaceKHR(instance_, surfaceHandle);

    // Vulkan 1.2 features.
    vk::PhysicalDeviceVulkan12Features features12 = {
        .descriptorIndexing = true,
        .bufferDeviceAddress = true,
    };

    // Vulkan 1.3 features.
    vk::PhysicalDeviceVulkan13Features features13 = {
        .synchronization2 = true,
        .dynamicRendering = true,
    };

    // Select a physical device (the GPU).
    vkb::PhysicalDeviceSelector selector(vkbInst);
    vkb::PhysicalDevice physicalDevice = selector
        .set_surface(*surface_)
        .set_minimum_version(1, 3)
        .set_required_features_12(features12)
        .set_required_features_13(features13)
        .select()
        .value();

    // Create the final Vulkan device.
    vkb::DeviceBuilder deviceBuilder(physicalDevice);
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a Vulkan application.
    physicalDevice_ = vk::raii::PhysicalDevice(instance_, physicalDevice.physical_device);
    device_ = vk::raii::Device(physicalDevice_, vkbDevice.device);

    // Get a Graphics queue.
    graphicsQueue_ = vk::raii::Queue(device_, vkbDevice.get_queue(vkb::QueueType::graphics).value());
    graphicsQueueFamily_ = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // Initialize the memory allocator.
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.physicalDevice = *physicalDevice_;
    allocatorInfo.device = *device_;
    allocatorInfo.instance = *instance_;
    allocatorInfo.vulkanApiVersion = vulkanApiVersion;
    vmaCreateAllocator(&allocatorInfo, &allocator_);
}

void Engine::initSwapchain() {
    createSwapchain(windowExtent_.width, windowExtent_.height);

    // Draw/depth image sizes will match the window.
    vk::Extent3D imageExtent = {
        .width = windowExtent_.width,
        .height = windowExtent_.height,
        .depth = 1
    };

    drawImage_ = createImage(imageExtent, vk::Format::eR16G16B16A16Sfloat,
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eColorAttachment);

    depthImage_ = createImage(imageExtent, vk::Format::eD32Sfloat,
        vk::ImageUsageFlagBits::eDepthStencilAttachment);
}

void Engine::createSwapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder(*physicalDevice_, *device_, *surface_);

    vk::SurfaceFormatKHR surfaceFormat = {
        .format = vk::Format::eB8G8R8A8Unorm,
        .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
    };

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(surfaceFormat)
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    // Store swapchain and its related images.
    swapchain_ = vk::raii::SwapchainKHR(device_, vkbSwapchain.swapchain);
    swapchainImageFormat_ = surfaceFormat.format;
    swapchainExtent_ = vkbSwapchain.extent;
    swapchainImages_ = swapchain_.getImages();

    spdlog::debug("Created swapchain of size {} by {} with {} images.", swapchainExtent_.width, swapchainExtent_.height, swapchainImages_.size());

    const auto imageViews = vkbSwapchain.get_image_views().value();
    std::transform(imageViews.begin(), imageViews.end(), std::back_inserter(swapchainImageViews_), [this](VkImageView v) {
        return vk::raii::ImageView(device_, v);
    });
}

bool Engine::resizeSwapchain() {
    device_.waitIdle();

    destroySwapchain();

    int width = 0, height = 0;
    SDL_GetWindowSize(window_, &width, &height);
    if (width == 0 || height == 0) {
        return false;
    }

    windowExtent_.width = width;
    windowExtent_.height = height;

    createSwapchain(windowExtent_.width, windowExtent_.height);

    resizeRequested = false;
    return true;
}

void Engine::destroySwapchain() {
    swapchainImageViews_.clear();
    swapchain_.clear();
}

void Engine::initCommands() {
    // Create a command pool for commands submitted to the graphics queue.
    // We also want the pool to allow for resetting of individual command buffers.
    vk::CommandPoolCreateInfo commandPoolInfo = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphicsQueueFamily_,
    };

    for (unsigned int i = 0; i < FRAME_OVERLAP; ++i) {
        frames_[i].commandPool = device_.createCommandPool(commandPoolInfo);

        // Allocate the default command buffer that we will use for rendering.
        vk::CommandBufferAllocateInfo cmdAllocInfo = {
            .commandPool = frames_[i].commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        frames_[i].mainCommandBuffer = std::move(device_.allocateCommandBuffers(cmdAllocInfo).front());
    }

    immCommandPool_ = device_.createCommandPool(commandPoolInfo);

    // Allocate the command buffer for immediate submits.
    vk::CommandBufferAllocateInfo cmdAllocInfo = {
        .commandPool = immCommandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    immCommandBuffer_ = std::move(device_.allocateCommandBuffers(cmdAllocInfo).front());
}

void Engine::initSyncStructures() {
    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        frames_[i].swapchainSemaphore = vk::raii::Semaphore(device_, vk::SemaphoreCreateInfo());
        frames_[i].renderFence = vk::raii::Fence(device_, { .flags = vk::FenceCreateFlagBits::eSignaled });
    }

    for (size_t i = 0; i < swapchainImages_.size(); ++i) {
        renderSemaphores_.emplace_back(device_, vk::SemaphoreCreateInfo());
    }

    immFence_ = vk::raii::Fence(device_, { .flags = vk::FenceCreateFlagBits::eSignaled });
}

void Engine::initDescriptors() {
    // Create a descriptor pool that will hold 10 sets with 1 image/sampler each.
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        { vk::DescriptorType::eStorageImage, 1 },
        { vk::DescriptorType::eCombinedImageSampler, 1 }
    };

    globalDescriptorAllocator.initPool(device_, 10, sizes);

    // Descriptor set for our compute draw.
    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, vk::DescriptorType::eStorageImage);
        _drawImageDescriptorLayout = builder.build(device_, vk::ShaderStageFlagBits::eCompute);

        _drawImageDescriptors = globalDescriptorAllocator.allocate(device_, _drawImageDescriptorLayout);

        DescriptorWriter writer;
        writer.writeImage(0, *drawImage_.imageView, nullptr, vk::ImageLayout::eGeneral, vk::DescriptorType::eStorageImage);

        writer.updateSet(device_, _drawImageDescriptors);
    }

    /*{
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(*device_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(*device_, 1000, frame_sizes);
    }*/

    {
        DescriptorLayoutBuilder builder;
        builder.addBinding(0, vk::DescriptorType::eCombinedImageSampler);
        _singleImageDescriptorLayout = builder.build(device_, vk::ShaderStageFlagBits::eFragment);

        _singleImageDescriptors = globalDescriptorAllocator.allocate(device_, _singleImageDescriptorLayout);
    }
}

void Engine::initPipelines() {
    vk::PushConstantRange pushConstant = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(ComputePushConstants),
    };

    vk::DescriptorSetLayout layoutCopyFIXME = _drawImageDescriptorLayout;

    vk::PipelineLayoutCreateInfo computeLayout = {
        .setLayoutCount = 1,
        .pSetLayouts = &layoutCopyFIXME,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant,
    };

    gradientPipelineLayout_ = device_.createPipelineLayout(computeLayout);

    vk::raii::ShaderModule computeDrawShader = createShaderModule("shaders/gradient_color.comp.spv", device_);

    vk::PipelineShaderStageCreateInfo stageinfo = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = computeDrawShader,
        .pName = "main",
    };

    vk::ComputePipelineCreateInfo computePipelineCreateInfo = {
        .stage = stageinfo,
        .layout = gradientPipelineLayout_,
    };

    gradientPipeline_ = device_.createComputePipeline(nullptr, computePipelineCreateInfo);

    computeDrawShader.clear();

    initMeshPipeline();
}

void Engine::initMeshPipeline() {
    vk::PushConstantRange bufferRange = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants),
    };

    vk::DescriptorSetLayout layoutCopyFIXME = _singleImageDescriptorLayout;

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {
        .setLayoutCount = 1,
        .pSetLayouts = &layoutCopyFIXME,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &bufferRange,
    };

    meshPipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);

    vk::raii::ShaderModule triangleFragShader = createShaderModule("shaders/tex_image.frag.spv", device_);

    vk::raii::ShaderModule triangleVertexShader = createShaderModule("shaders/colored_triangle_mesh.vert.spv", device_);

    PipelineBuilder pipelineBuilder;

    pipelineBuilder.setPipelineLayout(meshPipelineLayout_);

    pipelineBuilder.setShaders(triangleVertexShader, triangleFragShader);

    pipelineBuilder.setInputTopology(vk::PrimitiveTopology::eTriangleList);

    pipelineBuilder.setPolygonMode(vk::PolygonMode::eFill);

    pipelineBuilder.setCullMode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);

    pipelineBuilder.setMultisamplingNone();

    pipelineBuilder.disableBlending();

    pipelineBuilder.enableDepthtest(true, vk::CompareOp::eGreaterOrEqual);

    pipelineBuilder.setColorAttachmentFormat(drawImage_.imageFormat);
    pipelineBuilder.setDepthFormat(depthImage_.imageFormat);

    meshPipeline_ = pipelineBuilder.buildPipeline(device_);

    triangleFragShader.clear();
    triangleVertexShader.clear();
}

void Engine::initImGui() {
    // Create descriptor pool for ImGui.
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {
        {{ vk::DescriptorType::eCombinedImageSampler, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE }}
    };

    vk::DescriptorPoolCreateInfo poolInfo = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 0,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    for (const auto& poolSize : poolSizes) {
        poolInfo.maxSets += poolSize.descriptorCount;
    }

    imguiPool = vk::raii::DescriptorPool(device_, poolInfo);

    // Setup Dear ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window_);
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = *instance_;
    initInfo.PhysicalDevice = *physicalDevice_;
    initInfo.Device = *device_;
    initInfo.QueueFamily = graphicsQueueFamily_;
    initInfo.Queue = *graphicsQueue_;
    initInfo.DescriptorPool = *imguiPool;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = 2;
    initInfo.UseDynamicRendering = true;
    // FIXME: should specify vulkanApiVersion in here.

    // Dynamic rendering parameters for ImGui to use.
    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainImageFormat_
    };

    initInfo.PipelineInfoMain = {};
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingCreateInfo;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);
}

void Engine::initDefaultData() {
    /*std::array<Vertex,4> rect_vertices;

    rect_vertices[0].position = {0.5,-0.5, 0};
    rect_vertices[1].position = {0.5,0.5, 0};
    rect_vertices[2].position = {-0.5,-0.5, 0};
    rect_vertices[3].position = {-0.5,0.5, 0};

    rect_vertices[0].color = {0,0, 0,1};
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1};
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t,6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = uploadMesh(rect_indices,rect_vertices);*/

    //3 default textures, white, grey, black. 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = createImage((void*)&white, vk::Extent3D{ 1, 1, 1 }, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = createImage((void*)&grey, vk::Extent3D{ 1, 1, 1 }, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _blackImage = createImage((void*)&black, vk::Extent3D{ 1, 1, 1 }, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled);

    //checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 *16 > pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y*16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = createImage(pixels.data(), vk::Extent3D{16, 16, 1}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled);

    vk::SamplerCreateInfo sampl = {
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest
    };
    _defaultSamplerNearest = device_.createSampler(sampl);

    sampl.magFilter = vk::Filter::eLinear;
    sampl.minFilter = vk::Filter::eLinear;
    _defaultSamplerLinear = device_.createSampler(sampl);

    testMeshes = loadGltfMeshes(this,"assets/basicmesh.glb").value();

    DescriptorWriter writer;
    writer.writeImage(0, _errorCheckerboardImage.imageView, _defaultSamplerNearest, vk::ImageLayout::eShaderReadOnlyOptimal, vk::DescriptorType::eCombinedImageSampler);
    writer.updateSet(device_, _singleImageDescriptors);
}

void Engine::draw() {
    // Wait until the gpu has finished rendering the last frame, timeout of 1 second.
    device_.waitForFences(*getCurrentFrame().renderFence, vk::True, 1000000000);    // FIXME: need to VK_CHECK() this

    //getCurrentFrame()._frameDescriptors.clear_pools(*device_);

    // Request image from the swapchain.
    // If we get `vk::Result::eSuboptimalKHR` we could force a resize, but the resulting image is sufficient.
    uint32_t swapchainImageIndex = 0;
    try {
        auto result = swapchain_.acquireNextImage(1000000000, getCurrentFrame().swapchainSemaphore, nullptr);
        if (result.first == vk::Result::eErrorOutOfDateKHR) {
            resizeRequested = true;
            return;
        }
        swapchainImageIndex = result.second;
    } catch(const vk::SystemError& e) {
        if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
            resizeRequested = true;
            return;
        }
        // FIXME: VK_CHECK() the result now
    }

    // Reset the fence after checking for resize. If we reset it too early and skip this frame for a resize, the next wait for it would deadlock.
    device_.resetFences(*getCurrentFrame().renderFence);

    drawExtent_.width = static_cast<uint32_t>(std::min(swapchainExtent_.width, drawImage_.imageExtent.width) * renderScale);
    drawExtent_.height = static_cast<uint32_t>(std::min(swapchainExtent_.height, drawImage_.imageExtent.height) * renderScale);

    vk::CommandBuffer cmd = getCurrentFrame().mainCommandBuffer;

    // The command buffer will be implicitly reset when we call begin since we specify `vk::CommandPoolCreateFlagBits::eResetCommandBuffer` for the pool.
    //cmd.reset();

    // Begin the command buffer recording. We will use this command buffer exactly once.
    vk::CommandBufferBeginInfo cmdBeginInfo = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    cmd.begin(cmdBeginInfo);

    // Transition our main draw image into general layout so we can write into it.
    // We will overwrite it all so we don't care what the older layout was.
    transitionImage(cmd, drawImage_.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    drawBackground(cmd);

    transitionImage(cmd, drawImage_.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eColorAttachmentOptimal);
    transitionImage(cmd, depthImage_.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal);

    drawGeometry(cmd);

    // Transition the draw image and the swapchain image into their correct transfer layouts.
    transitionImage(cmd, drawImage_.image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    copyImageToImage(cmd, drawImage_.image, swapchainImages_[swapchainImageIndex], drawExtent_, swapchainExtent_);

    transitionImage(cmd, swapchainImages_[swapchainImageIndex], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal);

    drawImGui(cmd, *swapchainImageViews_[swapchainImageIndex]);

    // Set swapchain image layout to present so we can draw it.
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

    // Finalize the command buffer (we can no longer add commands, but it can now be executed).
    cmd.end();

    vk::SemaphoreSubmitInfo waitInfo = {
        .semaphore = getCurrentFrame().swapchainSemaphore,
        .value = 1,
        .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .deviceIndex = 0
    };
    vk::CommandBufferSubmitInfo cmdInfo = {
        .commandBuffer = cmd,
        .deviceMask = 0
    };
    // The render semaphores are per swapchain image, instead of per frame.
    // See here for details: https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
    vk::SemaphoreSubmitInfo signalInfo = {
        .semaphore = renderSemaphores_[swapchainImageIndex],
        .value = 1,
        .stageMask = vk::PipelineStageFlagBits2::eAllGraphics,
        .deviceIndex = 0
    };
    vk::SubmitInfo2 submitInfo = {
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo
    };

    // Submit command buffer to the queue and execute it.
    graphicsQueue_.submit2(submitInfo, getCurrentFrame().renderFence);

    // Prepare present, this will put the image we just rendered to into the
    // visible window. We want to wait on the render semaphore for that, as it's
    // necessary that drawing commands have finished before the image is
    // displayed to the user.
    vk::PresentInfoKHR presentInfo = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderSemaphores_[swapchainImageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain_,
        .pImageIndices = &swapchainImageIndex
    };
    try {
        auto result = graphicsQueue_.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
            resizeRequested = true;
        }
    } catch(const vk::SystemError& e) {
        if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
            resizeRequested = true;
        }
        // FIXME: VK_CHECK() the result now
    }

    frameNumber_++;
}

void Engine::drawBackground(vk::CommandBuffer cmd) {
    /*
    //make a clear-color from frame number. This will flash with a 120 frame period.
    VkClearColorValue clearValue;
    float flash = std::abs(std::sin(_frameNumber / 120.f));
    clearValue = { { 0.0f, 0.0f, flash, 1.0f } };

    VkImageSubresourceRange clearRange = imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);

    //clear image
    vkCmdClearColorImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);
    */

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, gradientPipeline_);

    // Bind the descriptor set containing the draw image for the compute pipeline.
    vk::DescriptorSet tempFIXME = _drawImageDescriptors;
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, gradientPipelineLayout_, 0, 1, &tempFIXME, 0, nullptr);

    cmd.pushConstants(gradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ComputePushConstants), &gradientConstants_);

    // Execute the compute pipeline dispatch. We are using 16 x 16 workgroup size.
    cmd.dispatch(static_cast<uint32_t>(std::ceil(drawExtent_.width / 16.0)), static_cast<uint32_t>(std::ceil(drawExtent_.height / 16.0)), 1);
}

void Engine::drawGeometry(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment = {
        .imageView = drawImage_.imageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingAttachmentInfo depthAttachment = {
        .imageView = depthImage_.imageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = { .depthStencil = { .depth = 0.0f } }
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = { .offset = { 0, 0 }, .extent = drawExtent_ },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment
    };
    cmd.beginRendering(renderingInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, meshPipeline_);

    // Set dynamic viewport and scissor.
    vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(drawExtent_.width),
        .height = static_cast<float>(drawExtent_.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };
    cmd.setViewport(0, viewport);

    vk::Rect2D scissor = {
        .offset = { 0, 0 },
        .extent = drawExtent_
    };
    cmd.setScissor(0, scissor);

    /*//allocate a new uniform buffer for the scene data
    AllocatedBuffer gpuSceneDataBuffer = createBuffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //add it to the deletion queue of this frame so it gets deleted once its been used
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
        });

    //write the buffer
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.info.pMappedData;
    *sceneUniformData = sceneData;

    //create a descriptor set that binds that buffer and update it
    VkDescriptorSet globalDescriptor = getCurrentFrame()._frameDescriptors.allocate(*device_, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(*device_, globalDescriptor);*/

    vk::DescriptorSet tempFIXME = _singleImageDescriptors;
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, meshPipelineLayout_, 0, 1, &tempFIXME, 0, nullptr);

    glm::mat4 view = glm::translate(glm::vec3{ 0, 0, -5 });
    glm::mat4 projection = glm::perspective(glm::radians(70.0f), static_cast<float>(drawExtent_.width) / static_cast<float>(drawExtent_.height), 10000.0f, 0.1f);

    // Invert the Y direction on projection matrix so that we are more similar to OpenGL and gltf axis.
    projection[1][1] *= -1.0f;

    GPUDrawPushConstants pushConstants;
    pushConstants.worldMatrix = projection * view;
    pushConstants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;

    cmd.pushConstants(meshPipelineLayout_, vk::ShaderStageFlagBits::eVertex, 0, sizeof(GPUDrawPushConstants), &pushConstants);
    cmd.bindIndexBuffer(testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, vk::IndexType::eUint32);

    cmd.drawIndexed(testMeshes[2]->surfaces[0].count, 1, testMeshes[2]->surfaces[0].startIndex, 0, 0);

    cmd.endRendering();
}

void Engine::drawImGui(vk::CommandBuffer cmd, vk::ImageView targetImageView)
{
    vk::RenderingAttachmentInfo colorAttachment = {
        .imageView = targetImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = { .offset = { 0, 0 }, .extent = swapchainExtent_ },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment
    };
    cmd.beginRendering(renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    cmd.endRendering();
}

void Engine::immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function) {
    device_.resetFences(*immFence_);

    vk::CommandBuffer cmd = immCommandBuffer_;

    vk::CommandBufferBeginInfo cmdBeginInfo = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    cmd.begin(cmdBeginInfo);

    function(cmd);

    cmd.end();

    vk::CommandBufferSubmitInfo cmdInfo = {
        .commandBuffer = cmd,
        .deviceMask = 0
    };
    vk::SubmitInfo2 submitInfo = {
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo
    };

    // Submit command buffer to the queue and execute it.
    graphicsQueue_.submit2(submitInfo, immFence_);

    device_.waitForFences(*immFence_, vk::True, std::numeric_limits<uint64_t>::max());    // FIXME: need to VK_CHECK() this
}

AllocatedBuffer Engine::createBuffer(size_t allocSize, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    vk::BufferCreateInfo bufferInfo = {
        .size = allocSize,
        .usage = usage
    };

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    AllocatedBuffer newBuffer;
    VkBuffer buffer = {};
    VK_CHECK(vmaCreateBuffer(allocator_, &*bufferInfo, &allocInfo, &buffer, &newBuffer.allocation, &newBuffer.info));
    newBuffer.buffer = buffer;

    return newBuffer;
}

AllocatedImage Engine::createImage(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped) {
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    vk::ImageCreateInfo imageInfo = {
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = size,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage
    };
    if (mipmapped) {
        imageInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // Always allocate images on dedicated GPU memory.
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkImage image = {};
    VK_CHECK(vmaCreateImage(allocator_, &*imageInfo, &allocInfo, &image, &newImage.allocation, nullptr));
    newImage.image = image;

    // If the format is a depth format, we will need to have it use the correct aspect flag.
    vk::ImageAspectFlags aspectFlag = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD32Sfloat) {
        aspectFlag = vk::ImageAspectFlagBits::eDepth;
    }

    // Build an image-view for the image.
    vk::ImageViewCreateInfo viewInfo = {
        .image = newImage.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlag,
            .baseMipLevel = 0,
            .levelCount = imageInfo.mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    newImage.imageView = device_.createImageView(viewInfo);

    return newImage;
}

AllocatedImage Engine::createImage(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped) {
    size_t dataSize = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer = createBuffer(dataSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, dataSize);

    AllocatedImage newImage = createImage(size, format, usage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc, mipmapped);

    immediateSubmit([&size,&uploadbuffer,&newImage](vk::CommandBuffer cmd) {
        transitionImage(cmd, newImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageExtent = size
        };

        cmd.copyBufferToImage(uploadbuffer.buffer, newImage.image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

        transitionImage(cmd, newImage.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    uploadbuffer.clear(allocator_);

    return newImage;
}
