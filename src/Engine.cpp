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

void copyImageToImage(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize)
{
    VkImageBlit2 blitRegion{ .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2, .pNext = nullptr };

    blitRegion.srcOffsets[1].x = srcSize.width;
    blitRegion.srcOffsets[1].y = srcSize.height;
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = dstSize.width;
    blitRegion.dstOffsets[1].y = dstSize.height;
    blitRegion.dstOffsets[1].z = 1;

    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcSubresource.mipLevel = 0;

    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstSubresource.mipLevel = 0;

    VkBlitImageInfo2 blitInfo{ .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2, .pNext = nullptr };
    blitInfo.dstImage = destination;
    blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blitInfo.srcImage = source;
    blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blitInfo.filter = VK_FILTER_LINEAR;
    blitInfo.regionCount = 1;
    blitInfo.pRegions = &blitRegion;

    vkCmdBlitImage2(cmd, &blitInfo);
}

}

void Engine::init() {
    // We initialize SDL and create a window with it.
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

    //main loop
    while (!closeWindow) {
        //Handle events on queue
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);

            //close the window when user alt-f4s or clicks the X button
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
            //throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resizeRequested) {
            resizeSwapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // some imgui UI to test
        //ImGui::ShowDemoWindow();

        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.0f);
            ImGui::InputFloat4("data1", reinterpret_cast<float*>(&gradientConstants_.data1));
            ImGui::InputFloat4("data2", reinterpret_cast<float*>(&gradientConstants_.data2));
            ImGui::InputFloat4("data3", reinterpret_cast<float*>(&gradientConstants_.data3));
            ImGui::InputFloat4("data4", reinterpret_cast<float*>(&gradientConstants_.data4));
        }
        ImGui::End();

        // make imgui calculate internal draw structures
        ImGui::Render();

        draw();
    }
}

void Engine::cleanup() {
    //make sure the gpu has stopped doing its things
    vkDeviceWaitIdle(*device_);

    for (auto& mesh : testMeshes) {
        destroyBuffer(mesh->meshBuffers.indexBuffer);
        destroyBuffer(mesh->meshBuffers.vertexBuffer);
    }

    _defaultSamplerNearest.clear();
    _defaultSamplerLinear.clear();

    destroyImage(_whiteImage);
    destroyImage(_greyImage);
    destroyImage(_blackImage);
    destroyImage(_errorCheckerboardImage);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(*device_, imguiPool, nullptr);

    meshPipelineLayout_.clear();//vkDestroyPipelineLayout(*device_, _meshPipelineLayout, nullptr);
    meshPipeline_.clear();//vkDestroyPipeline(*device_, _meshPipeline, nullptr);

    gradientPipelineLayout_.clear();//vkDestroyPipelineLayout(*device_, _gradientPipelineLayout, nullptr);
    gradientPipeline_.clear();//vkDestroyPipeline(*device_, _gradientPipeline, nullptr);

    /*for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        _frames[i]._frameDescriptors.destroy_pools(*device_);
    }*/

    globalDescriptorAllocator.destroy_pool(*device_);

    vkDestroyDescriptorSetLayout(*device_, _drawImageDescriptorLayout, nullptr);

    immFence_.clear();//vkDestroyFence(*device_, _immFence, nullptr);

    immCommandBuffer_.clear();
    immCommandPool_.clear();//vkDestroyCommandPool(*device_, _immCommandPool, nullptr);

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        _frames[i].mainCommandBuffer.clear();
        _frames[i].commandPool.clear();//vkDestroyCommandPool(*device_, _frames[i]._commandPool, nullptr);

        //destroy sync objects
        _frames[i].renderFence.clear();//vkDestroyFence(*device_, _frames[i]._renderFence, nullptr);
        _frames[i].renderSemaphore.clear();//vkDestroySemaphore(*device_, _frames[i]._renderSemaphore, nullptr);
        _frames[i].swapchainSemaphore.clear();//vkDestroySemaphore(*device_ ,_frames[i]._swapchainSemaphore, nullptr);
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

    //create vertex buffer
    newSurface.vertexBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(*device_, &deviceAdressInfo);

    //create index buffer
    newSurface.indexBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);    // FIXME: vma flag is deprecated.

    AllocatedBuffer staging = createBuffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.info.pMappedData;

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroyBuffer(staging);

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

    const auto imageViews = vkbSwapchain.get_image_views().value();
    std::transform(imageViews.begin(), imageViews.end(), std::back_inserter(swapchainImageViews_), [this](VkImageView v) {
        return vk::raii::ImageView(device_, v);
    });
}

void Engine::resizeSwapchain() {
    device_.waitIdle();

    destroySwapchain();

    int width, height;
    SDL_GetWindowSize(window_, &width, &height);
    windowExtent_.width = width;
    windowExtent_.height = height;

    createSwapchain(windowExtent_.width, windowExtent_.height);

    resizeRequested = false;
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
        _frames[i].commandPool = device_.createCommandPool(commandPoolInfo);

        // Allocate the default command buffer that we will use for rendering.
        vk::CommandBufferAllocateInfo cmdAllocInfo = {
            .commandPool = _frames[i].commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        _frames[i].mainCommandBuffer = std::move(device_.allocateCommandBuffers(cmdAllocInfo).front());
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
    // One fence to control when the gpu has finished rendering the frame.
    // Two semaphores to synchronize rendering with swapchain.
    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        _frames[i].swapchainSemaphore = vk::raii::Semaphore(device_, vk::SemaphoreCreateInfo());
        _frames[i].renderSemaphore = vk::raii::Semaphore(device_, vk::SemaphoreCreateInfo());
        _frames[i].renderFence = vk::raii::Fence(device_, { .flags = vk::FenceCreateFlagBits::eSignaled });
    }

    immFence_ = vk::raii::Fence(device_, { .flags = vk::FenceCreateFlagBits::eSignaled });
}

void Engine::initDescriptors() {
    //create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    globalDescriptorAllocator.init_pool(*device_, 10, sizes);

    //make the descriptor set layout for our compute draw
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(*device_, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    //allocate a descriptor set for our draw image
    _drawImageDescriptors = globalDescriptorAllocator.allocate(*device_, _drawImageDescriptorLayout);

    DescriptorWriter writer;
    writer.write_image(0, *drawImage_.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    writer.update_set(*device_, _drawImageDescriptors);

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
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(*device_, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    _singleImageDescriptors = globalDescriptorAllocator.allocate(*device_, _singleImageDescriptorLayout);
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
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE },
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 0;
    for (VkDescriptorPoolSize& pool_size : pool_sizes) {
        pool_info.maxSets += pool_size.descriptorCount;
    }
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(*device_, &pool_info, nullptr, &imguiPool));

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window_);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *instance_;
    init_info.PhysicalDevice = *physicalDevice_;
    init_info.Device = *device_;
    init_info.QueueFamily = graphicsQueueFamily_;
    init_info.Queue = *graphicsQueue_;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.UseDynamicRendering = true;
    // FIXME: should specify vulkanApiVersion in here.

    // Dynamic rendering parameters for imgui to use
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {};
    pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    VkFormat colorAttachmentFormat = static_cast<VkFormat>(swapchainImageFormat_);
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &colorAttachmentFormat;

    init_info.PipelineInfoMain = {};
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingCreateInfo;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
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
    writer.write_image(0, *_errorCheckerboardImage.imageView, *_defaultSamplerNearest, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(*device_, _singleImageDescriptors);
}

void Engine::draw() {
    // Wait until the gpu has finished rendering the last frame, timeout of 1 second.
    device_.waitForFences(*getCurrentFrame().renderFence, vk::True, 1000000000);    // FIXME: need to VK_CHECK() this
    device_.resetFences(*getCurrentFrame().renderFence);

    //getCurrentFrame()._frameDescriptors.clear_pools(*device_);

    // Request image from the swapchain.
    uint32_t swapchainImageIndex;
    try {
        auto result = swapchain_.acquireNextImage(1000000000, getCurrentFrame().swapchainSemaphore, nullptr);
        if (result.first == vk::Result::eErrorOutOfDateKHR || result.first == vk::Result::eSuboptimalKHR) {
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

    drawExtent_.width = std::min(swapchainExtent_.width, drawImage_.imageExtent.width) * renderScale;
    drawExtent_.height = std::min(swapchainExtent_.height, drawImage_.imageExtent.height) * renderScale;

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
    vk::SemaphoreSubmitInfo signalInfo = {
        .semaphore = getCurrentFrame().renderSemaphore,
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
    // visible window. We want to wait on the renderSemaphore for that, as it's
    // necessary that drawing commands have finished before the image is
    // displayed to the user.
    vk::PresentInfoKHR presentInfo = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*getCurrentFrame().renderSemaphore,
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

    _frameNumber++;
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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *gradientPipelineLayout_, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, *gradientPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &gradientConstants_);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(drawExtent_.width / 16.0), std::ceil(drawExtent_.height / 16.0), 1);
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

    //bind a texture
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, *meshPipelineLayout_, 0, 1, &_singleImageDescriptors, 0, nullptr);

    glm::mat4 view = glm::translate(glm::vec3{ 0,0,-5 });
    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)drawExtent_.width / (float)drawExtent_.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    projection[1][1] *= -1;

    GPUDrawPushConstants push_constants;
    push_constants.worldMatrix = projection * view;
    push_constants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;

    vkCmdPushConstants(cmd, *meshPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
    vkCmdBindIndexBuffer(cmd, testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

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

AllocatedBuffer Engine::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    // allocate buffer
    VkBufferCreateInfo bufferInfo = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(allocator_, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
        &newBuffer.info));

    return newBuffer;
}

void Engine::destroyBuffer(const AllocatedBuffer& buffer) {
    vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
}

AllocatedImage Engine::createImage(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    vk::ImageCreateInfo img_info = {
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
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // allocate and create the image
    VkImage image = {};
    VK_CHECK(vmaCreateImage(allocator_, &(*img_info), &allocinfo, &image, &newImage.allocation, nullptr));
    newImage.image = image;

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    vk::ImageAspectFlags aspectFlag = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD32Sfloat) {
        aspectFlag = vk::ImageAspectFlagBits::eDepth;
    }

    // build a image-view for the image
    vk::ImageViewCreateInfo view_info = {
        .image = newImage.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlag,
            .baseMipLevel = 0,
            .levelCount = img_info.mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    newImage.imageView = device_.createImageView(view_info);

    return newImage;
}

AllocatedImage Engine::createImage(void* data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage, bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer = createBuffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = createImage(size, format, usage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc, mipmapped);

    immediateSubmit([&](VkCommandBuffer cmd) {
        transitionImage(cmd, new_image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copyRegion);

        transitionImage(cmd, new_image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        });

    destroyBuffer(uploadbuffer);

    return new_image;
}

void Engine::destroyImage(AllocatedImage& img)
{
    img.imageView.clear();
    vmaDestroyImage(allocator_, img.image, img.allocation);
}
