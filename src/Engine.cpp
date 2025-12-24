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

VkImageSubresourceRange imageSubresourceRange(VkImageAspectFlags aspectMask) {
    VkImageSubresourceRange subImage {};
    subImage.aspectMask = aspectMask;
    subImage.baseMipLevel = 0;
    subImage.levelCount = VK_REMAINING_MIP_LEVELS;
    subImage.baseArrayLayer = 0;
    subImage.layerCount = VK_REMAINING_ARRAY_LAYERS;

    return subImage;
}

void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier2 imageBarrier {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    imageBarrier.pNext = nullptr;

    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    imageBarrier.oldLayout = currentLayout;
    imageBarrier.newLayout = newLayout;

    VkImageAspectFlags aspectMask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange = imageSubresourceRange(aspectMask);
    imageBarrier.image = image;

    VkDependencyInfo depInfo {};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.pNext = nullptr;

    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

VkSemaphoreSubmitInfo semaphoreSubmitInfo(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore) {
    VkSemaphoreSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.semaphore = semaphore;
    submitInfo.stageMask = stageMask;
    submitInfo.deviceIndex = 0;
    submitInfo.value = 1;

    return submitInfo;
}

VkCommandBufferSubmitInfo commandBufferSubmitInfo(VkCommandBuffer cmd) {
    VkCommandBufferSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    info.pNext = nullptr;
    info.commandBuffer = cmd;
    info.deviceMask = 0;

    return info;
}

VkSubmitInfo2 submitInfo(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo, VkSemaphoreSubmitInfo* waitSemaphoreInfo) {
    VkSubmitInfo2 info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.pNext = nullptr;

    info.waitSemaphoreInfoCount = waitSemaphoreInfo == nullptr ? 0 : 1;
    info.pWaitSemaphoreInfos = waitSemaphoreInfo;

    info.signalSemaphoreInfoCount = signalSemaphoreInfo == nullptr ? 0 : 1;
    info.pSignalSemaphoreInfos = signalSemaphoreInfo;

    info.commandBufferInfoCount = 1;
    info.pCommandBufferInfos = cmd;

    return info;
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

VkRenderingAttachmentInfo attachment_info(
    VkImageView view, VkClearValue* clear ,VkImageLayout layout /*= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL*/)
{
    VkRenderingAttachmentInfo colorAttachment {};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.pNext = nullptr;

    colorAttachment.imageView = view;
    colorAttachment.imageLayout = layout;
    colorAttachment.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (clear) {
        colorAttachment.clearValue = *clear;
    }

    return colorAttachment;
}

VkRenderingAttachmentInfo depth_attachment_info(
    VkImageView view, VkImageLayout layout /*= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL*/)
{
    VkRenderingAttachmentInfo depthAttachment {};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.pNext = nullptr;

    depthAttachment.imageView = view;
    depthAttachment.imageLayout = layout;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 0.f;

    return depthAttachment;
}

VkRenderingInfo rendering_info(VkExtent2D renderExtent, VkRenderingAttachmentInfo* colorAttachment,
    VkRenderingAttachmentInfo* depthAttachment)
{
    VkRenderingInfo renderInfo {};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.pNext = nullptr;

    renderInfo.renderArea = VkRect2D { VkOffset2D { 0, 0 }, renderExtent };
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = colorAttachment;
    renderInfo.pDepthAttachment = depthAttachment;
    renderInfo.pStencilAttachment = nullptr;

    return renderInfo;
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

    vkDestroyFence(*device_, _immFence, nullptr);

    vkDestroyCommandPool(*device_, _immCommandPool, nullptr);

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        vkDestroyCommandPool(*device_, _frames[i]._commandPool, nullptr);

        //destroy sync objects
        vkDestroyFence(*device_, _frames[i]._renderFence, nullptr);
        vkDestroySemaphore(*device_, _frames[i]._renderSemaphore, nullptr);
        vkDestroySemaphore(*device_ ,_frames[i]._swapchainSemaphore, nullptr);
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
    //create a command pool for commands submitted to the graphics queue.
    //we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo =  {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.pNext = nullptr;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = graphicsQueueFamily_;

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateCommandPool(*device_, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = {};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.pNext = nullptr;
        cmdAllocInfo.commandPool = _frames[i]._commandPool;
        cmdAllocInfo.commandBufferCount = 1;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(*device_, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    VK_CHECK(vkCreateCommandPool(*device_, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.pNext = nullptr;

    cmdAllocInfo.commandPool = _immCommandPool;
    cmdAllocInfo.commandBufferCount = 1;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VK_CHECK(vkAllocateCommandBuffers(*device_, &cmdAllocInfo, &_immCommandBuffer));
}

void Engine::initSyncStructures() {
    //create synchronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to synchronize rendering with swapchain
    //we want the fence to start signaled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(*device_, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(*device_, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(*device_, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }

    VK_CHECK(vkCreateFence(*device_, &fenceCreateInfo, nullptr, &_immFence));
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
    // wait until the gpu has finished rendering the last frame. Timeout of 1
    // second
    VK_CHECK(vkWaitForFences(*device_, 1, &getCurrentFrame()._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(*device_, 1, &getCurrentFrame()._renderFence));

    //getCurrentFrame()._frameDescriptors.clear_pools(*device_);

    //request image from the swapchain
    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(*device_, *swapchain_, 1000000000, getCurrentFrame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resizeRequested = true;
        return;
    }

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = getCurrentFrame()._mainCommandBuffer;

    // now that we are sure that the commands finished executing, we can safely
    // reset the command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    _drawExtent.width = std::min(swapchainExtent_.width, drawImage_.imageExtent.width) * renderScale;;
    _drawExtent.height = std::min(swapchainExtent_.height, drawImage_.imageExtent.height) * renderScale;

    //start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    transitionImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    drawBackground(cmd);

    transitionImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transitionImage(cmd, depthImage_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    drawGeometry(cmd);

    //transition the draw image and the swapchain image into their correct transfer layouts
    transitionImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    copyImageToImage(cmd, drawImage_.image, swapchainImages_[swapchainImageIndex], _drawExtent, swapchainExtent_);

    // set swapchain image layout to Attachment Optimal so we can draw it
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // draw imgui into the swapchain image
    drawImGui(cmd, *swapchainImageViews_[swapchainImageIndex]);

    // set swapchain image layout to Present so we can draw it
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue.
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmdinfo = commandBufferSubmitInfo(cmd);

    VkSemaphoreSubmitInfo waitInfo = semaphoreSubmitInfo(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,getCurrentFrame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = semaphoreSubmitInfo(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, getCurrentFrame()._renderSemaphore);

    VkSubmitInfo2 submit = submitInfo(&cmdinfo,&signalInfo,&waitInfo);

    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(*graphicsQueue_, 1, &submit, getCurrentFrame()._renderFence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that,
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkSwapchainKHR swapchain = *swapchain_;
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &getCurrentFrame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(*graphicsQueue_, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resizeRequested = true;
    }

    //increase the number of frames drawn
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

    // bind the gradient drawing compute pipeline
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, gradientPipeline_);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *gradientPipelineLayout_, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, *gradientPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &gradientConstants_);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void Engine::drawGeometry(vk::CommandBuffer cmd) {
    VkRenderingAttachmentInfo colorAttachment = attachment_info(*drawImage_.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = depth_attachment_info(*depthImage_.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, meshPipeline_);

    //set dynamic viewport and scissor
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;

    vkCmdSetScissor(cmd, 0, 1, &scissor);

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
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_drawExtent.width / (float)_drawExtent.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    projection[1][1] *= -1;

    GPUDrawPushConstants push_constants;
    push_constants.worldMatrix = projection * view;
    push_constants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;

    vkCmdPushConstants(cmd, *meshPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
    vkCmdBindIndexBuffer(cmd, testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, testMeshes[2]->surfaces[0].count, 1, testMeshes[2]->surfaces[0].startIndex, 0, 0);

    vkCmdEndRendering(cmd);
}

void Engine::drawImGui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = rendering_info(swapchainExtent_, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void Engine::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function) {
    VK_CHECK(vkResetFences(*device_, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo{};
    cmdinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdinfo.pNext = nullptr;
    cmdinfo.commandBuffer = cmd;
    cmdinfo.deviceMask = 0;

    VkSubmitInfo2 submit = submitInfo(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(*graphicsQueue_, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(*device_, 1, &_immFence, true, 9999999999));
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
        transitionImage(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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

        transitionImage(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

    destroyBuffer(uploadbuffer);

    return new_image;
}

void Engine::destroyImage(AllocatedImage& img)
{
    img.imageView.clear();
    vmaDestroyImage(allocator_, img.image, img.allocation);
}
