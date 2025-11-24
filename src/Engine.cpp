#include <Engine.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <VkBootstrap.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <thread>

constexpr bool bUseValidationLayers = true;

namespace {

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

}

void Engine::init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    // FIXME: recommended to use SDL_SetAppMetadata() after startup, see SDL api reference
    // FIXME: need to call SDL_Quit() at end?

    SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    window_ = SDL_CreateWindow("Vulkan Engine", windowExtent_.width, windowExtent_.height, window_flags);

    initVulkan();

    initSwapchain();

    initCommands();

    initSyncStructures();
}

void Engine::run() {
    SDL_Event e;
    bool bQuit = false;

    //main loop
    while (!bQuit)
    {
        //Handle events on queue
        while (SDL_PollEvent(&e))
        {
            //close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_EVENT_QUIT) bQuit = true;

            if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
                freeze_rendering = true;
            } else if (e.type == SDL_EVENT_WINDOW_RESTORED) {
                freeze_rendering = false;
            }
        }

        if (freeze_rendering) {
            //throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}

void Engine::cleanup() {
    //make sure the gpu has stopped doing its things
    vkDeviceWaitIdle(*device_);

    for (unsigned int i = 0; i < FRAME_OVERLAP; i++) {
        vkDestroyCommandPool(*device_, _frames[i]._commandPool, nullptr);

        //destroy sync objects
        vkDestroyFence(*device_, _frames[i]._renderFence, nullptr);
        vkDestroySemaphore(*device_, _frames[i]._renderSemaphore, nullptr);
        vkDestroySemaphore(*device_ ,_frames[i]._swapchainSemaphore, nullptr);
    }

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

void Engine::initVulkan() {
    vkb::InstanceBuilder builder;

    // Make the Vulkan instance, with basic debug features.
    auto instRet = builder.set_app_name("Example Vulkan Application")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)    // FIXME: move to vulkan 1.4? see: https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/00_Setup/01_Instance.html
        .build();

    vkb::Instance vkbInst = instRet.value();

    // Grab the instance.
    instance_ = vk::raii::Instance(context_, vkbInst.instance);
    debugMessenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, vkbInst.debug_messenger);

    VkSurfaceKHR surfaceHandle;
    SDL_Vulkan_CreateSurface(window_, *instance_, nullptr, &surfaceHandle);
    surface_ = vk::raii::SurfaceKHR(instance_, surfaceHandle);

    //vulkan 1.3 features
    vk::PhysicalDeviceVulkan13Features features13 = {
        .synchronization2 = true,
        .dynamicRendering = true,
    };

    //vulkan 1.2 features
    vk::PhysicalDeviceVulkan12Features features12 = {
        .descriptorIndexing = true,
        .bufferDeviceAddress = true,
    };

    //use vkbootstrap to select a gpu.
    //We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
    vkb::PhysicalDeviceSelector selector{ vkbInst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(*surface_)
        .select()
        .value();

    //create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    physicalDevice_ = vk::raii::PhysicalDevice(instance_, physicalDevice.physical_device);
    device_ = vk::raii::Device(physicalDevice_, vkbDevice.device);

    // use vkbootstrap to get a Graphics queue
    graphicsQueue_ = vk::raii::Queue(device_, vkbDevice.get_queue(vkb::QueueType::graphics).value());
    graphicsQueueFamily_ = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = *physicalDevice_;
    allocatorInfo.device = *device_;
    allocatorInfo.instance = *instance_;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &allocator_);
}

void Engine::initSwapchain() {
    createSwapchain(windowExtent_.width, windowExtent_.height);

    //draw image size will match the window
    vk::Extent3D drawImageExtent = {
        windowExtent_.width,
        windowExtent_.height,
        1
    };

    //hardcoding the draw format to 32 bit float
    drawImage_.imageFormat = vk::Format::eR16G16B16A16Sfloat;
    drawImage_.imageExtent = drawImageExtent;

    vk::ImageUsageFlags drawImageUsages =
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eColorAttachment;

    vk::ImageCreateInfo rimg_info = {
        .imageType = vk::ImageType::e2D,
        .format = drawImage_.imageFormat,
        .extent = drawImage_.imageExtent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = drawImageUsages
    };

    //for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    VkImage image = {};
    vmaCreateImage(allocator_, &(*rimg_info), &rimg_allocinfo, &image, &drawImage_.allocation, nullptr);
    drawImage_.image = image;

    //build a image-view for the draw image to use for rendering
    vk::ImageViewCreateInfo rview_info = {
        .image = drawImage_.image,
        .viewType = vk::ImageViewType::e2D,
        .format = drawImage_.imageFormat,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    drawImage_.imageView = device_.createImageView(rview_info);
}

void Engine::createSwapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{ *physicalDevice_, *device_, *surface_ };

    vk::SurfaceFormatKHR surfaceFormat = {
        .format = vk::Format::eB8G8R8A8Unorm,
        .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
    };

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(surfaceFormat)
        //use vsync present mode
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    //store swapchain and its related images
    swapchain_ = vk::raii::SwapchainKHR(device_, vkbSwapchain.swapchain);
    swapchainImageFormat_ = surfaceFormat.format;
    swapchainExtent_ = vkbSwapchain.extent;
    swapchainImages_ = swapchain_.getImages();//= vkbSwapchain.get_images().value();
    const auto imageViews = vkbSwapchain.get_image_views().value();
    std::transform(imageViews.begin(), imageViews.end(), std::back_inserter(swapchainImageViews_), [this](VkImageView v) {
        return vk::raii::ImageView(device_, v);
    });
}

void Engine::destroySwapchain() {
    swapchainImageViews_.clear();
    swapchain_.clear();//vkDestroySwapchainKHR(*device_, _swapchain, nullptr);

    // destroy swapchain resources
    //for (size_t i = 0; i < _swapchainImageViews.size(); i++) {
    //    vkDestroyImageView(*device_, _swapchainImageViews[i], nullptr);
    //}
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
}

void Engine::draw() {
    // wait until the gpu has finished rendering the last frame. Timeout of 1
    // second
    VK_CHECK(vkWaitForFences(*device_, 1, &getCurrentFrame()._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(*device_, 1, &getCurrentFrame()._renderFence));

    //request image from the swapchain
    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(*device_, *swapchain_, 1000000000, getCurrentFrame()._swapchainSemaphore, nullptr, &swapchainImageIndex));

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

    _drawExtent.width = drawImage_.imageExtent.width;
    _drawExtent.height = drawImage_.imageExtent.height;

    //start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    transitionImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    drawBackground(cmd);

    //transition the draw image and the swapchain image into their correct transfer layouts
    transitionImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    copyImageToImage(cmd, drawImage_.image, swapchainImages_[swapchainImageIndex], _drawExtent, swapchainExtent_);

    // set swapchain image layout to Present so we can show it on the screen
    transitionImage(cmd, swapchainImages_[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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

    VK_CHECK(vkQueuePresentKHR(*graphicsQueue_, &presentInfo));

    //increase the number of frames drawn
    _frameNumber++;
}

void Engine::drawBackground(vk::CommandBuffer cmd) {
    //make a clear-color from frame number. This will flash with a 120 frame period.
    VkClearColorValue clearValue;
    float flash = std::abs(std::sin(_frameNumber / 120.f));
    clearValue = { { 0.0f, 0.0f, flash, 1.0f } };

    VkImageSubresourceRange clearRange = imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);

    //clear image
    vkCmdClearColorImage(cmd, drawImage_.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);
}
