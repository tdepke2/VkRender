# Create the main library from all source files.
add_library(vkr_src
    Engine.cpp
    Engine.h
    VmaUsage.cpp
)

# Specify where to find the header files to include.
target_include_directories(vkr_src PUBLIC .)

# Link to libraries and set C++ compilation version.
# FIXME: we may want this to be private instead, so that main (or external program using this lib) doesn't need to link to the internal dependencies.
target_link_libraries(vkr_src PUBLIC
    Vulkan::Vulkan
    SDL3::SDL3
    vk-bootstrap::vk-bootstrap
    VulkanMemoryAllocator::VulkanMemoryAllocator
    spdlog::spdlog
)
vkr_add_cxx_properties(vkr_src)

add_executable(vk_render
    main.cpp
)
target_link_libraries(vk_render PRIVATE vkr_src)
vkr_add_cxx_properties(vk_render)
