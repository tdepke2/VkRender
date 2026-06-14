set(VKR_COMPONENTS_SRCS
    components/Camera.h
    components/Renderable.h
    components/Transform.h
)

# Create the main library from all source files.
add_library(vkr_src
    ${VKR_COMPONENTS_SRCS}
    CameraInstance.cpp
    CameraInstance.h
    Common.h
    ComponentArray.h
    ConsoleVars.cpp
    ConsoleVars.h
    Descriptors.cpp
    Descriptors.h
    DisableWarning.h
    Engine.cpp
    Engine.h
    GlmFormatters.h
    Loader.cpp
    Loader.h
    Pipelines.cpp
    Pipelines.h
    Scene.cpp
    Scene.h
    SceneView.h
    StringHash.h
    TransformInstance.cpp
    TransformInstance.h
    View.cpp
    View.h
    VmaUsage.cpp
    VmaUsage.h
)

# Organize source and header files for IDEs.
source_group("Header Files/components" REGULAR_EXPRESSION "components/.*\.h")
source_group("Source Files/components" REGULAR_EXPRESSION "components/.*\.cpp")

# Specify where to find the header files to include.
target_include_directories(vkr_src PUBLIC .)

# Link to libraries and set C++ compilation version.
# FIXME: we may want this to be private instead, so that main (or external program using this lib) doesn't need to link to the internal dependencies.
target_link_libraries(vkr_src PUBLIC
    Vulkan::Vulkan
    SDL3::SDL3
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    spdlog::spdlog
    imgui-vulkan::imgui-vulkan
    glm::glm
    fastgltf::fastgltf
)
vkr_add_cxx_properties(vkr_src)

target_compile_definitions(vkr_src PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE     # The OpenGL clip space is -1 to 1 in z-axis, we want 0 to 1.
    VULKAN_HPP_NO_CONSTRUCTORS      # Don't use the constructors for Vulkan types, we will use designated initializers from C++20.
)

add_executable(vk_render
    main.cpp
)
target_link_libraries(vk_render PRIVATE vkr_src)
vkr_add_cxx_properties(vk_render)

include(../cmake/TargetShaders.cmake)

target_glsl_shaders(vk_render BASE_DIR "${CMAKE_SOURCE_DIR}/shaders" SOURCES
    "gradient.comp"
    "gradient_color.comp"
    "colored_triangle.frag"
    "colored_triangle.vert"
    "colored_triangle_mesh.vert"
    "tex_image.frag"
)
