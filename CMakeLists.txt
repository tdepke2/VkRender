cmake_minimum_required(VERSION 3.15...3.30)

include(cmake/CompileProperties.cmake)

# Set project name and version.
project(VkRender
    VERSION 1.0
    DESCRIPTION "Render engine using Vulkan"
    LANGUAGES CXX
)

# Check if this is the main project (not included with add_subdirectory).
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Set default build type if unspecified.
    # From https://cliutils.gitlab.io/modern-cmake/chapters/features.html
    set(DEFAULT_BUILD_TYPE "Release")
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE
        )
        # Set the possible values of build type for cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
        )
    elseif(CMAKE_BUILD_TYPE)
        message(STATUS "Current build type is '${CMAKE_BUILD_TYPE}'.")
    elseif(CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Build type is not set, select it during build with '--config' option and one of the following: '${CMAKE_CONFIGURATION_TYPES}'.")
    endif()

    # Place binaries in lib/ or bin/ respectively instead of in the sources directory.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    # Have CMake create a "compile_commands.json" file for clangd.
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Enable support for folders in IDEs.
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Enable testing by default.
    option(BUILD_TESTING "Build the testing tree." ON)
    message(STATUS "Building tests is '${BUILD_TESTING}'.")
    if (BUILD_TESTING)
        enable_testing()
    endif()

    # SDL3 settings.
    set(SDL_EXAMPLES OFF)

    set(SDL_AUDIO OFF)
    set(SDL_CAMERA OFF)
    set(SDL_DIALOG OFF)
    set(SDL_GPU OFF)
    set(SDL_HAPTIC OFF)
    set(SDL_HIDAPI OFF)
    set(SDL_JOYSTICK OFF)
    set(SDL_POWER OFF)
    set(SDL_RENDER OFF)
    set(SDL_SENSOR OFF)
    set(SDL_VIDEO ON)

    message("")
endif()

# FIXME: recommended to use FindVulkan.cmake from Khronos instead: https://docs.vulkan.org/tutorial/latest/02_Development_environment.html
find_package(Vulkan REQUIRED)

include(cmake/FetchArchive.cmake)
#set(FETCHCONTENT_QUIET OFF)

# Import Simple DirectMedia Layer for the window system and other utilities.
fetch_archive_declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG release-3.2.26
)
list(APPEND FETCH_PENDING "SDL3")

# Import vk-bootstrap for convenient Vulkan initialization.
fetch_archive_declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG v1.4.330
)
list(APPEND FETCH_PENDING "vk-bootstrap")

# Import VulkanMemoryAllocator for Vulkan memory handling.
fetch_archive_declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.3.0
)
list(APPEND FETCH_PENDING "VulkanMemoryAllocator")

# Import spdlog for logging utilities.
fetch_archive_declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.16.0
)
list(APPEND FETCH_PENDING "spdlog")

# Import Dear ImGui for GUI library.
fetch_archive_declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.92.5
)
list(APPEND FETCH_PENDING "imgui")

# Import OpenGL Mathematics for vector/matrix types.
fetch_archive_declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG 1.0.2
)
list(APPEND FETCH_PENDING "glm")

# Import fastgltf for glTF 2.0 model loading.
fetch_archive_declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf
    GIT_TAG v0.9.0
    GIT_SHALLOW TRUE    # FIXME: found this from fastgltf cmake scripts, should use for all dependencies? they also set `SOURCE_SUBDIR doesnt-exist` and then manually define the target which is interesting.
)
list(APPEND FETCH_PENDING "fastgltf")

# Populate all of the pending content.
foreach(CONTENT_NAME IN LISTS FETCH_PENDING)
    message(STATUS "Fetching sources for ${CONTENT_NAME}...")
    fetch_archive_make_available(${CONTENT_NAME})
    message("")
endforeach()
unset(FETCH_PENDING)

include(cmake/ImGuiVulkan.cmake)

# Run the next CMakeLists file (builds the executable).
add_subdirectory(src)

# Add tests if this is the main project and testing is enabled.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
