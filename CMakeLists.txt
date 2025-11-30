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

include(cmake/FetchContentArchive.cmake)
#set(FETCHCONTENT_QUIET OFF)

# Import SDL3 for the window system and other utilities.
message(STATUS "Fetching sources for SDL3...")
fetch_content_archive(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG release-3.2.26
)
message("")

# Import vk-bootstrap for convenient Vulkan initialization.
message(STATUS "Fetching sources for vk-bootstrap...")
fetch_content_archive(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG v1.4.330
)
message("")

# Import VulkanMemoryAllocator for Vulkan memory handling.
message(STATUS "Fetching sources for VulkanMemoryAllocator...")
fetch_content_archive(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.3.0
)
message("")

# Import spdlog for logging utilities.
message(STATUS "Fetching sources for spdlog...")
fetch_content_archive(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.16.0
)
message("")

# Run the next CMakeLists file (builds the executable).
add_subdirectory(src)

# Add tests if this is the main project and testing is enabled.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
