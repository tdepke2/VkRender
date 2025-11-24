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

    message("")
endif()

# FIXME: recommended to use FindVulkan.cmake from Khronos instead: https://docs.vulkan.org/tutorial/latest/02_Development_environment.html
find_package(Vulkan REQUIRED)

find_program(GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)


include(CMakePrintHelpers)
function(add_shaders TARGET_NAME)
    message(STATUS "add_shaders for ${TARGET_NAME}")
    cmake_parse_arguments(SHADER
        "" "" "SOURCES" ${ARGN}
    )
    if(NOT DEFINED SHADER_SOURCES)
        message(FATAL_ERROR "add_shaders() for ${TARGET_NAME} does not have any shaders specified.")
    endif()
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_SOURCE_FILENAME "${SHADER_SOURCE}" NAME)
        set(SHADER_SPIRV "${PROJECT_SOURCE_DIR}/shaders/${SHADER_SOURCE_FILENAME}.spv")
        add_custom_command(
            OUTPUT "${SHADER_SPIRV}"
            COMMAND "${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE} -o ${SHADER_SPIRV}"
            DEPENDS "${GLSLANG_VALIDATOR}"
            COMMENT "Compiling shaders."
            VERBATIM
        )
        cmake_print_variables(SHADER_SOURCE SHADER_SPIRV)
        target_sources(${TARGET_NAME} PRIVATE "${SHADER_SOURCE}")
        target_sources(${TARGET_NAME} PRIVATE "${SHADER_SPIRV}")
    endforeach()
    #add_custom_command(
    #    OUTPUT "${SHADERS_DIR}/slang.spv"
    #    COMMAND ${GLSLANG_VALIDATOR} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
    #    WORKING_DIRECTORY ${SHADERS_DIR}
    #    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    #    COMMENT "Compiling Slang Shaders"
    #    VERBATIM
    #)
    #add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

# Need to look into this more:
# vkguide CMakeLists
# https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/02_Graphics_pipeline_basics/01_Shader_modules.html
# https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/attachments/CMakeLists.txt
# https://www.reddit.com/r/vulkan/comments/1avm0fg/how_to_compile_shaders_with_cmake_that_one_game/


include(cmake/FetchContentArchive.cmake)
#set(FETCHCONTENT_QUIET OFF)

# FIXME: SDL3 has a lot of extra subsystems that are not being used, should disable them.

# Import SDL3 for the window system and other utilities.
message(STATUS "Fetching sources for SDL3...")
FetchContentArchive(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL
    GIT_TAG release-3.2.26
)
message("")

# Import vk-bootstrap for convenient Vulkan initialization.
message(STATUS "Fetching sources for vk-bootstrap...")
FetchContentArchive(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG v1.4.330
)
message("")

# Import VulkanMemoryAllocator for Vulkan memory handling.
message(STATUS "Fetching sources for VulkanMemoryAllocator...")
FetchContentArchive(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.3.0
)
message("")

# Import spdlog for logging utilities.
message(STATUS "Fetching sources for spdlog...")
FetchContentArchive(
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
