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

include(FetchContent)
#set(FETCHCONTENT_QUIET OFF)

# Import spdlog for logging utilities.
#message(STATUS "Fetching sources for spdlog...")
#FetchContent_Declare(
#    spdlog
#    GIT_REPOSITORY https://github.com/gabime/spdlog.git
#    GIT_TAG v1.16.0
#)
#FetchContent_MakeAvailable(spdlog)
#message("")

include(CMakePrintHelpers)

# FIXME: still need a cache option to force online download
# we should also put this in a cmake helper file
function(FetchContentAdvanced)
    message(STATUS "Called FetchContentAdvanced with: ${ARGN}")
    cmake_parse_arguments(PARSED_ARGS
        "" "GIT_TAG;URL_MD5" "" ${ARGN}
    )
    cmake_print_variables(
        PARSED_ARGS_GIT_TAG
        PARSED_ARGS_URL_MD5
        PARSED_ARGS_UNPARSED_ARGUMENTS
        PARSED_ARGS_KEYWORDS_MISSING_VALUES
    )
    list(GET ARGN 0 PARSED_ARGS_NAME)

    if (DEFINED PARSED_ARGS_GIT_TAG)
        string(SUBSTRING "${PARSED_ARGS_GIT_TAG}" 0 16 PARSED_ARGS_VERSION)
    elseif (DEFINED PARSED_ARGS_URL_MD5)
        string(SUBSTRING "${PARSED_ARGS_URL_MD5}" 0 16 PARSED_ARGS_VERSION)
    else()
        message(FATAL_ERROR "FetchContentAdvanced() for ${PARSED_ARGS_NAME} cannot determine version.")
    endif()

    set(ARCHIVE_FILENAME "${PROJECT_SOURCE_DIR}/extern/${PARSED_ARGS_NAME}-${PARSED_ARGS_VERSION}.zip")
    cmake_print_variables(ARCHIVE_FILENAME)
    if (EXISTS "${ARCHIVE_FILENAME}")
        message(STATUS "Archive found")
        cmake_print_variables(FETCHCONTENT_BASE_DIR)
        if (NOT EXISTS "${FETCHCONTENT_BASE_DIR}/${PARSED_ARGS_NAME}-src")
            message(STATUS "Unpacking...")
            file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}/${PARSED_ARGS_NAME}-src")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xvf "${ARCHIVE_FILENAME}" --format=zip
                WORKING_DIRECTORY "${FETCHCONTENT_BASE_DIR}/${PARSED_ARGS_NAME}-src"
            )
        else()
            message(STATUS "Archive already unpacked.")
        endif()
        string(TOUPPER "${PARSED_ARGS_NAME}" PARSED_ARGS_NAME_UPPER)
        set(FETCHCONTENT_SOURCE_DIR_${PARSED_ARGS_NAME_UPPER}
            "${FETCHCONTENT_BASE_DIR}/${PARSED_ARGS_NAME}-src" CACHE PATH
            "When not empty, overrides where to find pre-populated content for ${PARSED_ARGS_NAME}"
        )
        FetchContent_Declare(${ARGN})
        FetchContent_MakeAvailable(${PARSED_ARGS_NAME})
    else()
        message(STATUS "Archive not found, download it")
        FetchContent_Declare(${ARGN})
        FetchContent_MakeAvailable(${PARSED_ARGS_NAME})
        cmake_print_variables(${PARSED_ARGS_NAME}_SOURCE_DIR)

        #file(ARCHIVE_CREATE
        #    OUTPUT "${ARCHIVE_FILENAME}"
        #    PATHS "${${PARSED_ARGS_NAME}_SOURCE_DIR}"
        #    FORMAT gnutar
        #    COMPRESSION GZip
        #    VERBOSE
        #)

        # CMake 3.18 adds a file(ARCHIVE_CREATE ...) option for making a file archive, but there's not yet an option to set the working directory. This method with execute_process can be used instead.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar cvf "${ARCHIVE_FILENAME}" --format=zip "."
            WORKING_DIRECTORY "${${PARSED_ARGS_NAME}_SOURCE_DIR}"
        )
    endif()

endfunction()

# Import spdlog for logging utilities.
message(STATUS "Fetching sources for spdlog...")
FetchContentAdvanced(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.16.0
)
message("")

# Run the next CMakeLists file (builds the executable).
add_subdirectory(src)

# Add tests if this is the main project and testing is enabled.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
