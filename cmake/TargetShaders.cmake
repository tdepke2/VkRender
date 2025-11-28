find_program(GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

# Adds GLSL shaders to a target that will be compiled to SPIR-V at build time.
# The shaders will be recompiled when their source files or any of their include
# files are updated. The compiled SPIR-V output is kept in the build area, but
# the files are also copied back to where the sources live so that the main
# program doesn't need to look in the build area for the shaders.
#
# Most of this is based on the Vulkan-Samples from Khronos Group, with
# improvements made for dependency tracking. See the following for similar
# implementations:
# https://github.com/KhronosGroup/Vulkan-Samples/blob/main/bldsys/cmake/sample_helper.cmake
# https://thatonegamedev.com/cpp/cmake/how-to-compile-shaders-with-cmake/
# https://github.com/tomilov/sah_kd_tree/blob/develop/cmake/Shaders.cmake
# https://stackoverflow.com/questions/71003674/using-glslcs-depfile-to-make-included-files-automatically-trigger-recompile-of
#
function(target_glsl_shaders TARGET_NAME)
    cmake_parse_arguments(SHADER
        "" "BASE_DIR" "SOURCES" ${ARGN}
    )
    if(NOT DEFINED SHADER_SOURCES)
        message(FATAL_ERROR "target_glsl_shaders() for ${TARGET_NAME} does not have any shaders specified.")
    endif()

    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shader-glsl-spv")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    set(OUTPUT_FILES "")

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
        set(SHADER_SOURCE_FULL "${SHADER_BASE_DIR}/${SHADER_SOURCE}")
        get_filename_component(SHADER_FILENAME "${SHADER_SOURCE_FULL}" NAME)
        get_filename_component(SHADER_EXTENSION "${SHADER_SOURCE_FULL}" LAST_EXT)
        get_filename_component(SHADER_DIRECTORY "${SHADER_SOURCE_FULL}" DIRECTORY)
        set(OUTPUT_FILE "${OUTPUT_DIR}/${SHADER_FILENAME}.spv")

        # The MAIN_DEPENDENCY in the call to add_custom_command will associate the shader file with our custom target, so let's organize the files for an IDE.
        source_group(TREE "${SHADER_BASE_DIR}" PREFIX "Shaders" FILES "${SHADER_SOURCE_FULL}")

        # Skip extensions that can't be compiled (include files).
        if(${SHADER_EXTENSION} STREQUAL ".glsl" OR ${SHADER_EXTENSION} STREQUAL ".h")
            continue()
        endif()

        add_custom_command(
            OUTPUT "${OUTPUT_FILE}"
            COMMAND ${GLSLANG_VALIDATOR}
                # Some debug info flags are available for glslangValidator, the sah_kd_tree project on github has an example.
                ARGS --target-env vulkan1.3 --spirv-val "${SHADER_SOURCE_FULL}" -o "${OUTPUT_FILE}" --depfile "${OUTPUT_FILE}.d"
            COMMAND ${CMAKE_COMMAND}
                ARGS -E copy "${OUTPUT_FILE}" "${SHADER_DIRECTORY}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Compiling shader ${SHADER_SOURCE}"
            MAIN_DEPENDENCY "${SHADER_SOURCE_FULL}"
            BYPRODUCTS "${OUTPUT_FILE}.d"
            DEPFILE "${OUTPUT_FILE}.d"
            VERBATIM
        )
        list(APPEND OUTPUT_FILES "${OUTPUT_FILE}")
    endforeach()

    # The custom target will always "build" since custom targets are always out of date, but it doesn't take any action if the dependencies are up to date.
    set(GLSL_TARGET_NAME ${TARGET_NAME}_glsl)
    add_custom_target(${GLSL_TARGET_NAME}
        DEPENDS "${OUTPUT_FILES}"
        COMMENT "All shaders are up to date."
    )

    add_dependencies(${TARGET_NAME} ${GLSL_TARGET_NAME})
endfunction()
