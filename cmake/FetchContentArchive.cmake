include(FetchContent)

option(FETCHCONTENTARCHIVE_FORCE_DOWNLOAD "Prevent usage of local archive, content will always be downloaded" OFF)

# Provides an alternative for the FetchContent module that obtains a package
# from a local archive instead of from an online repository. When a new version
# of a package is requested, it will be downloaded and archived in "extern/".
# The archive can be checked into this repository and then future builds will
# use this local copy. This makes the build reliable as online repositories can
# go away, but it's useful to know where a certain package came from and makes
# it easy to upgrade the version.
#
# The FETCHCONTENTARCHIVE_FORCE_DOWNLOAD option can be used to override the
# archive behavior, and make this function behave like the normal FetchContent
# functions.
#FIXME: should rename the function to be consistent with underscore_naming_style
function(FetchContentArchive)
    if(NOT FETCHCONTENT_QUIET)
        message(STATUS "Running FetchContentArchive() with: ${ARGN}")
    endif()
    cmake_parse_arguments(PARSED_ARGS
        "" "GIT_TAG;URL_MD5" "" ${ARGN}
    )
    list(GET ARGN 0 PARSED_ARGS_NAME)
    string(TOLOWER "${PARSED_ARGS_NAME}" PARSED_ARGS_NAME_LOWER)
    string(TOUPPER "${PARSED_ARGS_NAME}" PARSED_ARGS_NAME_UPPER)

    if(DEFINED PARSED_ARGS_GIT_TAG)
        string(SUBSTRING "${PARSED_ARGS_GIT_TAG}" 0 16 PARSED_ARGS_VERSION)
    elseif(DEFINED PARSED_ARGS_URL_MD5)
        string(SUBSTRING "${PARSED_ARGS_URL_MD5}" 0 16 PARSED_ARGS_VERSION)
    else()
        message(FATAL_ERROR "FetchContentArchive() for ${PARSED_ARGS_NAME} cannot determine version.")
    endif()

    set(ARCHIVE_FILENAME "${PROJECT_SOURCE_DIR}/extern/${PARSED_ARGS_NAME}-${PARSED_ARGS_VERSION}.zip")
    if(FETCHCONTENT_QUIET)
        set(ARCHIVE_OPTIONS "f")
    else()
        set(ARCHIVE_OPTIONS "vf")
    endif()
    set(SOURCE_DIR_EXPECTED "${FETCHCONTENT_BASE_DIR}/${PARSED_ARGS_NAME_LOWER}-src")

    if(NOT FETCHCONTENTARCHIVE_FORCE_DOWNLOAD AND EXISTS "${ARCHIVE_FILENAME}")
        if(NOT EXISTS "${SOURCE_DIR_EXPECTED}")
            if(NOT FETCHCONTENT_QUIET)
                message(STATUS "Archive found, unpacking: ${ARCHIVE_FILENAME}")
            endif()
            file(MAKE_DIRECTORY "${SOURCE_DIR_EXPECTED}")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar "x${ARCHIVE_OPTIONS}" "${ARCHIVE_FILENAME}" --format=zip
                WORKING_DIRECTORY "${SOURCE_DIR_EXPECTED}"
            )
        elseif(NOT FETCHCONTENT_QUIET)
            message(STATUS "Archive found and already unpacked.")
        endif()

        set(FETCHCONTENT_SOURCE_DIR_${PARSED_ARGS_NAME_UPPER}
            "${SOURCE_DIR_EXPECTED}" CACHE PATH
            "When not empty, overrides where to find pre-populated content for ${PARSED_ARGS_NAME}"
        )
        FetchContent_Declare(${ARGN})
        FetchContent_MakeAvailable(${PARSED_ARGS_NAME})
    else()
        # When FetchContent is run with FETCHCONTENT_SOURCE_DIR_<uppercaseName> set, it seems to ignore checking if the new package version is different from the existing one.
        if(NOT FETCHCONTENTARCHIVE_FORCE_DOWNLOAD AND FETCHCONTENT_SOURCE_DIR_${PARSED_ARGS_NAME_UPPER})
            message(FATAL_ERROR "FetchContentArchive() for ${PARSED_ARGS_NAME} archive not found but source dir already set, did you change a package version? This should only be done with a clean build.")
        endif()

        if(NOT FETCHCONTENT_QUIET)
            message(STATUS "Archive not found, fetching...")
        endif()
        FetchContent_Declare(${ARGN})
        FetchContent_MakeAvailable(${PARSED_ARGS_NAME})

        # Usually FetchContent_MakeAvailable() sets ${PARSED_ARGS_NAME}_SOURCE_DIR to point to the source dir, but for some packages this seems to fail. We can get this variable another way using FetchContent_GetProperties().
        FetchContent_GetProperties(${PARSED_ARGS_NAME}
            SOURCE_DIR SOURCE_DIR_ACTUAL
        )

        if(NOT FETCHCONTENTARCHIVE_FORCE_DOWNLOAD)
            # Could use cmake_path() for this comparison (introduced in CMake 3.20).
            if(NOT "${SOURCE_DIR_ACTUAL}" STREQUAL "${SOURCE_DIR_EXPECTED}")
                message(FATAL_ERROR
                    "FetchContentArchive() for ${PARSED_ARGS_NAME} source dir paths differ:\n"
                    "${SOURCE_DIR_ACTUAL}\n"
                    "${SOURCE_DIR_EXPECTED}"
                )
            endif()

            if(NOT FETCHCONTENT_QUIET)
                message(STATUS "Creating archive in: ${ARCHIVE_FILENAME}")
            endif()

            # CMake 3.18 adds a file(ARCHIVE_CREATE ...) option for making a file archive, but there's not yet an option to set the working directory.
            # This method with execute_process can be used instead.
            file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/extern")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar "c${ARCHIVE_OPTIONS}" "${ARCHIVE_FILENAME}" --format=zip "."
                WORKING_DIRECTORY "${SOURCE_DIR_ACTUAL}"
            )
        endif()
    endif()

endfunction()
