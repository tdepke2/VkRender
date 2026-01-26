# Import Catch2 for unit testing.
message(STATUS "Fetching sources for Catch2...")
fetch_archive_declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.12.0
)
fetch_archive_make_available(Catch2)
message("")

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(vkr_src_test
    ComponentArray.test.cpp
)
target_link_libraries(vkr_src_test PRIVATE
    vkr_src
    Catch2::Catch2WithMain
)
vkr_add_cxx_properties(vkr_src_test)

catch_discover_tests(vkr_src_test)
