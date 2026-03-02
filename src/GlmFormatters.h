#pragma once

#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <spdlog/fmt/fmt.h>

// Formatting GLM types to string is already available in the library, the output for some types is not the best though.
//#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/gtx/string_cast.hpp>

template<>
struct fmt::formatter<glm::vec3> {
    constexpr auto parse(format_parse_context& ctx) const -> format_parse_context::iterator {
        return ctx.end();
    }
    auto format(const glm::vec3& vec, format_context& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "vec3({:f}, {:f}, {:f})", vec[0], vec[1], vec[2]);
    }
};

template<>
struct fmt::formatter<glm::mat4> {
    constexpr auto parse(format_parse_context& ctx) const -> format_parse_context::iterator {
        return ctx.end();
    }
    auto format(const glm::mat4& mat, format_context& ctx) const -> format_context::iterator {
        // GLM uses column-major ordering and we keep that format here.
        return format_to(ctx.out(), "mat4x4[\n{:f}, {:f}, {:f}, {:f}\n{:f}, {:f}, {:f}, {:f}\n{:f}, {:f}, {:f}, {:f}\n{:f}, {:f}, {:f}, {:f}]",
            mat[0][0], mat[0][1], mat[0][2], mat[0][3],
            mat[1][0], mat[1][1], mat[1][2], mat[1][3],
            mat[2][0], mat[2][1], mat[2][2], mat[2][3],
            mat[3][0], mat[3][1], mat[3][2], mat[3][3]
        );
    }
};

template<>
struct fmt::formatter<glm::quat> {
    constexpr auto parse(format_parse_context& ctx) const -> format_parse_context::iterator {
        return ctx.end();
    }
    auto format(const glm::quat& quat, format_context& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "quat({:f}, {{{:f}, {:f}, {:f}}})", quat.w, quat.x, quat.y, quat.z);
    }
};
