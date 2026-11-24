#pragma once

#include <expected>
#include <format>
#include <string>

#include <vulkan/vulkan.h>

namespace OpenDlssRr::Vulkan {

struct Error {
    std::string message;
};

[[nodiscard]] inline std::unexpected<Error> fail(std::string message) {
    return std::unexpected(Error{std::move(message)});
}

[[nodiscard]] inline std::expected<void, Error> check(const VkResult result, const char* call) {
    if (result != VK_SUCCESS)
        return fail(std::format("{} failed: {}", call, static_cast<int>(result)));
    return {};
}

}
