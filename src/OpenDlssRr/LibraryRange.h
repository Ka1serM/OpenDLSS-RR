#pragma once

#include <cstdint>
#include <string_view>

//! Header-only and without Vulkan, so that a host can download the range in a build without the runtime, such as a browser build.

namespace OpenDlssRr {

//! The bytes of the DLSS Ray Reconstruction library that hold all weights: `size` bytes at `offset` of the file at `url`.
struct LibraryRange {
    //! A link that a browser can fetch: it sends CORS headers and answers HTTP Range requests (the github.com link redirects without them).
    std::string_view url;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::string_view sha256;
};

//! The range of libnvidia-ngx-dlssd.so.310.9.16 of the DLSS SDK; HTTP header `Range: bytes=offset-(offset + size - 1)`.
inline constexpr LibraryRange libraryRange{
    .url = "https://raw.githubusercontent.com/NVIDIA/DLSS/v310.9.16/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16",
    .offset = 193196,
    .size = 7340340,
    .sha256 = "63a801eff9b703b3b4344617b8a2a956ef8aa877cfbb5940764d9b39078ec1f7",
};

}
