#pragma once

#include <cstddef>
#include <span>

namespace OpenDlssRr {

//! The SPIR-V of each Slang file in src/OpenDlssRr/Shaders, compiled by the host with its entry point name (slangc -fvk-use-entrypoint-name).
struct ShaderCode {
    std::span<const std::byte> block;
    //! The entry point stem of Block.slang.
    std::span<const std::byte> stem;
    std::span<const std::byte> mlp;
    std::span<const std::byte> pooledHistory;
    std::span<const std::byte> histogramBasic;
    std::span<const std::byte> downsample;
    std::span<const std::byte> post;
};

}
