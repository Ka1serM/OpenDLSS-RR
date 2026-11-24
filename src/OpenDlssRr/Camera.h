#pragma once

#include <array>
#include <cstdint>

namespace OpenDlssRr {

//! The matrices of one evaluation, row major with row vectors (v * M).
struct CameraMatrices {
    std::array<float, 16> worldToView{};
    std::array<float, 16> viewToClip{};
};

//! The words of the camera block: six 4x4 matrices, row major.
inline constexpr std::uint32_t cameraWordCount = 96;

//! The camera block of the features pass. All operations are float32 in a fixed order: the compiler must not contract them.
[[nodiscard]] std::array<std::uint32_t, cameraWordCount> getCameraWords(const CameraMatrices& current, const CameraMatrices& previous,
                                                                        std::array<float, 2> jitter, std::array<std::uint32_t, 2> renderSize);

}
