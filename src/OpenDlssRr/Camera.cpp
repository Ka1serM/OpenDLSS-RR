#include "OpenDlssRr/Camera.h"

#include <bit>
#include <optional>

namespace OpenDlssRr {

namespace {

using Matrix = std::array<float, 16>;

float at(const Matrix& matrix, const int row, const int column) {
    return matrix[static_cast<std::size_t>(row * 4 + column)];
}

//! Each element is ((a0 * b0 + a1 * b1) + a2 * b2) + a3 * b3.
Matrix multiply(const Matrix& left, const Matrix& right) {
    Matrix result{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column) {
            float sum = at(left, row, 0) * at(right, 0, column);
            for (int index = 1; index < 4; ++index)
                sum = sum + at(left, row, index) * at(right, index, column);
            result[static_cast<std::size_t>(row * 4 + column)] = sum;
        }
    return result;
}

//! The adjugate divided by the determinant. The determinant is expanded along each row and the one of the last row is kept.
std::optional<Matrix> inverse(const Matrix& matrix) {
    Matrix adjugate{};
    float determinant = 0.0f;
    for (int row = 0; row < 4; ++row) {
        std::array<int, 3> rows{};
        for (int other = 0, count = 0; other < 4; ++other)
            if (other != row)
                rows[static_cast<std::size_t>(count++)] = other;
        determinant = 0.0f;
        for (int column = 0; column < 4; ++column) {
            std::array<int, 3> columns{};
            for (int other = 0, count = 0; other < 4; ++other)
                if (other != column)
                    columns[static_cast<std::size_t>(count++)] = other;
            const auto m = [&](const int r, const int c) { return at(matrix, rows[static_cast<std::size_t>(r)], columns[static_cast<std::size_t>(c)]); };
            float first = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
            first = first * m(0, 0);
            float second = m(2, 2) * m(1, 0) - m(1, 2) * m(2, 0);
            second = second * m(0, 1);
            first = first - second;
            float third = m(2, 1) * m(1, 0) - m(1, 1) * m(2, 0);
            third = third * m(0, 2);
            float cofactor = third + first;
            cofactor = cofactor * (((row ^ column) & 1) ? 1.0f : -1.0f);
            adjugate[static_cast<std::size_t>(column * 4 + row)] = cofactor;
            determinant = determinant + cofactor * at(matrix, row, column);
        }
        if (determinant == 0.0f)
            return std::nullopt;
    }
    const float reciprocal = 1.0f / determinant;
    for (float& value : adjugate)
        value = value * reciprocal;
    return adjugate;
}

//! A singular matrix gives zeros.
Matrix inverseOrZero(const Matrix& matrix) {
    return inverse(matrix).value_or(Matrix{});
}

}

std::array<std::uint32_t, cameraWordCount> getCameraWords(const CameraMatrices& current, const CameraMatrices& previous, const std::array<float, 2> jitter,
                                                          const std::array<std::uint32_t, 2> renderSize) {
    const float width = static_cast<float>(renderSize[0]);
    const float height = static_cast<float>(renderSize[1]);
    Matrix viewport{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    viewport[0] = viewport[12] = width * 0.5f;
    viewport[5] = height * -0.5f;
    viewport[13] = height * 0.5f;

    const Matrix previousViewToWorld = inverseOrZero(previous.worldToView);
    const Matrix viewToWorld = inverseOrZero(current.worldToView);
    //! The previous view to the world, relative to the current camera position.
    Matrix relative = previousViewToWorld;
    for (int index = 0; index < 3; ++index)
        relative[static_cast<std::size_t>(12 + index)] =
            previousViewToWorld[static_cast<std::size_t>(12 + index)] - viewToWorld[static_cast<std::size_t>(12 + index)];
    Matrix rotation = viewToWorld;
    rotation[12] = rotation[13] = rotation[14] = 0.0f;

    const Matrix worldToPreviousClip = multiply(inverseOrZero(relative), previous.viewToClip);
    const Matrix worldToClip = multiply(inverseOrZero(rotation), current.viewToClip);
    const Matrix clipToWorld = inverseOrZero(worldToClip);
    const Matrix pixelToWorld = inverseOrZero(multiply(worldToClip, viewport));
    const Matrix worldToPreviousPixel = multiply(multiply(worldToClip, multiply(clipToWorld, worldToPreviousClip)), viewport);
    Matrix jittered = current.viewToClip;
    jittered[8] = (jitter[0] + jitter[0]) / width + jittered[8];
    jittered[9] = -2.0f * jitter[1] / height + jittered[9];

    std::array<std::uint32_t, cameraWordCount> words{};
    std::size_t index = 0;
    for (const Matrix* matrix :
         std::array<const Matrix*, 6>{&worldToPreviousClip, &current.viewToClip, &jittered, &clipToWorld, &pixelToWorld, &worldToPreviousPixel})
        for (const float value : *matrix)
            words[index++] = std::bit_cast<std::uint32_t>(value);
    return words;
}

}
