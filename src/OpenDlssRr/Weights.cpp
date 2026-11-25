#include "OpenDlssRr/Weights.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <span>

#include "OpenDlssRr/Sha256.h"

namespace OpenDlssRr {

namespace {

using Vulkan::Error;
using Vulkan::fail;

enum class Precision { Half, Single };
enum class Rounding { None, E4m3 };

struct ConstantEntry {
    std::string_view name;
    std::uint64_t offset;
    Precision precision;
};

struct TensorEntry {
    std::string_view name;
    std::uint64_t offset;
    std::array<std::uint32_t, 3> shape;
    Rounding rounding;
};

#include "OpenDlssRr/Weights.inc"

std::expected<std::vector<std::byte>, Error> readFile(const std::filesystem::path& path) {
    const std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(path.string().c_str(), "rb"), std::fclose);
    if (!file)
        return fail(std::format("cannot open {}", path.string()));
    std::vector<std::byte> bytes;
    std::byte buffer[1 << 16];
    for (std::size_t count; (count = std::fread(buffer, 1, sizeof(buffer), file.get())) > 0;)
        bytes.insert(bytes.end(), buffer, buffer + count);
    if (std::ferror(file.get()))
        return fail(std::format("cannot read {}", path.string()));
    return bytes;
}

//! An environment variable, or empty.
std::filesystem::path environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::filesystem::path(value) : std::filesystem::path();
}

//! A file in the folder with the size of the library, whatever its name.
std::filesystem::path findInFolder(const std::filesystem::path& folder) {
    std::error_code error;
    for (std::filesystem::directory_iterator entry(folder, error), end; !error && entry != end; entry.increment(error))
        if (entry->is_regular_file(error))
            if (const std::uintmax_t size = entry->file_size(error); size == librarySize || size == libraryRange.size)
                return entry->path();
    return {};
}

//! The bytes of the whole library or of its weights range, and the library offset of their first byte.
struct Source {
    std::span<const std::byte> bytes;
    std::uint64_t base = 0;
};

//! Bytes at a library offset.
std::expected<std::span<const std::byte>, Error> range(const Source& source, const std::uint64_t offset, const std::uint64_t size) {
    if (offset < source.base || offset - source.base > source.bytes.size() || size > source.bytes.size() - (offset - source.base))
        return fail(std::format("the range {:#x} + {} is outside the library", offset, size));
    return source.bytes.subspan(offset - source.base, size);
}

std::vector<std::uint16_t> halves(const std::span<const std::byte> bytes) {
    std::vector<std::uint16_t> values(bytes.size() / 2);
    std::memcpy(values.data(), bytes.data(), values.size() * 2);
    return values;
}

}

float halfToFloat(const std::uint16_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1f;
    const std::uint32_t mantissa = half & 0x3ff;
    if (exponent == 0)
        return std::bit_cast<float>(sign) + (sign ? -1.0f : 1.0f) * static_cast<float>(mantissa) * 0x1p-24f;
    if (exponent == 31)
        return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 13));
    return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

std::filesystem::path Weights::folder() {
#ifdef _WIN32
    if (const std::filesystem::path data = environment("LOCALAPPDATA"); !data.empty())
        return data / "OpenDLSS-RR";
#else
    if (const std::filesystem::path data = environment("XDG_DATA_HOME"); !data.empty())
        return data / "OpenDLSS-RR";
    if (const std::filesystem::path home = environment("HOME"); !home.empty())
        return home / ".local" / "share" / "OpenDLSS-RR";
#endif
    return {};
}

std::expected<std::filesystem::path, Error> Weights::find(const std::filesystem::path& library) {
    if (!library.empty())
        return library;
    if (std::filesystem::path path = environment("OPENDLSS_RR_LIBRARY"); !path.empty())
        return path;
    const std::filesystem::path user = folder();
    if (user.empty())
        return fail("no DLSS Ray Reconstruction library: download " + std::string(downloadUrl) + " and set OPENDLSS_RR_LIBRARY to its path");
    if (std::filesystem::path path = findInFolder(user); !path.empty())
        return path;
    std::error_code error;
    std::filesystem::create_directories(user, error);
    return fail("no DLSS Ray Reconstruction library: download " + std::string(downloadUrl) + " into " + user.string());
}

std::expected<void, Error> Weights::open(const std::filesystem::path& library) {
    const auto file = readFile(library);
    if (!file)
        return std::unexpected(file.error());
    if (auto result = open(std::span<const std::byte>(*file)); !result)
        return fail(library.string() + ": " + result.error().message);
    return {};
}

std::expected<void, Error> Weights::check(const std::span<const std::byte> bytes) {
    if (bytes.size() != librarySize && bytes.size() != libraryRange.size)
        return fail(std::format("not the supported DLSS Ray Reconstruction library ({} bytes, not {} or the {} bytes at {} of it): download {}", bytes.size(),
                                librarySize, libraryRange.size, libraryRange.offset, downloadUrl));
    const std::string digest = sha256(bytes);
    if (digest != (bytes.size() == librarySize ? librarySha256 : libraryRange.sha256))
        return fail(std::format("not the supported DLSS Ray Reconstruction library (SHA-256 {}): download {}", digest, downloadUrl));
    return {};
}

std::expected<void, Error> Weights::open(const std::span<const std::byte> data) {
    if (auto checked = check(data); !checked)
        return checked;
    const Source bytes{data, data.size() == librarySize ? 0 : libraryRange.offset};

    const auto lutBytes = range(bytes, lutOffset, 65536 * 2);
    if (!lutBytes)
        return std::unexpected(lutBytes.error());
    const std::vector<std::uint16_t> lut = halves(*lutBytes);
    for (std::size_t index = 0; index < pairScale.size(); ++index) {
        const auto factor = range(bytes, pairScaleOffsets[index], sizeof(float));
        if (!factor)
            return std::unexpected(factor.error());
        std::memcpy(&pairScale[index], factor->data(), sizeof(float));
    }

    constants.clear();
    for (const ConstantEntry& entry : constantEntries) {
        const bool single = entry.precision == Precision::Single;
        const auto value = range(bytes, entry.offset, single ? sizeof(float) : sizeof(std::uint16_t));
        if (!value)
            return std::unexpected(value.error());
        float number = 0.0f;
        if (single)
            std::memcpy(&number, value->data(), sizeof(number));
        else
            number = halfToFloat(halves(*value)[0]);
        constants.emplace(entry.name, number);
    }

    tensors.clear();
    for (const TensorEntry& entry : tensorEntries) {
        WeightTensor tensor;
        std::uint64_t count = 1;
        for (const std::uint32_t extent : entry.shape) {
            if (extent == 0)
                break;
            tensor.shape.push_back(extent);
            count *= extent;
        }
        const auto source = range(bytes, entry.offset, count * 2);
        if (!source)
            return std::unexpected(source.error());
        tensor.values = halves(*source);
        if (entry.rounding == Rounding::E4m3)
            for (std::uint16_t& value : tensor.values)
                value = lut[value];
        tensors.emplace(entry.name, std::move(tensor));
    }
    return {};
}

}
