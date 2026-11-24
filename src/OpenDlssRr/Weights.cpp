#include "OpenDlssRr/Weights.h"

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

//! A library that open() reads: its file name and the SHA-256 of its bytes.
struct Library {
    std::string_view name;
    std::string_view sha256;
};

//! The offset of a constant in each library of `libraries`.
struct ConstantEntry {
    std::string_view name;
    std::array<std::uint64_t, 2> offsets;
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

//! The folders of the NVIDIA driver (Linux), then the OpenDLSS-RR folder of the user data.
std::vector<std::filesystem::path> searchFolders() {
    std::vector<std::filesystem::path> folders;
#ifdef _WIN32
    if (const std::filesystem::path data = environment("LOCALAPPDATA"); !data.empty())
        folders.push_back(data / "OpenDLSS-RR");
#else
    for (const char* folder : {"/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/lib64", "/lib/x86_64-linux-gnu"})
        folders.emplace_back(folder);
    if (const std::filesystem::path data = environment("XDG_DATA_HOME"); !data.empty())
        folders.push_back(data / "OpenDLSS-RR");
    else if (const std::filesystem::path home = environment("HOME"); !home.empty())
        folders.push_back(home / ".local/share/OpenDLSS-RR");
#endif
    return folders;
}

std::expected<std::span<const std::byte>, Error> range(const std::span<const std::byte> file, const std::uint64_t offset, const std::uint64_t size) {
    if (offset > file.size() || size > file.size() - offset)
        return fail(std::format("the range {:#x} + {} is outside the library", offset, size));
    return file.subspan(offset, size);
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

std::expected<std::filesystem::path, Error> Weights::find(const std::filesystem::path& library) {
    if (!library.empty())
        return library;
    if (std::filesystem::path path = environment("OPENDLSS_RR_LIBRARY"); !path.empty())
        return path;
    std::error_code error;
    for (const std::filesystem::path& folder : searchFolders())
        for (const Library& known : libraries)
            if (std::filesystem::is_regular_file(folder / known.name, error))
                return folder / known.name;
    return fail("no DLSS Ray Reconstruction library: download " + std::string(downloadUrl)
                + " and set its path, set OPENDLSS_RR_LIBRARY, or copy it to the OpenDLSS-RR folder of the user data");
}

std::expected<void, Error> Weights::open(const std::filesystem::path& library) {
    const auto file = readFile(library);
    if (!file)
        return std::unexpected(file.error());
    const std::span<const std::byte> bytes = *file;
    const std::string digest = sha256(bytes);
    std::size_t version = 0;
    while (version < std::size(libraries) && libraries[version].sha256 != digest)
        ++version;
    if (version == std::size(libraries))
        return fail(std::format("{} has SHA-256 {}, which is not that of a supported library", library.string(), digest));

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
        const auto value = range(bytes, entry.offsets[version], single ? sizeof(float) : sizeof(std::uint16_t));
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
