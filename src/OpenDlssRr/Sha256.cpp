#include "OpenDlssRr/Sha256.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <format>

namespace OpenDlssRr {

namespace {

constexpr std::array<std::uint32_t, 64> roundConstants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74,
    0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d,
    0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e,
    0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void compress(std::array<std::uint32_t, 8>& state, const std::byte* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
        std::memcpy(&words[index], block + 4 * index, 4);
        words[index] = std::byteswap(words[index]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
        const std::uint32_t low = words[index - 15], high = words[index - 2];
        words[index] = words[index - 16] + (std::rotr(low, 7) ^ std::rotr(low, 18) ^ (low >> 3)) + words[index - 7]
                       + (std::rotr(high, 17) ^ std::rotr(high, 19) ^ (high >> 10));
    }
    auto [a, b, c, d, e, f, g, h] = state;
    for (std::size_t index = 0; index < 64; ++index) {
        const std::uint32_t first = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + roundConstants[index] + words[index];
        const std::uint32_t second = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    const std::array<std::uint32_t, 8> added{a, b, c, d, e, f, g, h};
    for (std::size_t index = 0; index < 8; ++index)
        state[index] += added[index];
}

}

std::string sha256(const std::span<const std::byte> bytes) {
    std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const std::size_t whole = bytes.size() / 64 * 64;
    for (std::size_t at = 0; at < whole; at += 64)
        compress(state, bytes.data() + at);
    std::array<std::byte, 128> tail{};
    const std::size_t rest = bytes.size() - whole;
    std::memcpy(tail.data(), bytes.data() + whole, rest);
    tail[rest] = std::byte{0x80};
    const std::size_t tailSize = rest < 56 ? 64 : 128;
    const std::uint64_t bits = std::byteswap(std::uint64_t{bytes.size()} * 8);
    std::memcpy(tail.data() + tailSize - 8, &bits, 8);
    for (std::size_t at = 0; at < tailSize; at += 64)
        compress(state, tail.data() + at);
    std::string digest;
    for (const std::uint32_t word : state)
        digest += std::format("{:08x}", word);
    return digest;
}

}
