#include "OpenDlssRr/EmbeddedShaders.h"

namespace OpenDlssRr {

namespace Embedded {

extern const std::span<const std::byte> Block;
extern const std::span<const std::byte> Mlp;
extern const std::span<const std::byte> PooledHistory;
extern const std::span<const std::byte> HistogramBasic;
extern const std::span<const std::byte> Downsample;
extern const std::span<const std::byte> Post;

}

ShaderCode getEmbeddedShaders() {
    return {
        .block = Embedded::Block,
        .stem = Embedded::Block,
        .mlp = Embedded::Mlp,
        .pooledHistory = Embedded::PooledHistory,
        .histogramBasic = Embedded::HistogramBasic,
        .downsample = Embedded::Downsample,
        .post = Embedded::Post,
    };
}

}
