#pragma once

#include "OpenDlssRr/ShaderCode.h"

namespace OpenDlssRr {

//! The shaders that the target OpenDlssRr::Shaders compiles at build time (OPENDLSS_RR_EMBED_SHADERS).
[[nodiscard]] ShaderCode getEmbeddedShaders();

}
