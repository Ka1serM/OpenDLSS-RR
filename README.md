# OpenDLSS-RR

DLSS Ray Reconstruction as a plain Vulkan 1.3 library, with the network as readable Slang compute shaders. It has no vendor-specific
code: it runs on any GPU with `VK_KHR_cooperative_matrix` (NVIDIA, AMD and Intel).

The weights are not in this repository. The runtime reads them from a DLSS Ray Reconstruction library when it opens. To install it,
download [`libnvidia-ngx-dlssd.so.310.9.16`](https://github.com/NVIDIA/DLSS/raw/v310.9.16/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16) from the
[DLSS SDK](https://github.com/NVIDIA/DLSS) and put it, under any name, in:

| system | folder |
| --- | --- |
| Linux | `$XDG_DATA_HOME/OpenDLSS-RR`, by default `~/.local/share/OpenDLSS-RR` |
| Windows | `%LOCALAPPDATA%\OpenDLSS-RR` |

The same file works on Linux and Windows: the runtime only reads its data. A host can also give the path of the file, under any name,
in `Settings::library`, or its bytes in `Settings::libraryData`.

The runtime only needs 7.3 MB of the 48 MB file. A host, such as a browser, can download just that range (`libraryRange` in `OpenDlssRr/LibraryRange.h`) and use it
in place of the file anywhere (`docs/weights.md`).

## Use

```cmake
add_subdirectory(external/OpenDLSS-RR)
target_link_libraries(MyRenderer PRIVATE OpenDlssRr::OpenDlssRr OpenDlssRr::Shaders)
```

```cpp
#include <OpenDlssRr/EmbeddedShaders.h>
#include <OpenDlssRr/RayReconstruction.h>

// Before vkCreateDevice: enable getSupportedDeviceExtensions() and the features of checkDevice().
OpenDlssRr::RayReconstruction reconstruction;
auto opened = reconstruction.open({.instance = instance, .physicalDevice = physicalDevice, .device = device,
                                   .shaders = OpenDlssRr::getEmbeddedShaders(),
                                   .renderWidth = 960, .renderHeight = 540, .outputWidth = 1920, .outputHeight = 1080});
// Each frame, into a command buffer of the host:
auto recorded = reconstruction.record(commandBuffer, textures, {.jitter = jitter, .worldToView = view, .viewToClip = projection});
```

- `docs/runtime.md`: how to add the library to a renderer: device extensions and features, inputs and synchronization.
- `docs/network.md`: the network and its passes.
- `docs/weights.md`: which library the runtime reads, where it looks for it, and how it reads the weights.

## Build

CMake 3.29, a C++23 compiler (tested with Clang 22 and GCC 16), the Vulkan headers and `slangc` (on `PATH`, in `SLANGC` or in
`$VULKAN_SDK/bin`).

```
cmake -B build -G Ninja
cmake --build build
```

`git config core.hooksPath tools/githooks` turns on the pre-commit hook: clang-format, and `tools/check_comments.py` (only one-line `//!`
comments).

## Accuracy

Against DLSS Ray Reconstruction on the same inputs, the output differs by about 0.5 to 0.9 % relative RMS.

## License

MIT (`LICENSE`). NVIDIA and DLSS are trademarks of NVIDIA Corporation. This project is not affiliated with NVIDIA.
