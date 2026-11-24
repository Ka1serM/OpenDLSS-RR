# Runtime

OpenDLSS-RR is a static library of plain Vulkan 1.3: `OpenDlssRr::OpenDlssRr`, public header `src/OpenDlssRr/RayReconstruction.h`.
It has no vendor-specific code: any GPU with the features below runs it, including llvmpipe.

## Host integration

1. Add the repository with `add_subdirectory` and link `OpenDlssRr::OpenDlssRr`. The library needs only the Vulkan headers
   (`Vulkan::Headers`). It does not link the loader.
   - With `OPENDLSS_RR_EMBED_SHADERS` (on by default), link `OpenDlssRr::Shaders` too and pass `getEmbeddedShaders()`
     (`OpenDlssRr/EmbeddedShaders.h`) as `Settings::shaders`. The build compiles the shaders with `slangc` (on `PATH`, in `SLANGC` or in
     `$VULKAN_SDK/bin`).
   - A host that compiles Slang itself turns the option off and compiles each file of `src/OpenDlssRr/Shaders` to SPIR-V with its entry
     point names (`slangc -target spirv -profile spirv_1_5 -fvk-use-entrypoint-name`).
2. Before you create the device, enable the extensions of `getSupportedDeviceExtensions()` (`VK_KHR_push_descriptor` and
   `VK_KHR_cooperative_matrix`) and the features that `checkDevice()` checks: `shaderInt64`, `shaderInt16`,
   `shaderStorageImageExtendedFormats`, `shaderFloat16`, `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`,
   `shaderBufferInt64Atomics`, `vulkanMemoryModel`, `vulkanMemoryModelDeviceScope`, `cooperativeMatrix`, and `subgroupSizeControl` when the
   default subgroup size is not 32. `checkDevice()` also checks for 16x16x16 FP16 cooperative matrices with FP16 and FP32 accumulators.
   Both take the instance of the host and, optionally, its `vkGetInstanceProcAddr` (else the one of `libvulkan.so.1` or `vulkan-1.dll`).
3. Call `RayReconstruction::open()` with `Settings`: the instance, physical device, device, an optional pipeline cache, the shaders, an
   optional file or folder of the DLSS Ray Reconstruction library of the user (`library`, `docs/weights.md`), and the render and output sizes. Each
   pair of sizes needs its own object.
4. For each frame, call `record()` with a command buffer of the host. The library never submits.
   - All images must be in `VK_IMAGE_LAYOUT_GENERAL`. The output is `R16G16B16A16_SFLOAT` with storage use.
   - The jitter is in render pixels.
   - `worldToView` and `viewToClip` are row major, row vectors (`v * M`). The runtime keeps the matrices of the previous frame.
   - Make sure that the inputs are visible to compute shaders before the commands. Make sure that later reads of the output wait for compute shader writes.
5. Destroy the object only after the GPU is done with all its commands.

[NoorRay](https://github.com/Ka1serM/NoorRay) (`src/NoorRay/Render/Upscale/DlssRrOpen.cpp`) is an example host. It gets the Vulkan handles through `NoorRHI::Interop`.

## The weights

The weights are not in this repository. Each user supplies a DLSS Ray Reconstruction library (`docs/weights.md`). `open()` finds it, checks
its SHA-256, reads the weight buffers and the kernel constants at the offsets of `Weights.inc`, unpacks them once and creates the pipelines.
The first `record()` uploads the weights. Nothing is written to disk.

## Frame

All sizes come from the render size and the output size:

| size | value |
| --- | --- |
| L0 (network) | render size rounded up to 32; level l is 2^l times smaller |
| post | 3 · ceil(output / 2) |
| mip | ceil(post / 2) |
| pooled mip | ceil(mip / 2) |
| post pitch | post rounded up to even |

One evaluation is these passes, with a full barrier after each:

1. First frame or reset: clear the exposure images to 1 (the first frame also clears the exposure sums). Else: `PooledHistory` (mip -> pooled mip).
2. `HistogramBasic`: the exposure of the colour.
3. The network (`docs/network.md`): features, stem, the blocks, down and up convolutions and colour heads of the 16 network kernels, and the
   output heads. It writes the colour and the images that post reads, the motion images and the feedback image.
4. `Post`: one launch over the whole post size. It filters the colour into the history and the mip.
5. `Downsample`: one launch, from the post size to the output.

Pairs of state images alternate with the frame parity (frame index % 2): the history, the motion image, the exposure images and the exposure
sums. The frame index word of the features (the dither frame) is the frame index % 8.

## Camera

The features use six 4x4 matrices made from the current and the previous `worldToView` and `viewToClip` (`Camera.h`): an adjugate inverse
and products that add from left to right, in float32.
On the first frame and on a reset, the previous matrices are the current ones.

## Reset

A reset after the first frame clears the exposure images in place of the pooled-history pass, and the network and post ignore the history.
The frame index keeps counting: a reset does not set it to zero.

## Limits

- A singular `worldToView`, or a singular product with the projection, gives zero matrices.
- Only these values are supported for the other parameters: a render subrect of the full render size, motion vector scale 1,
  pre-exposure and exposure scale 1.
- The render size must not be larger than the output size (the post tile holds a scale of 2/3 at most).
