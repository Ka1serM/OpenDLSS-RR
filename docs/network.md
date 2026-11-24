# Network

The network of DLSS Ray Reconstruction (preset F, library version 310.9) as plain layers. `src/OpenDlssRr/Network.cpp` and the Slang
passes in `src/OpenDlssRr/Shaders` implement this file. The weights come from the library of the user (`docs/weights.md`).

## Levels

A U-Net of windowed transformer blocks on 6 levels. L0 is the render size rounded up to a multiple of 32, so all levels halve exactly
(480x270 -> 480x288). Level l is 2^l times smaller.

| level | channels | heads | encoder | decoder |
| --- | --- | --- | --- | --- |
| L0 | 32 | 2 | enc0 (b06, b08) | dec0_0 (b36), dec0_1 (b38) |
| L1 | 64 | 2 | enc1 (b10) | dec1_0 (b32), dec1_1 (b34) |
| L2 | 96 | 4 | enc2 (b12) | dec2_0 (b28), dec2_1 (b30) |
| L3 | 128 | 4 | enc3 (b14) | dec3_0 (b24), dec3_1 (b26) |
| L4 | 160 | 8 | enc4 (b16) | dec4_0 (b20), dec4_1 (b22) |
| L5 | 160 | 8 | enc5 (b18) | |

`bNN` is the weight buffer of the kernel; `bNN.tKK` is its tensor KK. The kernels run in the order enc0 ... enc5,
dec4_0, dec4_1 ... dec0_0, dec0_1.

| kernel | does |
| --- | --- |
| enc0 | features, stem, block (b08.t00 ... t07), colour gate (b08.t08, t09) |
| enc1 | down L0 -> L1 (t00, t01), block (t02 ... t09), down L1 -> L2 (t10, t11), colour gate (t12, t13) |
| enc2 ... enc4 | block (t00 ... t07), down to the next level (t08, t09), colour gate (t10, t11) |
| enc5 | block (t00 ... t07), colour gate (t08, t09) |
| decL_0 | up of the coarser output (t00, t01) + skip scale · encoder output, block (t02 ... t09) |
| decL_1 (L = 1 ... 4) | block (t00 ... t07), colour blend (t08, t09) |
| dec0_1 | block (t00 ... t07), output heads (no weights) |

The constants of each kernel (colour gain, attention and MLP residual scales, skip scale) are not in the weight buffers.
`src/OpenDlssRr/Weights.inc` gives their offsets (`docs/weights.md`).

## Block

The same block on all levels, per 8x8 window of 64 tokens. There is no normalization.

1. **Projection**, no bias: t00 (2 · heads · 32, C). The first half are the queries, the second half the values, 32 channels per head.
   There are no separate keys: the keys are the queries, so the scores are symmetric.
2. **Scores**: `s = q_i · q_j + B[i][j]`. B is the bias table of the head, `(M + Mᵀ) · pairScale` from t01 (heads, 64, 64). The reader makes
   it once, when it loads the weights.
3. **Weights**: `x = clamp(s · 0.0972232 / √32, ±0.556152)`, `e = 32 (0.375 + x (0.927246 − x²))`, `w = 2^floor(e) · (1 + frac(e))`, then
   each row is divided by its sum. This is a soft-clamped,
   piecewise linear exp2, not a softmax: a plain softmax gives about 18 % error.
4. **Attention output**: `y = x + attention · (concat_heads(Σ w v) @ t02ᵀ + t03)`.
5. **MLP**: `z = y + mlp · (gelu(y @ t04ᵀ + t05) @ t06ᵀ + t07)`, C -> 4C -> C.
   `gelu(h) = h · (0.5 + t (0.412162 − 0.081081 |t|))`, `t = clamp(h, ±2)`: 0 below −2, h above 2. The erf or tanh GELU is worse.

Windows start at `8i − shift`. A window that leaves the level reads it mirrored without a repeat of the edge pixel
(`reflect(i) = min(|i|, 2n − 2 − |i|)`). There is no attention mask. Only the tokens inside the level are written.

| kernels | even frame | odd frame |
| --- | --- | --- |
| enc0 ... enc4 | (2, 6) | (6, 2) |
| enc5, decL_0 | (4, 0) | (0, 4) |
| decL_1 | (0, 4) | (4, 0) |

## Down and up

- **Down**: a 2x2 convolution with stride 2, weight (C_out, C_in, 4), position `2 dy + dx`. enc1 keeps both of its weights flat,
  (C_out, 4 C_in), in the same order.
- **Up**: a 2x2 transposed convolution, weight (C_out, 4, C_in), bias (C_out, 4), position `2 (y % 2) + x % 2`. The coarse pixel `x / 2`
  mirrors at the coarse size; the position still comes from `x % 2`.
- **Skip**: decL_0 adds `skip · encoded[L]` to the up-sampled input. Nothing is concatenated.

## Colour heads

Each level has a colour (FP16 RGBA) next to the activations.

- **Gate** (encoders): `colour.rgb = min(c · gain · sigmoid(z @ Wᵀ + b), 65504)`, alpha 0. c is the exposed input colour (enc0) or the 2x2
  mean of the gated colour of the finer level.
- **Blend** (dec4_1 ... dec1_1): 4 outputs h. `alpha = sigmoid(h3)`,
  `colour = min(E (1 − alpha) + [up(U).rgb · gain · sigmoid(h0..2), 0] · alpha, 65504)`. E is the gated colour of the level, U the colour of
  the coarser level (enc5 for dec4_1), up() a bilinear 2x: pixel X reads `(X − 1) / 2` (0 for X = 0) and the next pixel, with the weight 0.25
  (X odd) or 0.75 (X even) on the second.
- **Output heads** (dec0_1), from the first 18 channels of z, no weights:

| channels | function | output |
| --- | --- | --- |
| 0 ... 2 | `g = gain · sigmoid(x)` | colour gain |
| 3 | `a = sigmoid(x)` | blend alpha; the colour is the blend above with g and the dec1 colour |
| 4 ... 6 | `sigmoid(x)`, alpha 0 | image 20 (guide, RGBA8 UNORM) |
| 7, 8, 9 | `sx = 8 sigmoid(clamp(x7, ±20))`, `sy = 8 sigmoid(clamp(x8, ±20))`, `r = tanh(0.307675 x9)` -> `(sx², sy², r sx sy, 0)` | image 22 (shape: a 2x2 covariance, RGBA16F) |
| 10 ... 13 | softmax over the 4 | image 21 (candidate weights, RGBA8 UNORM) |
| 14 ... 17 | `tanh(x)` | image 13 (feedback, RGBA8 SNORM), read by enc0 of the next frame |

Post reads the colour and the images 20, 21 and 22. 0.307675 is `exp(dec0_1.logCorrelation)`.

## enc0 features

43 features per pixel of L0 (K padded to 48). Positions outside the render size mirror into the render image.
"exposed" is colour · exposure.x; `logC(v) = 0.36 log10(max(1 + 100 v, 1e-4))`.

Motion is `(mv + jitter difference) · 3`, in history pixels. Nearest and farthest motion: the motion of the 3x3 neighbour with the smallest
and largest |depth| (centre first, then row by row, strict comparisons). History position `h = motion + (pixel + 0.5 − jitter) · 3`, valid when
it is inside the post size and the frame is not a reset.

| channels | feature |
| --- | --- |
| 0 ... 2 | logC(exposed colour); NaN and Inf -> 0, min 65504 before the log |
| 3 ... 5 | history at the nearest motion: the mip (log code / 2.45) bilinear at h, decoded, / exposure.y · exposure.x, logC, · historyScale, min 1024; the colour when invalid |
| 6 ... 8 | the same at the centre motion |
| 9 ... 11 | the same from the pooled mip |
| 12 ... 14 | the same at the farthest motion |
| 15 ... 17 | diffuse albedo, min 2 |
| 18 ... 20 | specular albedo, min 2 |
| 21 ... 24 | normal.xyz, roughness; min 2; 0 where both albedos sum to 0 |
| 25 | motion change at the nearest motion: `motionDifferenceScale · ln(1 + |motionDifferenceGain · clip((m − previous) / 3, ±1000)|)`; previous = the last motion image bilinear at the reprojected pixel; 0 on a reset |
| 26 ... 29 | feedback (image 13) at h + 3 · previous jitter, · feedbackScale; 0 when invalid or on a reset |
| 30 ... 33 | 0 (optional inputs that the runtime does not take) |
| 34 | exposure.w |
| 35 ... 37 | history at the specular motion |
| 38 | motion change at the specular motion |
| 39 ... 42 | feedback at the specular motion |

Specular motion: the virtual image of the reflection. Hit distance (≤ 0 -> 1e7), P = pixelToWorld of the pixel centre, V = normalize(P − camera),
N = the normal, hit = distance · reflect(V, N). The virtual point is `P + V sqrt(|hit|² − (N·hit)² + max(|N·hit|, 1e-5)²)`, projected with
worldToPreviousPixel, minus the pixel centre, times 3. When it is invalid: the centre motion.

The features pass also writes the motion image (nearest, specular), the farthest and the centre motion (FP16).

**Dither.** Each feature is rounded to FP16 (round to nearest even), then to 3 mantissa bits with a blue-noise threshold:
`noise = b06.t00[frame · 256 + (y & 15) · 16 + (x & 15)] >> 9` (frame = frame index % 8; x, y the mirrored position),
`value = trunc3(x + noise · ulp16(x))`. The exponent stays FP16. The table is 8 frames of a 16x16 blue-noise tile.

**Stem**: `h = gelu(q @ b06.t01ᵀ + b06.t02)` (43 -> 64), `s = h @ b06.t03ᵀ + b06.t04` (64 -> 32), then the block.

## Passes

`Network.cpp` records the network as compute passes of `Block.slang` and `Mlp.slang`. All matrix products use KHR cooperative matrices
(16x16x16, FP16 inputs and accumulators), so the device must have `VK_KHR_cooperative_matrix` with that size at subgroup scope.

- **Block** (`Block.slang`): one workgroup of 4 subgroups per 8x8 window. It computes the prologue (stem, down or up convolution), the
  attention, the MLP, an optional colour head, and optionally the down convolution of the next encoder as an epilogue. The tokens of a
  window are in the order of their 2x2 parity, then of their 4x4 position, so that a row tile of 16 tokens is one parity: the down and up
  convolutions are then products of whole row tiles. The weights are in tiles of 16x16 halves (`tileAt` in `Network.slang`).
- **L4 and L5**: the windows are few, so a block is split into three dispatches: the block input (column parts of the prologue), the
  attention of one head per workgroup, and `Mlp.slang` (projection, MLP and colour head, 16 pixels per workgroup of 8 subgroups).
- **Attention**: the weights P of a row tile go through shared memory once; P 1 and P V accumulate in the same pass, and P V is divided by
  P 1 at the end. P is scaled by 2^-8 so that P V stays in the FP16 range.

## Precision

The Slang passes do the maths in FP32 and store the activations as FP16. `f32tof16` does not round to nearest on all drivers, so the
features use an integer round-to-nearest-even conversion (`halfBits`) for the dither and the motion bits.

## Accuracy

Against DLSS Ray Reconstruction 310.9.16 on the same inputs, the output differs by about 0.5 to 0.9 % relative RMS,
for DLAA, 1.5x, 2x and 3x scales and odd sizes. llvmpipe and an RTX 3090 differ by 0.20 to 0.24 %.

## Open points

- Inputs that are not tested: non-linear and inverted depth, normals encoded in [0, 1], roughness in normals.a, the optional inputs of
  features 30 ... 33.
