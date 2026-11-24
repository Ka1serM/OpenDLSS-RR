# Weights

The weights are not in this repository. The user supplies a DLSS Ray Reconstruction library, and `Weights::open()` reads the weights from
it. Nothing is written to disk.

## Libraries

| file | source |
| --- | --- |
| `libnvidia-ngx-dlssd.so.310.9.16` | the DLSS SDK: [direct download](https://github.com/NVIDIA/DLSS/raw/v310.9.16/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16) |
| `libnvidia-ngx-builtin-dlssd.so.310.9.0` | the NVIDIA Linux driver (for example in `/usr/lib64` or `/usr/lib/x86_64-linux-gnu`) |

The library is only a data file for OpenDLSS-RR, so a Windows host can read the Linux file too. `open()` refuses a file with a different
SHA-256.

`Weights::find()` (public: `findLibrary()`) takes the first of:

1. `Settings::library`, if it is a file;
2. the path in the environment variable `OPENDLSS_RR_LIBRARY`, if `Settings::library` is empty;
3. a file in the folder `Settings::library`, or if it is empty in `getLibraryFolder()`: `$XDG_DATA_HOME/OpenDLSS-RR` or
   `~/.local/share/OpenDLSS-RR` (Linux), `%LOCALAPPDATA%\OpenDLSS-RR` (Windows). The name does not matter: a file with the size of a
   supported library is taken;
4. the built-in library of the NVIDIA driver in the folders of the driver (Linux only).

When it finds nothing, it creates `getLibraryFolder()` and the error gives the download link and the folder.

## Weights.inc

`src/OpenDlssRr/Weights.inc` describes where the data is in each library. It holds offsets and shapes only, no weight values.

| entry | content |
| --- | --- |
| `libraries` | the file name and SHA-256 of each supported library |
| `lutOffset` | a 64K-entry FP16 -> FP16 table that rounds each FP16 value to the nearest E4M3 value (3 mantissa bits, saturates at 448) |
| `pairScaleOffsets` | the two FP32 factors of the attention bias tables |
| `constantEntries` | the name, the offset in each library and the precision of each per-kernel scalar |
| `tensorEntries` | the name (`bNN.tKK`), offset, shape and rounding of each tensor; the offsets are the same in both libraries |

Each tensor is row-major FP16. A tensor with rounding `E4m3` goes through the table once, when the library is read. `bNN` is the weight
buffer of a network kernel (`docs/network.md`), `tKK` the tensor in it.

## Unpacking

`Weights::open()` reads the library once: it checks the SHA-256, reads each tensor, rounds the `E4m3` tensors and reads the constants and
the two `pairScale` factors.

`Network::open()` then makes the attention bias table of each head, `(M + Mᵀ) · a · b` with M the 64x64 tensor of the head and a, b the
`pairScale` factors. It puts the matrices in tiles of 16x16 halves and the vectors in FP32. The first `record()` uploads them.

## Constants

Each network kernel has some scalars that are not in the tensors. `Weights::constants` holds them as `<kernel>.<name>`.

| name | meaning |
| --- | --- |
| gain | colour head gain (encoders: `colour · gain · sigmoid(head)`) |
| attention | residual scale of the attention block |
| mlp | residual scale of the MLP block |
| skip | scale of the encoder skip in the decoders |
| dec0_1.logCorrelation | `tanh(exp(logCorrelation) · x)` of the shape head |
| post.logBlend | the history blend factor of post, `exp(logBlend)` |
| enc0.motionDifferenceGain | gain of the motion change feature (FP32) |
| enc0.motionDifferenceScale | scale of the motion change feature, `scale · ln(1 + gain · |Δ|)` (FP32) |
| enc0.historyScale | scale of the history features |
| enc0.feedbackScale | scale of the feedback features (image 13) |
