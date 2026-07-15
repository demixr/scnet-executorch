# SCNet deployment for Demixr

Reproducible deployment exports of the official MIT-licensed
[SCNet](https://github.com/starrytong/SCNet) music-separation model. This is
the SCNet counterpart to
[demucs-executorch](https://github.com/demixr/demucs-executorch).

The repository does not fork SCNet or commit its checkpoint. Export scripts
load the official implementation and checkpoint, preserve the learned
weights, validate output shapes and waveform parity, and produce fixed 7.8-second spectral
cores for Demixr.

## Release artifacts

| Asset | Runtime | Platforms | Format |
| --- | --- | --- | --- |
| `scnet_coreml.mlpackage.zip` | Core ML GPU | macOS / iOS | native ML Program |
| `scnet_vulkan.pte` | ExecuTorch Vulkan | Android | `.pte` |
| `scnet_cpu.onnx` | ONNX Runtime CPU | all supported native platforms | `.onnx` |
| `models.json` | download metadata and SHA-256 checksums | Demixr | JSON |

Apps can fetch a stable manifest URL and then select the platform artifact:

```text
https://github.com/demixr/scnet-executorch/releases/latest/download/models.json
```

Every artifact accepts a normalized complex spectrum shaped
`[1, 4, 2049, 338]` and returns four stems shaped
`[1, 4, 4, 2049, 338]`. Decode, STFT/normalization,
denormalization/iSTFT, chunk overlap-add, and WAV writing stay in the app.

## Why Core ML is native

SCNet contains 12 bidirectional LSTMs. ExecuTorch's export path decomposes
those into thousands of primitive operations. Native Core ML preserves all 12
as `lstm` operations: conversion takes seconds and warm inference for one
chunk measured about 0.36 seconds on an Apple Silicon development Mac with
correlation `0.99998866` against PyTorch.

Vulkan currently has no native LSTM operator, so its `.pte` is larger to lower
and contains many layout transitions. It nevertheless loads and executes end
to end on Android. The exporter records mobile 3D-texture limits and keeps the
2049-bin boundary buffer-backed, avoiding invalid image allocation on GPUs
whose 3D texture axes are limited to 2048.

## Export locally

```sh
git clone https://github.com/starrytong/SCNet.git SCNet
uv sync
uv run gdown 1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW -O checkpoint.th

mkdir -p dist
uv run python export_scnet_core.py \
  --source SCNet --checkpoint checkpoint.th \
  --segment-samples 343980 --output dist/scnet_cpu.onnx
uv run python export_scnet_coreml_native.py \
  --source SCNet --checkpoint checkpoint.th \
  --segment-samples 343980 --output dist/scnet_coreml.mlpackage
uv run python export_scnet_executorch.py vulkan \
  --source SCNet --checkpoint checkpoint.th \
  --segment-samples 343980 --output dist/scnet_vulkan.pte
```

Core ML export and validation require macOS. Vulkan artifacts are portable;
the release workflow builds them on macOS because it also builds Core ML.

## Validation performed

- Core ML: 12 native LSTM operations, first run 3.11 s, warm run 0.36 s,
  PyTorch correlation 0.99998866.
- Vulkan: module loaded and a full chunk executed through
  `executorch_flutter` on an Android emulator. SwiftShader measured 26.9 s
  load and 27.8 s inference; those are compatibility timings, not physical
  Android GPU performance claims.
- ONNX CPU: valid ONNX Runtime output; on the complete 50-track MUSDB18-7
  sample test set it was 25% faster than Demixr's HTDemucs ONNX model and
  improved the average per-stem median SDR by 0.34 dB (bass was 0.74 dB lower).

## Licensing and attribution

The deployment scripts are MIT licensed. SCNet is MIT licensed by its
authors; see the [official repository](https://github.com/starrytong/SCNet)
and cite the SCNet paper when appropriate. PyTorch, ExecuTorch, Core ML Tools,
ONNX, and ONNX Runtime retain their respective licenses.
