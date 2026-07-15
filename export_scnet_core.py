#!/usr/bin/env python3
"""Export the official SCNet checkpoint as a deployment-friendly ONNX core.

The reference SCNet graph contains complex STFT/iSTFT operations and alternating
RFFT/IRFFT feature conversions. PyTorch can serialize the full graph, but ONNX
Runtime 1.27 rejects the generated ScatterND nodes. This exporter therefore:

* keeps the audio STFT, scalar normalization, and iSTFT outside the model;
* replaces the internal complex FFTs with equivalent real matrix operations;
* exports a fixed-frame spectral core that CPU and GPU providers can share.

The script intentionally imports SCNet from a separately cloned official
repository. We do not vendor or subtly fork the research implementation.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import torch
from torch import Tensor, nn


class RealFeatureConversion(nn.Module):
    """Real-valued equivalent of SCNet's normalized RFFT or IRFFT."""

    def __init__(self, channels: int, time_frames: int, inverse: bool) -> None:
        super().__init__()
        self.channels = channels
        self.inverse = inverse
        n = time_frames
        bins = n // 2 + 1

        sample = torch.arange(n, dtype=torch.float32)
        frequency = torch.arange(bins, dtype=torch.float32)
        angle = 2 * math.pi * frequency[:, None] * sample[None, :] / n

        if inverse:
            # torch.fft.irfft(..., norm="ortho") represented as two real
            # matrix multiplies. Interior positive-frequency bins contribute
            # twice because their negative-frequency conjugates are implicit.
            weights = torch.full((bins,), 2.0)
            weights[0] = 1.0
            if n % 2 == 0:
                weights[-1] = 1.0
            scale = weights[:, None] / math.sqrt(n)
            self.register_buffer("real_matrix", torch.cos(angle) * scale)
            self.register_buffer("imag_matrix", -torch.sin(angle) * scale)
        else:
            scale = 1.0 / math.sqrt(n)
            self.register_buffer("real_matrix", torch.cos(angle) * scale)
            self.register_buffer("imag_matrix", -torch.sin(angle) * scale)

    def forward(self, x: Tensor) -> Tensor:
        if self.inverse:
            real = x[:, : self.channels // 2]
            imag = x[:, self.channels // 2 :]
            return torch.matmul(real, self.real_matrix) + torch.matmul(
                imag, self.imag_matrix
            )
        real = torch.matmul(x, self.real_matrix.transpose(0, 1))
        imag = torch.matmul(x, self.imag_matrix.transpose(0, 1))
        return torch.cat((real, imag), dim=1)


class SCNetSpectralCore(nn.Module):
    """SCNet from normalized complex spectrogram to normalized stem spectra."""

    def __init__(
        self, model: nn.Module, time_frames: int, *, fixed_shape: bool = False
    ) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.separation_net = model.separation_net
        self.sources = len(model.sources)
        self.initial_channels = model.dims[0]
        self.fixed_shape = fixed_shape
        self.time_frames = time_frames

        for index, conversion in enumerate(self.separation_net.feature_conversion):
            inverse = bool(conversion.inverse)
            self.separation_net.feature_conversion[index] = RealFeatureConversion(
                conversion.channels, time_frames, inverse
            )

    def forward(self, x: Tensor) -> Tensor:
        batch, _, frequency_bins, time_frames = x.shape
        skips: deque[Tensor] = deque()
        lengths: deque[list[int]] = deque()
        original_lengths: deque[list[int]] = deque()

        for layer in self.encoder:
            x, skip, band_lengths, band_original_lengths = layer(x)
            skips.append(skip)
            lengths.append(band_lengths)
            original_lengths.append(band_original_lengths)

        x = self.separation_net(x)

        for fusion, upsample in self.decoder:
            x = fusion(x, skips.pop())
            x = upsample(x, lengths.pop(), original_lengths.pop())

        if self.fixed_shape:
            # TorchScript/Core ML conversion otherwise emits
            # `aten::Int(NumToTensor(size(x)))` nodes even though deployment
            # inputs are fixed. Core ML Tools 9 misreads those values as
            # arrays. Keep the general dynamic path for ONNX and use literal
            # deployment dimensions only for the fixed Core ML artifact.
            return x.view(
                1,
                self.sources,
                self.initial_channels,
                2049,
                self.time_frames,
            )
        return x.view(
            batch,
            self.sources,
            self.initial_channels,
            frequency_bins,
            time_frames,
        )


def _load_official_model(source: Path, checkpoint: Path) -> nn.Module:
    sys.path.insert(0, str(source))
    from scnet.SCNet import SCNet  # pylint: disable=import-error,import-outside-toplevel

    model = SCNet().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("module."): value for key, value in saved["best_state"].items()
    }
    model.load_state_dict(state)
    return model


def _verify_feature_conversions(model: nn.Module, time_frames: int) -> None:
    torch.manual_seed(7)
    for index, reference in enumerate(model.separation_net.feature_conversion):
        inverse = bool(reference.inverse)
        frames = time_frames // 2 + 1 if inverse else time_frames
        replacement = RealFeatureConversion(reference.channels, time_frames, inverse)
        channels = reference.channels if inverse else reference.channels // 2
        sample = torch.randn(1, channels, 3, frames)
        expected = reference(sample)
        actual = replacement(sample)
        error = (expected - actual).abs().max().item()
        if error > 5e-4:
            raise RuntimeError(f"feature conversion {index} differs by {error:.6g}")
    print("Internal FFT replacement: numerically equivalent")


def _waveform_to_spectrum(waveform: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    """Apply the official SCNet padding, STFT, layout, and normalization."""
    hop = 1024
    padding = hop - waveform.shape[-1] % hop
    if (waveform.shape[-1] + padding) // hop % 2 == 0:
        padding += hop
    padded = torch.nn.functional.pad(waveform, (0, padding))
    batch, channels, length = padded.shape
    spectrum = torch.stft(
        padded.reshape(-1, length),
        n_fft=4096,
        hop_length=hop,
        win_length=4096,
        center=True,
        normalized=True,
        return_complex=True,
    )
    spectrum = torch.view_as_real(spectrum)
    frequency_bins, time_frames = spectrum.shape[1:3]
    spectrum = spectrum.permute(0, 3, 1, 2).reshape(
        batch, channels * 2, frequency_bins, time_frames
    )
    mean = spectrum.mean(dim=(1, 2, 3), keepdim=True)
    std = spectrum.std(dim=(1, 2, 3), keepdim=True)
    return (spectrum - mean) / (1e-5 + std), mean, std, padding


def _spectrum_to_waveform(
    spectrum: Tensor, mean: Tensor, std: Tensor, padding: int
) -> Tensor:
    """Reverse [_waveform_to_spectrum] exactly like official SCNet."""
    batch, sources, channels, frequency, frames = spectrum.shape
    spectrum = spectrum * std[:, None] + mean[:, None]
    spectrum = spectrum.reshape(-1, 2, frequency, frames).permute(0, 2, 3, 1)
    complex_spectrum = torch.view_as_complex(spectrum.contiguous())
    waveform = torch.istft(
        complex_spectrum,
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        center=True,
        normalized=True,
    )
    waveform = waveform.reshape(batch, sources, channels // 2, -1)
    return waveform[..., :-padding]


def _report_waveform_parity(reference: Tensor, actual: Tensor, label: str) -> None:
    correlation = float(
        torch.corrcoef(torch.stack((reference.flatten(), actual.flatten())))[0, 1]
    )
    maximum = float((reference - actual).abs().max())
    mean = float((reference - actual).abs().mean())
    print(
        f"{label} waveform parity: corr={correlation:.8f} "
        f"max_abs={maximum:.6g} mean_abs={mean:.6g}"
    )
    if correlation < 0.999:
        raise RuntimeError(f"{label} waveform correlation is too low: {correlation}")


def _check_onnx(
    path: Path,
    sample: Tensor,
    reference_model: nn.Module,
    segment_samples: int,
) -> None:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(["stems_spec"], {"mix_spec": sample.numpy()})[0]
    if list(output.shape) != [1, 4, 4, sample.shape[2], sample.shape[3]]:
        raise RuntimeError(f"unexpected ONNX output shape: {output.shape}")
    print(f"ONNX Runtime CPU: valid, output shape {list(output.shape)}")

    torch.manual_seed(23)
    waveform = torch.randn(1, 2, segment_samples) * 0.05
    spectrum, mean, std, padding = _waveform_to_spectrum(waveform)
    runtime_output = session.run(
        ["stems_spec"], {"mix_spec": spectrum.numpy()}
    )[0]
    actual = _spectrum_to_waveform(
        torch.from_numpy(runtime_output), mean, std, padding
    )
    with torch.inference_mode():
        reference = reference_model(waveform)
    _report_waveform_parity(reference, actual, "ONNX")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-samples", type=int, default=343_980)
    parser.add_argument("--skip-runtime-check", action="store_true")
    args = parser.parse_args()

    hop = 1024
    padding = hop - args.segment_samples % hop
    if (args.segment_samples + padding) // hop % 2 == 0:
        padding += hop
    time_frames = (args.segment_samples + padding) // hop + 1

    model = _load_official_model(args.source, args.checkpoint)
    _verify_feature_conversions(model, time_frames)
    core = SCNetSpectralCore(model, time_frames).eval()
    sample = torch.zeros(1, 4, 2049, time_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core,
        (sample,),
        args.output,
        input_names=["mix_spec"],
        output_names=["stems_spec"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")

    if not args.skip_runtime_check:
        # SCNetSpectralCore replaces feature-conversion modules in-place.
        # Validate reconstructed audio against a fresh, untouched reference.
        reference_model = _load_official_model(args.source, args.checkpoint)
        _check_onnx(args.output, sample, reference_model, args.segment_samples)


if __name__ == "__main__":
    main()
