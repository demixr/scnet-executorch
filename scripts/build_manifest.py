#!/usr/bin/env python3
"""Create release metadata consumed by Demixr's model downloader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    args = parser.parse_args()

    base = (
        "https://github.com/demixr/scnet-executorch/releases/download/"
        f"{args.tag}"
    )
    definitions = {
        "coreml": ("scnet_coreml.mlpackage.zip", ["macos", "ios"]),
        "vulkan": ("scnet_vulkan.pte", ["android"]),
        "onnx": ("scnet_cpu.onnx", ["macos", "ios", "android", "windows", "linux"]),
    }
    artifacts = {}
    for name, (filename, platforms) in definitions.items():
        path = args.directory / filename
        artifacts[name] = {
            "filename": filename,
            "platforms": platforms,
            "url": f"{base}/{filename}",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    manifest = {
        "schema": 1,
        "model": "scnet",
        "version": args.tag,
        "sample_rate": 44100,
        "segment_samples": 343980,
        "input_shape": [1, 4, 2049, 338],
        "output_shape": [1, 4, 4, 2049, 338],
        "stems": ["drums", "bass", "other", "vocals"],
        "artifacts": artifacts,
    }
    destination = args.directory / "models.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

