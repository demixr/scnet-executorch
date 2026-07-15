# Prebuilt deployment artifacts

`scnet_vulkan.pte` is committed because lowering SCNet's bidirectional LSTMs
temporarily exceeds the memory available on GitHub-hosted macOS runners. It is
produced reproducibly by `export_scnet_executorch.py` and was validated through
`executorch_flutter` with a full 7.8-second inference on Android.

SHA-256:

```text
de2d8d298987a7c35ad9caf1d151a70695b229af73eb39e1551c7d71a14298d7  scnet_vulkan.pte
```
