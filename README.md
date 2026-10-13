# llama-npu (terminal-first)

This repository is being prepared as a terminal-first Qualcomm Hexagon/QNN integration for `llama.cpp`. It is not an Android APK project.

## Design constraints

- Keep the upstream `llama.cpp` source as the inference engine; do not replace it with the unrelated MCNPU app.
- Use `bslsjdk/mcnpu` as the reference for QNN/HTP initialization, provider discovery, backend/device/context lifetime, graph creation, and diagnostics.
- Port the reusable native QNN runtime into a GGML backend or an explicitly selected offload path. Android-only JNI and `Context`/asset extraction code must not be copied into the terminal executable.
- Keep CPU inference available as a safe fallback until a QNN operation is verified end-to-end.
- Do not impose an arbitrary 4 GiB runtime-RAM ceiling; support the available device memory and let the user choose model/ context sizes that fit their phone.
- Never claim NPU acceleration until the terminal build demonstrates an actual graph execution and output parity.

## Current status

The repository was empty when initialized. The next integration stage must bring in upstream `llama.cpp` source and port the native QNN pieces from `bslsjdk/mcnpu` into a terminal-compatible GGML backend. This README is not a claim that integration or device validation is already complete.

Reference project: https://github.com/bslsjdk/mcnpu
Upstream engine: https://github.com/ggml-org/llama.cpp
