# Terminal QNN/HTP port plan

## What was taken from MCNPU

The reference implementation is `bslsjdk/mcnpu`, especially `app/src/main/cpp/mcnpu.cpp` and `NpuRuntime.java`. The reusable native sequence is:

1. Load `libQnnHtp.so` with `dlopen(RTLD_NOW | RTLD_GLOBAL)`.
2. Resolve `QnnInterface_getProviders`.
3. Select the provider with backend ID 6 (the HTP provider used by MCNPU).
4. Create the QNN backend, device, and context.
5. Build tensors/graphs and execute through the provider's QNN interface.
6. Bound graph caching and release the context at shutdown.

The Android-only wrapper is not portable: JNI entry points, Android logging, `Context` asset extraction, and app-private directories must be replaced with command-line arguments, stderr diagnostics, and explicit library/work directories.

## Current code in this repository

- `npu/qnn_probe.cpp`: terminal-only provider/backend/device/context initialization probe.
- `npu/qnn_matmul_smoke.cpp`: attempts a real FP16 QNN matmul graph and checks the result against a CPU reference.
- `npu/CMakeLists.txt`: standalone build for the probe using QNN headers.
- `scripts/bootstrap-upstream.sh`: obtains upstream `llama.cpp` source without building an APK.

The initialization probe alone does not execute a graph. The separate `qnn-matmul-smoke` target now attempts a real HTP matmul and validates the result against a CPU reference. It is a standalone backend smoke test, not yet wired into llama.cpp; it has not been compiled or run on the target phone.

## Build the probe

On an AArch64 Termux environment with a compatible QNN header directory:

```sh
cmake -S npu -B build-qnn-probe -DQNN_INC=/path/to/QNN/include
cmake --build build-qnn-probe -j1
bash scripts/run-qnn-probe.sh /path/to/qnn/libs
bash scripts/run-qnn-matmul-smoke.sh /path/to/qnn/libs
```

Use the same QNN header/runtime release family. A mismatched interface header and runtime can fail before provider discovery.

## Integration gates

1. Build the terminal probe and record provider/backend/device/context return codes.
2. Build and run `qnn-matmul-smoke`; fix any QNN-version/API incompatibilities and establish real HTP graph execution plus CPU parity.
3. Add a GGML backend that implements buffer allocation, tensor support checks, graph planning, graph creation, graph execution, and synchronization. Start with one verified operation, not whole-model offload.
4. Compare QNN output against CPU output for supported shapes and datatypes.
5. Add explicit fallback to CPU for unsupported ops or failed initialization.
6. Measure end-to-end token latency, memory, and stability on the actual phone. Measure peak runtime RAM and report it accurately; do not impose an arbitrary 4 GiB ceiling. The selected model, context, and available device memory determine viable settings.
7. Only advertise llama.cpp acceleration after a real GGML graph has executed on HTP and passed parity tests.

## Important known issue

The separate `phonemesh-rpc/llama.cpp` work had `dspqueue_write` failing with `0x48`. That FastRPC queue path is not interchangeable with QNN's public graph API. Do not copy an unverified FastRPC descriptor workaround into this QNN runtime. Keep the two paths separate until each has its own passing test.
