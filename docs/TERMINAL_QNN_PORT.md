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
- `npu/CMakeLists.txt`: standalone build for the probe using QNN headers.
- `scripts/bootstrap-upstream.sh`: obtains upstream `llama.cpp` source without building an APK.

The probe is only an initialization test. It does **not** yet construct or execute a QNN graph and does **not** accelerate llama.cpp.

## Build the probe

On an AArch64 Termux environment with a compatible QNN header directory:

```sh
cmake -S npu -B build-qnn-probe -DQNN_INC=/path/to/QNN/include
cmake --build build-qnn-probe -j1
LD_LIBRARY_PATH=/path/to/qnn/libs:$LD_LIBRARY_PATH \
  ./build-qnn-probe/qnn-probe /path/to/qnn/libs
```

Use the same QNN header/runtime release family. A mismatched interface header and runtime can fail before provider discovery.

## Integration gates

1. Build the terminal probe and record provider/backend/device/context return codes.
2. Add a GGML backend that implements buffer allocation, tensor support checks, graph planning, graph creation, graph execution, and synchronization. Start with one verified operation, not whole-model offload.
3. Compare QNN output against CPU output for supported shapes and datatypes.
4. Add explicit fallback to CPU for unsupported ops or failed initialization.
5. Measure end-to-end token latency, memory, and stability on the actual phone. Peak runtime RAM must stay below 4 GiB.
6. Only advertise acceleration after a real GGML graph has executed on HTP and passed parity tests.

## Important known issue

The separate `phonemesh-rpc/llama.cpp` work had `dspqueue_write` failing with `0x48`. That FastRPC queue path is not interchangeable with QNN's public graph API. Do not copy an unverified FastRPC descriptor workaround into this QNN runtime. Keep the two paths separate until each has its own passing test.
