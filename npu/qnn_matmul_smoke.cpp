#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnOpDef.h"
#include "QnnTypes.h"

namespace {
constexpr uint32_t kHtpBackendId = 6;
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);

void report(const char *stage, Qnn_ErrorHandle_t rc) {
    std::fprintf(stderr, "%s rc=%d (0x%x)\n", stage, static_cast<int>(rc),
                 static_cast<unsigned>(rc));
}

Qnn_Tensor_t makeTensor(const char *name, Qnn_TensorType_t type,
                        uint32_t *dims, uint32_t rank) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = QNN_DATATYPE_FLOAT_32;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}

int run(const std::string &libDir) {
    void *handle = dlopen((libDir + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char *err = dlerror();
        std::fprintf(stderr, "ERR DLOPEN: %s\n", err ? err : "unknown");
        return 2;
    }

    int result = 1;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    const QnnInterface_t *api = nullptr;

    dlerror();
    auto getProviders = reinterpret_cast<GetProviders>(dlsym(handle, "QnnInterface_getProviders"));
    const char *symErr = dlerror();
    if (!getProviders || symErr) {
        std::fprintf(stderr, "ERR SYMBOL QnnInterface_getProviders: %s\n",
                     symErr ? symErr : "null symbol");
        result = 3;
        goto cleanup;
    }

    {
        const QnnInterface_t **providers = nullptr;
        uint32_t count = 0;
        Qnn_ErrorHandle_t rc = getProviders(&providers, &count);
        report("getProviders", rc);
        if (rc != QNN_SUCCESS || !providers || count == 0) {
            result = 4;
            goto cleanup;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (providers[i] && providers[i]->backendId == kHtpBackendId) {
                api = providers[i];
                break;
            }
        }
        if (!api) {
            std::fprintf(stderr, "ERR HTP provider backend_id=%u not found (providers=%u)\n",
                         kHtpBackendId, count);
            result = 5;
            goto cleanup;
        }
    }

    {
        const auto &f = api->QNN_INTERFACE_VER_NAME;
        Qnn_ErrorHandle_t rc = f.backendCreate(nullptr, nullptr, &backend);
        report("backendCreate", rc);
        if (rc != QNN_SUCCESS || !backend) { result = 6; goto cleanup; }
        rc = f.deviceCreate(nullptr, nullptr, &device);
        report("deviceCreate", rc);
        if (rc != QNN_SUCCESS || !device) { result = 7; goto cleanup; }
        rc = f.contextCreate(backend, device, nullptr, &context);
        report("contextCreate", rc);
        if (rc != QNN_SUCCESS || !context) { result = 8; goto cleanup; }

        // A tiny real HTP matmul: A[2x3] * B[3x2] = C[2x2].
        // QNN tensor dimensions are innermost-first for this row-major layout.
        uint32_t da[2] = {3, 2}, db[2] = {2, 3}, dc[2] = {2, 2};
        Qnn_Tensor_t a = makeTensor("a", QNN_TENSOR_TYPE_APP_WRITE, da, 2);
        Qnn_Tensor_t b = makeTensor("b", QNN_TENSOR_TYPE_APP_WRITE, db, 2);
        Qnn_Tensor_t c = makeTensor("c", QNN_TENSOR_TYPE_APP_READ, dc, 2);

        rc = f.graphCreate(context, "llama_qnn_matmul_smoke", nullptr, &graph);
        report("graphCreate", rc);
        if (rc != QNN_SUCCESS || !graph) { result = 9; goto cleanup; }
        rc = f.tensorCreateGraphTensor(graph, &a);
        if (rc == QNN_SUCCESS) rc = f.tensorCreateGraphTensor(graph, &b);
        if (rc == QNN_SUCCESS) rc = f.tensorCreateGraphTensor(graph, &c);
        report("tensorCreateGraphTensor", rc);
        if (rc != QNN_SUCCESS) { result = 10; goto cleanup; }

        Qnn_Tensor_t inputs[2] = {a, b};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = "matmul";
        op.v1.packageName = "qti.aisw";
        op.v1.typeName = QNN_OP_MAT_MUL;
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 2;
        op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &c;
        rc = f.graphAddNode(graph, op);
        report("graphAddNode(matmul)", rc);
        if (rc != QNN_SUCCESS) { result = 11; goto cleanup; }
        rc = f.graphFinalize(graph, nullptr, nullptr);
        report("graphFinalize", rc);
        if (rc != QNN_SUCCESS) { result = 12; goto cleanup; }

        const float av[6] = {1, 2, 3, 4, 5, 6};
        const float bv[6] = {7, 8, 9, 10, 11, 12};
        float cv[4] = {-999, -999, -999, -999};
        Qnn_Tensor_t ea = a, eb = b, ec = c;
        ea.v1.clientBuf.data = const_cast<float *>(av);
        ea.v1.clientBuf.dataSize = sizeof(av);
        eb.v1.clientBuf.data = const_cast<float *>(bv);
        eb.v1.clientBuf.dataSize = sizeof(bv);
        ec.v1.clientBuf.data = cv;
        ec.v1.clientBuf.dataSize = sizeof(cv);
        Qnn_Tensor_t execInputs[2] = {ea, eb};
        Qnn_Tensor_t execOutputs[1] = {ec};
        rc = f.graphExecute(graph, execInputs, 2, execOutputs, 1, nullptr, nullptr);
        report("graphExecute(matmul)", rc);
        if (rc != QNN_SUCCESS) { result = 13; goto cleanup; }

        // CPU reference: [58,64;139,154]. Fail rather than silently claiming success.
        const float expected[4] = {58, 64, 139, 154};
        for (size_t i = 0; i < 4; ++i) {
            const float err = std::fabs(cv[i] - expected[i]);
            std::printf("out[%zu]=%.6f expected=%.6f abs_err=%.6f\n",
                        i, cv[i], expected[i], err);
            if (!std::isfinite(cv[i]) || err > 0.1f) {
                std::fprintf(stderr, "ERR OUTPUT_MISMATCH index=%zu\n", i);
                result = 14;
                goto cleanup;
            }
        }
        std::puts("PASS QNN HTP graph executed and matched CPU reference.");
        result = 0;
    }

cleanup:
    if (api) {
        const auto &f = api->QNN_INTERFACE_VER_NAME;
        if (context && f.contextFree) f.contextFree(context, nullptr);
        if (device && f.deviceFree) f.deviceFree(device);
        if (backend && f.backendFree) f.backendFree(backend);
    }
    dlclose(handle);
    return result;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s /path/to/qnn/libs\n", argv[0]);
        return 64;
    }
    return run(argv[1]);
}
