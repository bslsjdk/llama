#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnTypes.h"

namespace {
constexpr uint32_t kHtpBackendId = 6;
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);

void report(const char *stage, Qnn_ErrorHandle_t rc) {
    std::fprintf(stderr, "%s rc=%d (0x%x)\n", stage, static_cast<int>(rc),
                 static_cast<unsigned>(rc));
}

int run(const std::string &libDir) {
    const std::string libPath = libDir + "/libQnnHtp.so";
    void *handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::fprintf(stderr, "ERR DLOPEN path=%s reason=%s\n",
                     libPath.c_str(), dlerror() ? dlerror() : "unknown");
        return 2;
    }

    auto getProviders = reinterpret_cast<GetProviders>(
        dlsym(handle, "QnnInterface_getProviders"));
    if (!getProviders) {
        std::fprintf(stderr, "ERR SYMBOL QnnInterface_getProviders missing: %s\n",
                     dlerror() ? dlerror() : "unknown");
        dlclose(handle);
        return 3;
    }

    const QnnInterface_t **providers = nullptr;
    uint32_t count = 0;
    Qnn_ErrorHandle_t rc = getProviders(&providers, &count);
    report("getProviders", rc);
    if (rc != QNN_SUCCESS || !providers || count == 0) {
        std::fprintf(stderr, "ERR NO_PROVIDERS count=%u\n", count);
        dlclose(handle);
        return 4;
    }

    const QnnInterface_t *api = nullptr;
    for (uint32_t i = 0; i < count; ++i) {
        if (providers[i] && providers[i]->backendId == kHtpBackendId) {
            api = providers[i];
            break;
        }
    }
    if (!api) {
        std::fprintf(stderr, "ERR HTP_PROVIDER_MISSING providers=%u wanted_backend_id=%u\n",
                     count, kHtpBackendId);
        dlclose(handle);
        return 5;
    }

    const auto &f = api->QNN_INTERFACE_VER_NAME;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;

    rc = f.backendCreate(nullptr, nullptr, &backend);
    report("backendCreate", rc);
    if (rc != QNN_SUCCESS || !backend) {
        dlclose(handle);
        return 6;
    }

    rc = f.deviceCreate(nullptr, nullptr, &device);
    report("deviceCreate", rc);
    if (rc != QNN_SUCCESS || !device) {
        if (f.backendFree) f.backendFree(backend);
        dlclose(handle);
        return 7;
    }

    rc = f.contextCreate(backend, device, nullptr, &context);
    report("contextCreate", rc);
    if (rc != QNN_SUCCESS || !context) {
        if (f.deviceFree) f.deviceFree(device);
        if (f.backendFree) f.backendFree(backend);
        dlclose(handle);
        return 8;
    }

    std::printf("OK QNN_HPT_RUNTIME_INITIALIZED providers=%u backend_id=%u\n",
                count, kHtpBackendId);
    std::printf("NOTE This probe validates provider/backend/device/context initialization only; "
                "it does not yet execute a graph or prove llama.cpp acceleration.\n");

    if (f.contextFree) f.contextFree(context, nullptr);
    if (f.deviceFree) f.deviceFree(device);
    if (f.backendFree) f.backendFree(backend);
    dlclose(handle);
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s /path/to/qnn/libs\n", argv[0]);
        return 64;
    }
    return run(argv[1]);
}
