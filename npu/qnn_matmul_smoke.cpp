#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "fp16_utils.h"
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
                        Qnn_DataType_t dataType, uint32_t *dims, uint32_t rank) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dataType;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}



