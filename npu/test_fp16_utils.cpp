#include "fp16_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdint>

static bool check(float input, float expected, float tolerance, const char *label) {
    const uint16_t half = f32ToF16(input);
    const float output = f16ToF32(half);
    const bool ok = std::isfinite(expected) ? std::fabs(output - expected) <= tolerance
                                            : !std::isfinite(output);
    std::printf("%s input=%.9g half=0x%04x output=%.9g expected=%.9g %s\n",
                label, input, static_cast<unsigned>(half), output, expected, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check(0.0f, 0.0f, 0.0f, "zero");
    ok &= check(-0.0f, -0.0f, 0.0f, "negative-zero");
    ok &= check(1.0f, 1.0f, 0.0f, "one");
    ok &= check(-2.0f, -2.0f, 0.0f, "negative-two");
    ok &= check(65504.0f, 65504.0f, 0.0f, "max-finite");
    ok &= check(0.00006103515625f, 0.00006103515625f, 0.0f, "min-normal");
    ok &= check(0.000000059604644775390625f, 0.000000059604644775390625f, 0.0f, "min-subnormal");
    ok &= check(0.0000000298023223876953125f, 0.0f, 0.0f, "subnormal-tie-rounds-to-even");
    ok &= check(1.00048828125f, 1.0f, 0.0f, "normal-tie-rounds-to-even");
    ok &= check(1.00146484375f, 1.001953125f, 0.0f, "normal-tie-rounds-up-to-even");
    if (!ok) return 1;
    std::puts("PASS all FP16 conversion tests");
    return 0;
}
