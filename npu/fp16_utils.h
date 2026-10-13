#pragma once

#include <cstdint>
#include <cstring>

// Small, dependency-free IEEE-754 binary32 <-> binary16 conversion helpers.
uint16_t f32ToF16(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xffu) return static_cast<uint16_t>(sign | (mant ? 0x7e00u : 0x7c00u));
    int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t sub = mant >> shift;
        const uint32_t mask = (1u << shift) - 1u;
        const uint32_t rem = mant & mask;
        const uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (sub & 1u))) sub++;
        return static_cast<uint16_t>(sign | sub);
    }
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13));
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return h;
}

float f16ToF32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t e = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        if (mant == 0) x = sign;
        else {
            int32_t ee = -14;
            while (!(mant & 0x400u)) { mant <<= 1; --ee; }
            mant &= 0x3ffu;
            x = sign | (static_cast<uint32_t>(ee + 127) << 23) | (mant << 13);
        }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((e - 15u + 127u) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &x, sizeof(f)); return f;
}
