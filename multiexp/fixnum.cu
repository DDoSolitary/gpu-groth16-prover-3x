#pragma once

#include <cassert>

/*
 * var is the basic register type that we deal with. The
 * interpretation of (one or more) such registers is determined by the
 * struct used, e.g. digit, fixnum, etc.
 */

static constexpr size_t ELT_LIMBS32 = 24;
static constexpr size_t ELT_BYTES = ELT_LIMBS32 * sizeof(uint32_t);

struct fixnum {
    __device__ __forceinline__
    static void
    set_zero(uint32_t *a) { memset(a, 0, ELT_BYTES); }

    __device__ __forceinline__
    static void
    set_one(uint32_t *a) {
        set_zero(a);
        a[0] = 1;
    }

    __device__
    static void
    add_cy(uint32_t *r, int &cy_hi, const uint32_t *a, const uint32_t *b) {
        int cy = 0;
        for (int i = 0; i < ELT_LIMBS32; i++) {
            auto t = (uint64_t)a[i] + b[i] + cy;
            r[i] = t;
            cy = t >> 32;
        }
        cy_hi = cy;
    }

    __device__
    static void
    add(uint32_t *r, const uint32_t *a, const uint32_t *b) {
        int cy_hi;
        add_cy(r, cy_hi, a, b);
    }

    __device__
    static void
    sub_br(uint32_t *r, int &br_lo, const uint32_t *a, const uint32_t *b) {
        int br = 0;
        for (int i = 0; i < ELT_LIMBS32; i++) {
            auto t = (uint64_t)a[i] - b[i] - br;
            r[i] = t;
            br = -(uint32_t)(t >> 32);
        }
        br_lo = br;
    }

    __device__
    static void
    sub(uint32_t *r, const uint32_t *a, const uint32_t *b) {
        int br_lo;
        sub_br(r, br_lo, a, b);
    }

    __device__ static int is_zero(const uint32_t *r) {
        for (int i = 0; i < ELT_LIMBS32; i++) {
            if (r[i]) {
                return 0;
            }
        }
        return 1;
    }

    __device__ static int most_sig_dig(const uint32_t *x) {
        for (int i = ELT_LIMBS32 - 1; i >= 0; i--) {
            if (x[i]) {
                return i;
            }
        }
        return -1;
    }
};
