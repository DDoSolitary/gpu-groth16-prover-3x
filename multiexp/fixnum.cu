#pragma once

#include <cub/cub.cuh>

#include "primitives.cu"

/*
 * var is the basic register type that we deal with. The
 * interpretation of (one or more) such registers is determined by the
 * struct used, e.g. digit, fixnum, etc.
 */
typedef std::uint64_t var;

static constexpr size_t ELT_LIMBS = 12;
static constexpr size_t ELT_BYTES = ELT_LIMBS * sizeof(var);

static constexpr size_t BIG_WIDTH = ELT_LIMBS + 4; // = 16


struct digit {
    static constexpr int BYTES = sizeof(var);
    static constexpr int BITS = BYTES * 8;

    __device__ __forceinline__
    static void
    add(var &s, var a, var b) {
        s = a + b;
    }

    __device__ __forceinline__
    static void
    add_cy(var &s, int &cy, var a, var b) {
        s = a + b;
        cy = s < a;
    }

    __device__ __forceinline__
    static void
    sub(var &d, var a, var b) {
        d = a - b;
    }

    __device__ __forceinline__
    static void
    sub_br(var &d, int &br, var a, var b) {
        d = a - b;
        br = d > a;
    }

    __device__ __forceinline__
    static var
    zero() { return 0ULL; }

    __device__ __forceinline__
    static int
    is_max(var a) { return a == ~0ULL; }

    __device__ __forceinline__
    static int
    is_min(var a) { return a == 0ULL; }

    __device__ __forceinline__
    static int
    is_zero(var a) { return a == zero(); }

    __device__ __forceinline__
    static void
    mul_lo(var &lo, var a, var b) {
        lo = a * b;
    }

    // lo = a * b + c (mod 2^64)
    __device__ __forceinline__
    static void
    mad_lo(var &lo, var a, var b, var c) {
        internal::mad_lo(lo, a, b, c);
    }

    // as above but increment cy by the mad carry
    __device__ __forceinline__
    static void
    mad_lo_cy(var &lo, int &cy, var a, var b, var c) {
        internal::mad_lo(lo, a, b, c);
        cy += lo < c;
    }

    __device__ __forceinline__
    static void
    mad_hi(var &hi, var a, var b, var c) {
        internal::mad_hi(hi, a, b, c);
    }

    // as above but increment cy by the mad carry
    __device__ __forceinline__
    static void
    mad_hi_cy(var &hi, int &cy, var a, var b, var c) {
        internal::mad_hi(hi, a, b, c);
        cy += hi < c;
    }
};

template<int WIDTH>
class fixnum_layout {
#ifdef __ILUVATAR__
    typedef uint64_t Mask;
#else
    typedef uint32_t Mask;
#endif
    Mask mask;

public:
    __device__
    fixnum_layout() {
        mask = (Mask)-1 >> (CUB_PTX_WARP_THREADS - WIDTH);
        mask <<= cub::LaneId() & ~(WIDTH - 1);
    }

    __device__ __forceinline__
    Mask ballot(int val) {
        return (cub::WARP_BALLOT(val, mask) & mask) >> (cub::LaneId() & ~(WIDTH - 1));
    }

    template<typename T>
    __device__ __forceinline__
    T shfl(T val, int src) {
        return cub::ShuffleIndex<WIDTH>(val, src, mask);
    }

    template<typename T>
    __device__ __forceinline__
    T shfl_up(T val, int offset) {
        return cub::ShuffleUp<WIDTH>(val, offset, 0, mask);
    }

    template<typename T>
    __device__ __forceinline__
    T shfl_down(T val, int src) {
        return cub::ShuffleDown<WIDTH>(val, src, WIDTH - 1, mask);
    }
};

struct fixnum {
    // 16 because digit::BITS * 16 = 1024 > 768 = digit::bits * 12
    // Must be < 32 for effective_carries to work.
    static constexpr unsigned WIDTH = 16;

    __device__
    static fixnum_layout<WIDTH> layout() {
        return {};
    }

    __device__ __forceinline__
    static unsigned thread_rank() {
        return cub::LaneId() & (WIDTH - 1);
    }

    __device__ __forceinline__
    static var
    zero() { return digit::zero(); }

    __device__ __forceinline__
    static var
    one() {
        auto t = thread_rank();
        return (var)(t == 0);
    }

    __device__
    static void
    add_cy(var &r, int &cy_hi, const var &a, const var &b) {
        int cy;
        digit::add_cy(r, cy, a, b);
        // r propagates carries iff r = FIXNUM_MAX
        var r_cy = effective_carries(cy_hi, digit::is_max(r), cy);
        digit::add(r, r, r_cy);
    }

    __device__
    static void
    add(var &r, const var &a, const var &b) {
        int cy_hi;
        add_cy(r, cy_hi, a, b);
    }

    __device__
    static void
    sub_br(var &r, int &br_lo, const var &a, const var &b) {
        int br;
        digit::sub_br(r, br, a, b);
        // r propagates borrows iff r = FIXNUM_MIN
        var r_br = effective_carries(br_lo, digit::is_min(r), br);
        digit::sub(r, r, r_br);
    }

    __device__
    static void
    sub(var &r, const var &a, const var &b) {
        int br_lo;
        sub_br(r, br_lo, a, b);
    }

    __device__ static auto nonzero_mask(var r) {
        return fixnum::layout().ballot( ! digit::is_zero(r));
    }

    __device__ static int is_zero(var r) {
        return nonzero_mask(r) == 0U;
    }

    __device__ static int most_sig_dig(var x) {
        auto a = nonzero_mask(x);
        return (sizeof(a) * 8) - (internal::clz(a) + 1);
    }

    __device__ static int cmp(var x, var y) {
        var r;
        int br;
        sub_br(r, br, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    __device__
    static var
    effective_carries(int &cy_hi, int propagate, int cy) {
        auto grp = fixnum::layout();

        auto g = grp.ballot(cy);                  // carry generate
        auto p = grp.ballot(propagate);           // carry propagate
        auto allcarries = (p | g) + g;            // propagate all carries
        cy_hi = (allcarries >> WIDTH) & 1;        // detect hi overflow
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        return (allcarries >> thread_rank()) & 1;
    }
};
