#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <cub/cub.cuh>

namespace internal {
    typedef std::uint32_t u32;
    typedef std::uint64_t u64;

    // lo = a * b + c (mod 2^n)
    __device__ __forceinline__
    void
    mad_lo(u32 &lo, u32 a, u32 b, u32 c) {
        lo = a * b + c;
    }

    __device__ __forceinline__
    void
    mad_lo(u64 &lo, u64 a, u64 b, u64 c) {
        lo = a * b + c;
    }

    __device__ __forceinline__
    void
    mad_hi(u32 &hi, u32 a, u32 b, u32 c) {
        hi = __umulhi(a, b) + c;
    }

    __device__ __forceinline__
    void
    mad_hi(u64 &hi, u64 a, u64 b, u64 c) {
        hi = __umul64hi(a, b) + c;
    }
} // End namespace internal
