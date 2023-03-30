#pragma once

#include "fixnum.cu"

__device__ __constant__
const uint32_t MOD_Q[ELT_LIMBS32] = {
    0x245e8001, 0x5e9063de, 0x2cdd119f, 0xe39d5452,
    0x9ac425f0, 0x63881071, 0x767254a4, 0x685acce9,
    0xcb537e38, 0xb80f0da5, 0xf218059d, 0xb117e776,
    0xa15af79d, 0x99d124d9, 0xe8a0ed8d, 0x07fdb925,
    0x6c97d873, 0x5eb7e8f9, 0x5b8fafed, 0xb7f99750,
    0xeee2cdad, 0x10229022, 0x2d92c411, 0x0001c4c6
};

// -Q^{-1} (mod 2^32)
static constexpr uint32_t Q_NINV_MOD = 0xe45e7fff;

// 2^768 mod Q
__device__ __constant__
const uint32_t X_MOD_Q[ELT_LIMBS32] = {
    0xd9dc6f42, 0x98a8ecab, 0x5a034686, 0x91cd31c6,
    0xcd14572e, 0x97c3e4a0, 0xc788b601, 0x79589819,
    0x2108976f, 0xed269c94, 0xcf031d68, 0x1e0f4d8a,
    0x13338559, 0x320c3bb7, 0xd2f00a62, 0x598b4302,
    0xfd8ca621, 0x4074c9cb, 0x3865e88c, 0x0fa47edb,
    0x1ff9a195, 0x95455fb3, 0x9ec8e242, 0x00007b47
};


//template< const var *mod_, const var ninv_mod_, const var *binpow_mod_ >
//struct modulus_info {
struct MNT4_MOD {
    __device__ __forceinline__ static constexpr const uint32_t *mod() { return MOD_Q; }
    static constexpr uint32_t ninv_mod = Q_NINV_MOD;
    __device__ __forceinline__ static constexpr const uint32_t *monty_one() { return X_MOD_Q; }
};
//typedef modulus_info<MOD_Q, Q_NINV_MOD, X_MOD_Q> MNT4_MOD;

__device__ __constant__
const uint32_t MOD_R[ELT_LIMBS32] = {
    0x40000001, 0xd90776e2, 0x0fa13a4f, 0x4ea09917,
    0x3f005797, 0xd6c381bc, 0x34993aa4, 0xb9dff976,
    0x29212636, 0x3eebca94, 0xc859a99b, 0xb26c5c28,
    0xa15af79d, 0x99d124d9, 0xe8a0ed8d, 0x07fdb925,
    0x6c97d873, 0x5eb7e8f9, 0x5b8fafed, 0xb7f99750,
    0xeee2cdad, 0x10229022, 0x2d92c411, 0x0001c4c6
};

// -R^{-1} (mod 2^32)
const uint32_t R_NINV_MOD = 0x3fffffff;

// 2^768 mod R
__device__ __constant__
const uint32_t X_MOD_R[ELT_LIMBS32] = {
    0x7fff6f42, 0xb9968014, 0xb589cea8, 0x4eb16817,
    0x0c79e179, 0xa1ebd2d9, 0xc549c0da, 0x0f725cae,
    0xd3e6dad4, 0xab0c4ee6, 0xde0ccb62, 0x9fbca908,
    0x13338498, 0x320c3bb7, 0xd2f00a62, 0x598b4302,
    0xfd8ca621, 0x4074c9cb, 0x3865e88c, 0x0fa47edb,
    0x1ff9a195, 0x95455fb3, 0x9ec8e242, 0x00007b47
};

struct MNT6_MOD {
    __device__ __forceinline__ static constexpr const uint32_t *mod() { return MOD_R; }
    static constexpr uint32_t ninv_mod = R_NINV_MOD;
    __device__ __forceinline__ static constexpr const uint32_t *monty_one() { return X_MOD_R; }
};

// Apparently we still can't do partial specialisation of function
// templates in C++, so we do it in a class instead. Woot.
template< int n >
struct mul_ {
    template< typename G >
    __device__ static void x(G &z, const G &x);
};

template<>
template< typename G >
__device__ void
mul_<2>::x(G &z, const G &x) {
    // TODO: Shift by one bit
    G::add(z, x, x);
}

template<>
template< typename G >
__device__ void
mul_<4>::x(G &z, const G &x) {
    // TODO: Shift by two bits
    mul_<2>::x(z, x);  // z = 2x
    mul_<2>::x(z, z);  // z = 4x
}

template<>
template< typename G >
__device__ void
mul_<8>::x(G &z, const G &x) {
    // TODO: Shift by three bits
    mul_<4>::x(z, x);  // z = 4x
    mul_<2>::x(z, z);  // z = 8x
}

template<>
template< typename G >
__device__ void
mul_<16>::x(G &z, const G &x) {
    // TODO: Shift by four bits
    mul_<8>::x(z, x);  // z = 8x
    mul_<2>::x(z, z);  // z = 16x
}

template<>
template< typename G >
__device__ void
mul_<32>::x(G &z, const G &x) {
    // TODO: Shift by five bits
    mul_<16>::x(z, x); // z = 16x
    mul_<2>::x(z, z);  // z = 32x
}

template<>
template< typename G >
__device__ void
mul_<64>::x(G &z, const G &x) {
    // TODO: Shift by six bits
    mul_<32>::x(z, x); // z = 32x
    mul_<2>::x(z, z);  // z = 64x
}

template<>
template< typename G >
__device__ void
mul_<3>::x(G &z, const G &x) {
    G t;
    mul_<2>::x(t, x);
    G::add(z, t, x);
}

template<>
template< typename G >
__device__ void
mul_<11>::x(G &z, const G &x) {
    // TODO: Do this without carry/overflow checks
    // TODO: Check that this is indeed optimal
    // 11 = 8 + 2 + 1
    G t;
    mul_<2>::x(t, x);  // t = 2x
    G::add(z, t, x);   // z = 3x
    mul_<4>::x(t, t);  // t = 8x
    G::add(z, z, t);   // z = 11x
}

template<>
template< typename G >
__device__ void
mul_<13>::x(G &z, const G &x) {
    // 13 = 8 + 4 + 1
    G t;
    mul_<4>::x(t, x);  // t = 4x
    G::add(z, t, x);   // z = 5x
    mul_<2>::x(t, t);  // t = 8x
    G::add(z, z, t);   // z = 13x
}

template<>
template< typename G >
__device__ void
mul_<26>::x(G &z, const G &x) {
    // 26 = 16 + 8 + 2
    G t;
    mul_<2>::x(z, x); // z = 2x
    mul_<4>::x(t, z); // t = 8x
    G::add(z, z, t);  // z = 10x
    mul_<2>::x(t, t); // t = 16x
    G::add(z, z, t);  // z = 26x
}

template<>
template< typename G >
__device__ void
mul_<121>::x(G &z, const G &x) {
    // 121 = 64 + 32 + 16 + 8 + 1
    G t;
    mul_<8>::x(t, x); // t = 8x
    G::add(z, t, x);  // z = 9x
    mul_<2>::x(t, t); // t = 16x
    G::add(z, z, t);  // z = 25x
    mul_<2>::x(t, t); // t = 32x
    G::add(z, z, t);  // z = 57x
    mul_<2>::x(t, t); // t = 64x
    G::add(z, z, t);  // z = 121x
}

// TODO: Bleughk! This is obviously specific to MNT6 curve over Fp3.
template<>
template< typename Fp3 >
__device__ void
mul_<-1>::x(Fp3 &z, const Fp3 &x) {
    // multiply by (0, 0, 11) = 11 x^2  (where x^3 = alpha)
    static constexpr int CRV_A = 11;
    static constexpr int ALPHA = 11;
    Fp3 y = x;
    mul_<CRV_A * ALPHA>::x(z.a0, y.a1);
    mul_<CRV_A * ALPHA>::x(z.a1, y.a2);
    mul_<CRV_A>::x(z.a2, y.a0);
}


template< typename modulus_info >
struct Fp {
    typedef Fp PrimeField;

    uint32_t a[ELT_LIMBS32];

    static constexpr int DEGREE = 1;

    __device__
    static void
    load(Fp &x, const uint32_t *mem) {
        memcpy(x.a, mem, ELT_BYTES);
    }

    __device__
    static void
    store(uint32_t *mem, const Fp &x) {
        memcpy(mem, x.a, ELT_BYTES);
    }

    __device__
    static int
    are_equal(const Fp &x, const Fp &y) {
        for (int i = 0; i < ELT_LIMBS32; i++) {
            if (x.a[i] != y.a[i]) {
                return false;
            }
        }
        return true;
    }

    __device__
    static void
    set_zero(Fp &x) { fixnum::set_zero(x.a); }

    __device__
    static int
    is_zero(const Fp &x) { return fixnum::is_zero(x.a); }

    __device__
    static void
    set_one(Fp &x) { memcpy(x.a, modulus_info::monty_one(), ELT_BYTES); }

    __device__
    static void
    add(Fp &z, const Fp &x, const Fp &y) {
        int br;
        Fp r;
        auto mod = modulus_info::mod();
        fixnum::add(r.a, x.a, y.a);
        fixnum::sub_br(z.a, br, r.a, mod);
        if (br) {
            memcpy(z.a, r.a, ELT_BYTES);
        }
    }

    __device__
    static void
    neg(Fp &z, const Fp &x) {
        auto mod = modulus_info::mod();
        fixnum::sub(z.a, mod, x.a);
    }

    __device__
    static void
    sub(Fp &z, const Fp &x, const Fp &y) {
        int br;
        auto mod = modulus_info::mod();
        fixnum::sub_br(z.a, br, x.a, y.a);
        if (br) {
            fixnum::add(z.a, z.a, mod);
        }
    }

    __device__
    static void
    mul(Fp &z, const Fp &x, const Fp &y) {
        auto mod = modulus_info::mod();
        Fp r;
        fixnum::set_zero(r.a);
        uint32_t cy_hi = 0;
        for (int i = 0; i < ELT_LIMBS32; i++) {
            auto t0 = (uint64_t)x.a[0] * y.a[i] + r.a[0];
            auto lo = (uint32_t)t0;
            uint32_t cy = t0 >> 32;
            for (int j = 1; j < ELT_LIMBS32; j++) {
                auto t = (uint64_t)x.a[j] * y.a[i] + r.a[j] + cy;
                r.a[j - 1] = t;
                cy = t >> 32;
            }
            r.a[ELT_LIMBS32 - 1] = cy;
            auto m = lo * modulus_info::ninv_mod;
            cy = ((uint64_t)m * mod[0] + lo) >> 32;
            for (int j = 0; j < ELT_LIMBS32 - 1; j++) {
                auto t = (uint64_t)m * mod[j + 1] + r.a[j] + cy;
                r.a[j] = t;
                cy = t >> 32;
            }
            auto t = (uint64_t)r.a[ELT_LIMBS32 - 1] + cy + cy_hi;
            r.a[ELT_LIMBS32 - 1] = t;
            cy_hi = t >> 32;
        }
        assert(!cy_hi);
        int br;
        fixnum::sub_br(z.a, br, r.a, mod);
        if (br) {
            memcpy(z.a, r.a, ELT_BYTES);
        }
    }

    __device__
    static void
    sqr(Fp &z, const Fp &x) {
        // TODO: Find a faster way to do this. Actually only option
        // might be full squaring with REDC.
        mul(z, x, x);
    }

    __device__
    static void
    from_monty(Fp &y, const Fp &x) {
        Fp one;
        fixnum::set_one(one.a);
        mul(y, x, one);
    }
};



// Reference for multiplication and squaring methods below:
// https://pdfs.semanticscholar.org/3e01/de88d7428076b2547b60072088507d881bf1.pdf

template< typename Fp, int ALPHA >
struct Fp2 {
    typedef Fp PrimeField;

    // TODO: Use __builtin_align__(8) or whatever they use for the
    // builtin vector types.
    Fp a0, a1;

    static constexpr int DEGREE = 2;

    __device__
    static void
    load(Fp2 &x, const uint32_t *mem) {
        Fp::load(x.a0, mem);
        Fp::load(x.a1, mem + ELT_LIMBS32);
    }

    __device__
    static void
    store(uint32_t *mem, const Fp2 &x) {
        Fp::store(mem, x.a0);
        Fp::store(mem + ELT_LIMBS32, x.a1);
    }

    __device__
    static int
    are_equal(const Fp2 &x, const Fp2 &y) {
        return Fp::are_equal(x.a0, y.a0) && Fp::are_equal(x.a1, y.a1);
    }

    __device__
    static void
    set_zero(Fp2 &x) { Fp::set_zero(x.a0); Fp::set_zero(x.a1); }

    __device__
    static int
    is_zero(const Fp2 &x) { return Fp::is_zero(x.a0) && Fp::is_zero(x.a1); }

    __device__
    static void
    set_one(Fp2 &x) { Fp::set_one(x.a0); Fp::set_zero(x.a1); }

    __device__
    static void
    add(Fp2 &s, const Fp2 &x, const Fp2 &y) {
        Fp::add(s.a0, x.a0, y.a0);
        Fp::add(s.a1, x.a1, y.a1);
    }

    __device__
    static void
    sub(Fp2 &s, const Fp2 &x, const Fp2 &y) {
        Fp::sub(s.a0, x.a0, y.a0);
        Fp::sub(s.a1, x.a1, y.a1);
    }

    __device__
    static void
    mul(Fp2 &p, const Fp2 &a, const Fp2 &b) {
        Fp a0_b0, a1_b1, a0_plus_a1, b0_plus_b1, c, t0, t1;

        Fp::mul(a0_b0, a.a0, b.a0);
        Fp::mul(a1_b1, a.a1, b.a1);

        Fp::add(a0_plus_a1, a.a0, a.a1);
        Fp::add(b0_plus_b1, b.a0, b.a1);
        Fp::mul(c, a0_plus_a1, b0_plus_b1);

        mul_<ALPHA>::x(t0, a1_b1);
        Fp::sub(t1, c, a0_b0);

        Fp::add(p.a0, a0_b0, t0);
        Fp::sub(p.a1, t1, a1_b1);
    }


    __device__
    static void
    sqr(Fp2 &s, const Fp2 &a) {
        Fp a0_a1, a0_plus_a1, a0_plus_13_a1, t0, t1, t2;

        Fp::mul(a0_a1, a.a0, a.a1);
        Fp::add(a0_plus_a1, a.a0, a.a1);
        mul_<ALPHA>::x(t0, a.a1);
        Fp::add(a0_plus_13_a1, a.a0, t0);
        Fp::mul(t0, a0_plus_a1, a0_plus_13_a1);
        // TODO: Could do mul_14 to save a sub?
        Fp::sub(t1, t0, a0_a1);
        mul_<ALPHA>::x(t2, a0_a1);
        Fp::sub(s.a0, t1, t2);
        mul_<2>::x(s.a1, a0_a1);
    }
};


template< typename Fp, int ALPHA >
struct Fp3 {
    typedef Fp PrimeField;

    // TODO: Use __builtin_align__(8) or whatever they use for the
    // builtin vector types.
    Fp a0, a1, a2;

    static constexpr int DEGREE = 3;

    __device__
    static void
    load(Fp3 &x, const uint32_t *mem) {
        Fp::load(x.a0, mem);
        Fp::load(x.a1, mem + ELT_LIMBS32);
        Fp::load(x.a2, mem + 2*ELT_LIMBS32);
    }

    __device__
    static void
    store(uint32_t *mem, const Fp3 &x) {
        Fp::store(mem, x.a0);
        Fp::store(mem + ELT_LIMBS32, x.a1);
        Fp::store(mem + 2*ELT_LIMBS32, x.a2);
    }

    __device__
    static int
    are_equal(const Fp3 &x, const Fp3 &y) {
        return Fp::are_equal(x.a0, y.a0)
            && Fp::are_equal(x.a1, y.a1)
            && Fp::are_equal(x.a2, y.a2);
    }

    __device__
    static void
    set_zero(Fp3 &x) {
        Fp::set_zero(x.a0);
        Fp::set_zero(x.a1);
        Fp::set_zero(x.a2);
    }

    __device__
    static int
    is_zero(const Fp3 &x) {
        return Fp::is_zero(x.a0)
            && Fp::is_zero(x.a1)
            && Fp::is_zero(x.a2);
    }

    __device__
    static void
    set_one(Fp3 &x) {
        Fp::set_one(x.a0);
        Fp::set_zero(x.a1);
        Fp::set_zero(x.a2);
    }

    __device__
    static void
    add(Fp3 &s, const Fp3 &x, const Fp3 &y) {
        Fp::add(s.a0, x.a0, y.a0);
        Fp::add(s.a1, x.a1, y.a1);
        Fp::add(s.a2, x.a2, y.a2);
    }

    __device__
    static void
    sub(Fp3 &s, const Fp3 &x, const Fp3 &y) {
        Fp::sub(s.a0, x.a0, y.a0);
        Fp::sub(s.a1, x.a1, y.a1);
        Fp::sub(s.a2, x.a2, y.a2);
    }

    __device__
    static void
    mul(Fp3 &p, const Fp3 &a, const Fp3 &b) {
        Fp a0_b0, a1_b1, a2_b2;
        Fp a0_plus_a1, a1_plus_a2, a0_plus_a2, b0_plus_b1, b1_plus_b2, b0_plus_b2;
        Fp t0, t1, t2;

        Fp::mul(a0_b0, a.a0, b.a0);
        Fp::mul(a1_b1, a.a1, b.a1);
        Fp::mul(a2_b2, a.a2, b.a2);

        // TODO: Consider interspersing these additions among the
        // multiplications above.
        Fp::add(a0_plus_a1, a.a0, a.a1);
        Fp::add(a1_plus_a2, a.a1, a.a2);
        Fp::add(a0_plus_a2, a.a0, a.a2);

        Fp::add(b0_plus_b1, b.a0, b.a1);
        Fp::add(b1_plus_b2, b.a1, b.a2);
        Fp::add(b0_plus_b2, b.a0, b.a2);

        Fp::mul(t0, a1_plus_a2, b1_plus_b2);
        Fp::add(t1, a1_b1, a2_b2);
        Fp::sub(t0, t0, t1);
        mul_<ALPHA>::x(t0, t0);
        Fp::add(p.a0, a0_b0, t0);

        Fp::mul(t0, a0_plus_a1, b0_plus_b1);
        Fp::add(t1, a0_b0, a1_b1);
        mul_<ALPHA>::x(t2, a2_b2);
        Fp::sub(t2, t2, t1);
        Fp::add(p.a1, t0, t2);

        Fp::mul(t0, a0_plus_a2, b0_plus_b2);
        Fp::sub(t1, a1_b1, a0_b0);
        Fp::sub(t1, t1, a2_b2);
        Fp::add(p.a2, t0, t1);
    }

    __device__
    static void
    sqr(Fp3 &s, const Fp3 &a) {
        Fp a0a0, a1a1, a2a2;
        Fp a0_plus_a1, a1_plus_a2, a0_plus_a2;
        Fp t0, t1;

        Fp::sqr(a0a0, a.a0);
        Fp::sqr(a1a1, a.a1);
        Fp::sqr(a2a2, a.a2);

        // TODO: Consider interspersing these additions among the
        // squarings above.
        Fp::add(a0_plus_a1, a.a0, a.a1);
        Fp::add(a1_plus_a2, a.a1, a.a2);
        Fp::add(a0_plus_a2, a.a0, a.a2);

        Fp::sqr(t0, a1_plus_a2);
        // TODO: Remove sequential data dependencies (here and elsewhere)
        Fp::sub(t0, t0, a1a1);
        Fp::sub(t0, t0, a2a2);
        mul_<ALPHA>::x(t0, t0);
        Fp::add(s.a0, a0a0, t0);

        Fp::sqr(t0, a0_plus_a1);
        Fp::sub(t0, t0, a0a0);
        Fp::sub(t0, t0, a1a1);
        mul_<ALPHA>::x(t1, a2a2);
        Fp::add(s.a1, t0, t1);

        Fp::sqr(t0, a0_plus_a2);
        Fp::sub(t0, t0, a0a0);
        Fp::add(t0, t0, a1a1);
        Fp::sub(s.a2, t0, a2a2);
    }
};

typedef Fp<MNT4_MOD> Fp_MNT4;
typedef Fp2<Fp_MNT4, 13> Fp2_MNT4;

typedef Fp<MNT6_MOD> Fp_MNT6;
typedef Fp3<Fp_MNT6, 11> Fp3_MNT6;
