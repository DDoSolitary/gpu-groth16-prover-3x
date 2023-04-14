#include <cstdint>
#include <climits>
#include <utility>

#include "curves.cu"
#include "utils.cu"

static constexpr int threads_per_block = 256;
static_assert(threads_per_block % CUB_PTX_WARP_THREADS == 0, "block size must be multiple of warp size");
static constexpr int warps_per_block = threads_per_block / CUB_PTX_WARP_THREADS;
static constexpr int elts_per_block = warps_per_block * ELTS_PER_WARP;

__device__ int
get_idx() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = idx >> CUB_PTX_LOG_WARP_THREADS, lane_id = idx & (CUB_PTX_WARP_THREADS - 1);
    int lane_elt_idx = lane_id / ELT_LIMBS;
    if (lane_elt_idx >= ELTS_PER_WARP) {
        return INT_MAX;
    }
    return warp_id * ELTS_PER_WARP + lane_elt_idx;
}

template<typename T>
struct dev_cmp {
    __device__ bool operator()(T x, T y) {
        return x < y;
    }
};

template<typename Fr>
__global__ void
ec_scalar_from_monty(var *scalars_, size_t N) {
    int idx = get_idx();
    if (idx >= N) {
        return;
    }
    var *p = scalars_ + idx * ELT_LIMBS;
    Fr x;
    Fr::load(x, p);
    Fr::from_monty(x, x);
    Fr::store(p, x);
}

// extract bucket id for every window of every coefficient
template<int C>
__global__ void
ec_multiexp_scan_idx(const var *scalars, int *out_keys, int *out_items, size_t n) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr var C_MASK = ((var)1 << C) - (var)1;
    // we assume n * NWIN < INT_MAX
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx / NWIN;
    int win_id = idx % NWIN;
    if (i >= n) {
        return;
    }
    const var *scalar = scalars + i * ELT_LIMBS;
    int win_off = win_id * C;
    int j = win_off / digit::BITS, k = win_off % digit::BITS;
    int bucket = (scalar[j] >> k) & C_MASK;
    if (digit::BITS - k < C && j < ELT_LIMBS - 1) {
        bucket |= (scalar[j + 1] << (digit::BITS - k)) & C_MASK;
    }

    out_keys[idx] = (win_id << C) | bucket;
    out_items[idx] = i;
}

// find sizes of the buckets by differentiation
template<int C>
__global__ void
ec_multiexp_scan_sz(const int *keys, size_t n, int *out) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int win = idx / (n + 1);
    int i = idx % (n + 1);
    if (win >= NWIN) {
        return;
    }
    int lb = i == 0 ? 0 : keys[i - 1 + win * n] & C_MASK;
    int rb = i == n ? 1 << C : keys[i + win * n] & C_MASK;

    // use loop to handle empty buckets
    for (int j = lb; j < rb; j++) {
        out[(win << C) | j] = i + win * n;
    }
}

template<int C>
__global__ void
ec_multiexp_balance(int *bucket_sz, int *bucket_map, float target) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= NWIN) {
        return;
    }
    auto cur_sz = bucket_sz + idx * (1 << C);
    auto cur_map = bucket_map + idx * ((1 << C) - 1);
    int empty_cnt = 0;
    for (int i = 0; i < C_MASK; i++) {
        if (cur_sz[i] == cur_sz[i + 1]) {
            empty_cnt++;
        }
    }
    int buf[1 << C] = { cur_sz[0] }, cnt = 1;
    for (int i = 0; i < C_MASK; i++) {
        cur_map[i] = cnt - 1;
        int d = cur_sz[i + 1] - cur_sz[i];
        if (d == 0) {
            continue;
        }
        int k = lroundf(d / target);
        k = CUB_MIN(k, empty_cnt + 1);
        if (k <= 1) {
            buf[cnt++] = cur_sz[i + 1];
            continue;
        }
        int off = 0;
        for (int j = 0; j < k; j++) {
            off += d / k + (d % k > j);
            buf[cnt + j] = cur_sz[i] + off;
        }
        cnt += k;
        empty_cnt -= k - 1;
    }
    for (; cnt < (1 << C); cnt++) {
        buf[cnt] = buf[cnt - 1];
    }
    memcpy(cur_sz, buf, (1 << C) * sizeof(int));
}

template<int C>
void ec_multiexp_scan_mem_size(size_t n, size_t *temp_size, size_t *out_size) {
    static constexpr int NWIN = (753 + C - 1) / C;
    size_t sort_size;
    CubDebug(cub::DeviceMergeSort::SortPairs(nullptr, sort_size, (int *)nullptr, (int *)nullptr, n * NWIN, dev_cmp<int>()));
    *temp_size = n * NWIN * sizeof(int) + sort_size; // keys + cub temp
    *out_size = (1 << C) * NWIN * sizeof(int) + ((1 << C) - 1) * NWIN * sizeof(int) + n * NWIN * sizeof(int); // sz + map + idx
}

// put coefficient indices into buckets
// input is modified for conversion from montegomery form
template<typename Fr, int C>
void
ec_multiexp_scan(var *scalars, int *out, size_t n, void *temp, size_t temp_size, cudaStream_t stream) {
    static constexpr int NWIN = (753 + C - 1) / C;
    auto idx_size = n * NWIN;
    auto out_sz = out;
    auto out_map = out + (1 << C) * NWIN;
    auto out_idx = out + (1 << C) * NWIN + ((1 << C) - 1) * NWIN;
    auto keys = (int *)temp;
    auto sort_temp = (void *)(keys + idx_size);
    auto sort_temp_size = temp_size - idx_size * sizeof(int);

    // convert from montegomery form
    int nblocks = (n + elts_per_block - 1) / elts_per_block;
    ec_scalar_from_monty<Fr><<<nblocks, threads_per_block, 0, stream>>>(scalars, n);
    CubDebug(cudaGetLastError());

    // extract bucket id
    nblocks = (idx_size + threads_per_block - 1) / threads_per_block;
    ec_multiexp_scan_idx<C><<<nblocks, threads_per_block, 0, stream>>>(scalars, keys, out_idx, n);
    CubDebug(cudaGetLastError());

    // sort so that items with same window & id are grouped together
    CubDebug(cub::DeviceMergeSort::SortPairs(sort_temp, sort_temp_size, keys, out_idx, idx_size, dev_cmp<int>(), stream));

    // find bucket sizes
    nblocks = ((n + 1) * NWIN + threads_per_block - 1) / threads_per_block;
    ec_multiexp_scan_sz<C><<<nblocks, threads_per_block, 0, stream>>>(keys, n, out_sz);
    CubDebug(cudaGetLastError());

    // attempt to balance bucket sizes (mainly for last window)
    nblocks = (NWIN + threads_per_block - 1) / threads_per_block;
    ec_multiexp_balance<C><<<nblocks, threads_per_block, 0, stream>>>(out_sz, out_map, (float)n / (1 << C));
    CubDebug(cudaGetLastError());
}

// TODO: use whole warp for a bucket?
// calculate sum in each bucket
template<typename EC, int C>
__global__ void
ec_multiexp_point_merge(const var *pts, const int *bucket_info, var *out, size_t off, size_t n) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
    static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;

    int idx = get_idx();
    int group = idx % ELTS_PER_WARP;
    int warp = idx / ELTS_PER_WARP;
    int win = warp / C_MASK;
    int bucket = warp % C_MASK;
    if (win >= NWIN) {
        return;
    }
    const int *bucket_sz = bucket_info, *bucket_idx = bucket_info + (1 << C) * NWIN + ((1 << C) - 1) * NWIN;
    EC x, m;
    EC::set_zero(x);
    // bucket 0 is skipped
    int l = bucket_sz[bucket + win * (1 << C)], r = bucket_sz[bucket + 1 + win * (1 << C)];
    for (int j = l + group; j < r; j += ELTS_PER_WARP) {
        int k = bucket_idx[j] - off;
        if (k >= 0 && k < n) {
            EC::load_affine(m, pts + k * AFF_POINT_LIMBS);
            EC::mixed_add(x, x, m);
        }
    }
    EC::store_jac(out + idx * JAC_POINT_LIMBS, x);
}

// reduce all buckets in a window
template<typename EC, int C>
__global__ void
ec_multiexp_bucket_reduce(var *buckets, const int *bucket_map, var *out) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;

    int idx = get_idx();
    if (idx >= NWIN) {
        return;
    }

    auto cur_buckets = buckets + idx * C_MASK * ELTS_PER_WARP * JAC_POINT_LIMBS;
    auto cur_map = bucket_map + idx * ((1 << C) - 1);

    EC x, y;
    for (int i = C_MASK * ELTS_PER_WARP - 2; i >= 0; i--) {
        EC::load_jac(x, cur_buckets + i * JAC_POINT_LIMBS);
        EC::load_jac(y, cur_buckets + (i + 1) * JAC_POINT_LIMBS);
        EC::add(x, x, y);
        EC::store_jac(cur_buckets + i * JAC_POINT_LIMBS, x);
    }

    EC::set_zero(x);
    for (int i = C_MASK - 1; i >= 0; i--) {
        int k = cur_map[i];
        if (k < C_MASK) {
            EC::load_jac(y, cur_buckets + k * JAC_POINT_LIMBS * ELTS_PER_WARP);
            EC::add(x, x, y);
        }
    }

    // TODO: hard to parallelize, consider preprocessing
    for (int i = 0; i < idx * C; i++) {
        EC::dbl(x, x);
    }

    EC::store_jac(out + idx * JAC_POINT_LIMBS, x);
}

template< typename EC >
__global__ void
ec_sum_all(var *X, const var *Y, size_t n)
{
    int idx = get_idx();

    if (idx < n) {
        EC z, x, y;
        int off = idx * EC::NELTS * ELT_LIMBS;

        EC::load_jac(x, X + off);
        EC::load_jac(y, Y + off);

        EC::add(z, x, y);

        EC::store_jac(X + off, z);
    }
}

template<typename EC, int C>
void ec_multiexp_pippenger_mem_size(size_t *temp_size, size_t *out_size) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    *temp_size = C_MASK * NWIN * ELTS_PER_WARP * EC::NELTS * ELT_BYTES; // output of point merge
    *out_size = NWIN * EC::NELTS * ELT_BYTES; // avoid extra copy after inplace sum
}

template<typename EC, int C>
void
ec_multiexp_pippenger(const var *pts, const int *bucket_info, var *out, void *temp, size_t off, size_t n, cudaStream_t stream) {
    static constexpr int NWIN = (753 + C - 1) / C;
    static constexpr int C_MASK = (1 << C) - 1;
    
    int nblocks = (C_MASK * NWIN + warps_per_block - 1) / warps_per_block;
    ec_multiexp_point_merge<EC, C><<<nblocks, threads_per_block, 0, stream>>>(pts, bucket_info, (var *)temp, off, n);
    CubDebug(cudaGetLastError());

    auto bucket_map = bucket_info + (1 << C) * NWIN;
    nblocks = (NWIN + elts_per_block - 1) / elts_per_block;
    ec_multiexp_bucket_reduce<EC, C><<<nblocks, threads_per_block, 0, stream>>>((var *)temp, bucket_map, out);
    CubDebug(cudaGetLastError());

    for (size_t r = NWIN & 1, m = NWIN / 2; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m + elts_per_block - 1) / elts_per_block;
        ec_sum_all<EC><<<nblocks, threads_per_block, 0, stream>>>(out, out + m * EC::NELTS * ELT_LIMBS, m);
        if (r) {
            ec_sum_all<EC><<<1, threads_per_block, 0, stream>>>(out, out + 2 * m * EC::NELTS * ELT_LIMBS, 1);
        }
    }
    CubDebug(cudaGetLastError());
}

// C is the size of the precomputation
// R is the number of points we're handling per thread
template< typename EC, int C = 4, int RR = 8 >
__global__ void
ec_multiexp_straus(var *out, const var *multiples_, const var *scalars_, size_t N)
{
    int idx = get_idx();

    size_t n = (N + RR - 1) / RR;
    if (idx < n) {
        // TODO: Treat remainder separately so R can remain a compile time constant
        size_t R = (idx < n - 1) ? RR : (N % RR);

        typedef typename EC::group_type Fr;
        static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
        static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int out_off = idx * JAC_POINT_LIMBS;
        int m_off = idx * RR * AFF_POINT_LIMBS;
        int s_off = idx * RR * ELT_LIMBS;

        const var *multiples = multiples_ + m_off;
        // TODO: Consider loading multiples and/or scalars into shared memory

        // i is smallest multiple of C such that i > 753
        int i = C * ((753 + C - 1) / C); // C * ceiling(753/C)
        assert((i - C * 753) < C);
        static constexpr var C_MASK = (1U << C) - 1U;

        EC x;
        EC::set_zero(x);
        while (i >= C) {
            EC::template mul_2exp<C>(x, x);
            i -= C;

            int q = i / digit::BITS, r = i % digit::BITS;
            for (int j = 0; j < R; ++j) {
                auto scalar = scalars_ + s_off + j * ELT_LIMBS;
                var s = scalar[q];
                var win = (s >> r) & C_MASK;
                // Handle case where C doesn't divide digit::BITS
                int bottom_bits = digit::BITS - r;
                // detect when window overlaps digit boundary
                if (bottom_bits < C) {
                    s = scalar[q + 1];
                    win |= (s << bottom_bits) & C_MASK;
                }
                if (win > 0) {
                    EC m;
                    //EC::add(x, x, multiples[win - 1][j]);
                    EC::load_affine(m, multiples + ((win-1)*N + j)*AFF_POINT_LIMBS);
                    EC::mixed_add(x, x, m);
                }
            }
        }
        EC::store_jac(out + out_off, x);
    }
}

template< typename EC, int C, int R >
void
ec_reduce_straus(cudaStream_t &strm, var *out, const var *multiples, const var *scalars, size_t N)
{
    //CubDebug(cudaStreamCreate(&strm));

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    size_t n = (N + R - 1) / R;

    size_t nblocks = (n + elts_per_block - 1) / elts_per_block;

    ec_multiexp_straus<EC, C, R><<< nblocks, threads_per_block, 0, strm>>>(out, multiples, scalars, N);
    CubDebug(cudaGetLastError());

    size_t r = n & 1, m = n / 2;
    for ( ; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m + elts_per_block - 1) / elts_per_block;

        ec_sum_all<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m*pt_limbs, m);
        if (r)
            ec_sum_all<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2*m*pt_limbs, 1);
    }
}
