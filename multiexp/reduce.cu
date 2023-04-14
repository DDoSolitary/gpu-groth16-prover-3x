#include <cstdint>
#include <climits>
#include <utility>

#include "curves.cu"
#include "utils.cu"

template<int C, int L = 753, int B = 256>
struct ec_multiexp_config {
    static constexpr int THREADS_PER_BLOCK = B;
    static_assert(THREADS_PER_BLOCK % CUB_PTX_WARP_THREADS == 0, "block size must be multiple of warp size");
    static constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / CUB_PTX_WARP_THREADS;
    static constexpr int ELTS_PER_BLOCK = WARPS_PER_BLOCK * ELTS_PER_WARP;

    static constexpr int WIN_BITS = C;
    static constexpr int NWIN = (L + WIN_BITS - 1) / WIN_BITS;
    static constexpr int WIN_MASK = (1 << WIN_BITS) - 1;
    static constexpr int BUCKETS_PER_WIN = 1 << WIN_BITS;

    static constexpr int BUCKET_SZ_LEN = BUCKETS_PER_WIN * NWIN;
    static constexpr int BUCKET_MAP_LEN = (BUCKETS_PER_WIN - 1) * NWIN;
    static constexpr int BUCKET_SZ_OFF = 0;
    static constexpr int BUCKET_MAP_OFF = BUCKET_SZ_OFF + BUCKET_SZ_LEN;
    static constexpr int BUCKET_IDX_OFF = BUCKET_MAP_OFF + BUCKET_MAP_LEN;
};

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
template<typename C>
__global__ void
ec_multiexp_scan_idx(const var *scalars, int *out_keys, int *out_items, size_t n) {
    // we assume n * NWIN < INT_MAX
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx / C::NWIN;
    int win_id = idx % C::NWIN;
    if (i >= n) {
        return;
    }
    const var *scalar = scalars + i * ELT_LIMBS;
    int win_off = win_id * C::WIN_BITS;
    int j = win_off / digit::BITS, k = win_off % digit::BITS;
    int bucket = (scalar[j] >> k) & C::WIN_MASK;
    if (digit::BITS - k < C::WIN_BITS && j < ELT_LIMBS - 1) {
        bucket |= (scalar[j + 1] << (digit::BITS - k)) & C::WIN_MASK;
    }

    out_keys[idx] = (win_id << C::WIN_BITS) | bucket;
    out_items[idx] = i;
}

// find sizes of the buckets by differentiation
template<typename C>
__global__ void
ec_multiexp_scan_sz(const int *keys, size_t n, int *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int win = idx / (n + 1);
    int i = idx % (n + 1);
    if (win >= C::NWIN) {
        return;
    }
    int lb = i == 0 ? 0 : keys[i - 1 + win * n] & C::WIN_MASK;
    int rb = i == n ? C::BUCKETS_PER_WIN : keys[i + win * n] & C::WIN_MASK;

    // use loop to handle empty buckets
    for (int j = lb; j < rb; j++) {
        out[(win << C::WIN_BITS) | j] = i + win * n;
    }
}

template<typename C>
__global__ void
ec_multiexp_balance(int *bucket_sz, int *bucket_map, float target) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C::NWIN) {
        return;
    }
    auto cur_sz = bucket_sz + idx * C::BUCKETS_PER_WIN;
    auto cur_map = bucket_map + idx * (C::BUCKETS_PER_WIN - 1);
    int empty_cnt = 0;
    for (int i = 0; i < C::BUCKETS_PER_WIN - 1; i++) {
        if (cur_sz[i] == cur_sz[i + 1]) {
            empty_cnt++;
        }
    }
    int buf[C::BUCKETS_PER_WIN] = { cur_sz[0] }, cnt = 1;
    for (int i = 0; i < C::BUCKETS_PER_WIN - 1; i++) {
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
    for (; cnt < C::BUCKETS_PER_WIN; cnt++) {
        buf[cnt] = buf[cnt - 1];
    }
    memcpy(cur_sz, buf, C::BUCKETS_PER_WIN * sizeof(int));
}

template<typename C>
void ec_multiexp_scan_mem_size(size_t n, size_t *temp_size, size_t *out_size) {
    size_t sort_size;
    CubDebug(cub::DeviceMergeSort::SortPairs(nullptr, sort_size, (int *)nullptr, (int *)nullptr, n * C::NWIN, dev_cmp<int>()));
    *temp_size = n * C::NWIN * sizeof(int) + sort_size; // keys + cub temp
    *out_size = (C::BUCKET_SZ_LEN + C::BUCKET_MAP_LEN + n * C::NWIN) * sizeof(int); // sz + map + idx
}

// put coefficient indices into buckets
// input is modified for conversion from montegomery form
template<typename Fr, typename C>
void
ec_multiexp_scan(var *scalars, int *out, size_t n, void *temp, size_t temp_size, cudaStream_t stream) {
    auto idx_size = n * C::NWIN;
    auto out_sz = out + C::BUCKET_SZ_OFF;
    auto out_map = out + C::BUCKET_MAP_OFF;
    auto out_idx = out + C::BUCKET_IDX_OFF;
    auto keys = (int *)temp;
    auto sort_temp = (void *)(keys + idx_size);
    auto sort_temp_size = temp_size - idx_size * sizeof(int);

    // convert from montegomery form
    int nblocks = (n + C::ELTS_PER_BLOCK - 1) / C::ELTS_PER_BLOCK;
    ec_scalar_from_monty<Fr><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(scalars, n);
    CubDebug(cudaGetLastError());

    // extract bucket id
    nblocks = (idx_size + C::THREADS_PER_BLOCK - 1) / C::THREADS_PER_BLOCK;
    ec_multiexp_scan_idx<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(scalars, keys, out_idx, n);
    CubDebug(cudaGetLastError());

    // sort so that items with same window & id are grouped together
    CubDebug(cub::DeviceMergeSort::SortPairs(sort_temp, sort_temp_size, keys, out_idx, idx_size, dev_cmp<int>(), stream));

    // find bucket sizes
    nblocks = ((n + 1) * C::NWIN + C::THREADS_PER_BLOCK - 1) / C::THREADS_PER_BLOCK;
    ec_multiexp_scan_sz<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(keys, n, out_sz);
    CubDebug(cudaGetLastError());

    // attempt to balance bucket sizes (mainly for last window)
    nblocks = (C::NWIN + C::THREADS_PER_BLOCK - 1) / C::THREADS_PER_BLOCK;
    ec_multiexp_balance<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(out_sz, out_map, (float)n / C::BUCKETS_PER_WIN);
    CubDebug(cudaGetLastError());
}

// TODO: use whole warp for a bucket?
// calculate sum in each bucket
template<typename EC, typename C>
__global__ void
ec_multiexp_point_merge(const var *pts, const int *bucket_info, var *out, size_t off, size_t n) {
    int idx = get_idx();
    int group = idx % ELTS_PER_WARP;
    int warp = idx / ELTS_PER_WARP;
    int win = warp / (C::BUCKETS_PER_WIN - 1);
    int bucket = warp % (C::BUCKETS_PER_WIN - 1);
    if (win >= C::NWIN) {
        return;
    }
    const int *bucket_sz = bucket_info + C::BUCKET_SZ_OFF, *bucket_idx = bucket_info + C::BUCKET_IDX_OFF;
    EC x, m;
    EC::set_zero(x);
    // bucket 0 is skipped
    int l = bucket_sz[bucket + win * C::BUCKETS_PER_WIN], r = bucket_sz[bucket + 1 + win * C::BUCKETS_PER_WIN];
    for (int j = l + group; j < r; j += ELTS_PER_WARP) {
        int k = bucket_idx[j] - off;
        if (k >= 0 && k < n) {
            EC::load_affine(m, pts + k * EC::NLIMBS_AFF);
            EC::mixed_add(x, x, m);
        }
    }
    EC::store_jac(out + idx * EC::NLIMBS, x);
}

// reduce all buckets in a window
template<typename EC, typename C>
__global__ void
ec_multiexp_bucket_reduce(var *buckets, const int *bucket_map, var *out) {
    int idx = get_idx();
    if (idx >= C::NWIN) {
        return;
    }

    auto cur_buckets = buckets + idx * (C::BUCKETS_PER_WIN - 1) * ELTS_PER_WARP * EC::NLIMBS;
    auto cur_map = bucket_map + idx * (C::BUCKETS_PER_WIN - 1);

    EC x, y;
    for (int i = (C::BUCKETS_PER_WIN - 1) * ELTS_PER_WARP - 2; i >= 0; i--) {
        EC::load_jac(x, cur_buckets + i * EC::NLIMBS);
        EC::load_jac(y, cur_buckets + (i + 1) * EC::NLIMBS);
        EC::add(x, x, y);
        EC::store_jac(cur_buckets + i * EC::NLIMBS, x);
    }

    EC::set_zero(x);
    for (int i = C::BUCKETS_PER_WIN - 2; i >= 0; i--) {
        int k = cur_map[i];
        if (k < C::BUCKETS_PER_WIN - 1) {
            EC::load_jac(y, cur_buckets + k * ELTS_PER_WARP * EC::NLIMBS);
            EC::add(x, x, y);
        }
    }

    // TODO: hard to parallelize, consider preprocessing
    for (int i = 0; i < idx * C::WIN_BITS; i++) {
        EC::dbl(x, x);
    }

    EC::store_jac(out + idx * EC::NLIMBS, x);
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

template<typename EC, typename C>
void ec_multiexp_pippenger_mem_size(size_t *temp_size, size_t *out_size) {
    *temp_size = (C::BUCKETS_PER_WIN - 1) * C::NWIN * ELTS_PER_WARP * EC::NLIMBS * sizeof(var); // output of point merge
    *out_size = C::NWIN * EC::NLIMBS * sizeof(var); // avoid extra copy after inplace sum
}

template<typename EC, typename C>
void
ec_multiexp_pippenger(const var *pts, const int *bucket_info, var *out, void *temp, size_t off, size_t n, cudaStream_t stream) {
    int nblocks = ((C::BUCKETS_PER_WIN - 1) * C::NWIN + C::WARPS_PER_BLOCK - 1) / C::WARPS_PER_BLOCK;
    ec_multiexp_point_merge<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(pts, bucket_info, (var *)temp, off, n);
    CubDebug(cudaGetLastError());

    auto bucket_map = bucket_info + C::BUCKET_MAP_OFF;
    nblocks = (C::NWIN + C::ELTS_PER_BLOCK - 1) / C::ELTS_PER_BLOCK;
    ec_multiexp_bucket_reduce<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>((var *)temp, bucket_map, out);
    CubDebug(cudaGetLastError());

    for (size_t r = C::NWIN & 1, m = C::NWIN / 2; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m + C::ELTS_PER_BLOCK - 1) / C::ELTS_PER_BLOCK;
        ec_sum_all<EC><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(out, out + m * EC::NLIMBS, m);
        if (r) {
            ec_sum_all<EC><<<1, C::THREADS_PER_BLOCK, 0, stream>>>(out, out + 2 * m * EC::NLIMBS, 1);
        }
    }
    CubDebug(cudaGetLastError());
}
