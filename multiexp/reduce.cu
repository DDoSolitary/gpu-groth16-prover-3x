#include <algorithm>
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
    static constexpr int NWIN = CUB_QUOTIENT_CEILING(L, WIN_BITS);
    static constexpr int WIN_MASK = (1 << WIN_BITS) - 1;
    static constexpr int BUCKETS_PER_WIN = 1 << WIN_BITS;

    static constexpr int BUCKET_SZ_LEN = BUCKETS_PER_WIN * NWIN;
    static constexpr int BUCKET_MAP_LEN = (BUCKETS_PER_WIN - 1) * NWIN;
    static constexpr int BUCKET_SZ_OFF = 0;
    static constexpr int BUCKET_MAP_OFF = BUCKET_SZ_OFF + BUCKET_SZ_LEN;
    static constexpr int BUCKET_IDX_OFF = BUCKET_MAP_OFF + BUCKET_MAP_LEN;

    static constexpr int MERGE_ITEMS_PER_WIN = (BUCKETS_PER_WIN - 1) * ELTS_PER_WARP;
    static constexpr int MERGE_LEN = MERGE_ITEMS_PER_WIN * NWIN;

    static constexpr int SORT_BITS = WIN_BITS + sizeof(int) * 8 - __builtin_clz(NWIN);
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
ec_multiexp_scan_idx(const var *scalars, uint32_t *out_keys, int *out_items, size_t n) {
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
ec_multiexp_scan_sz(const uint32_t *keys, size_t n, int *out) {
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
ec_multiexp_balance(const int *in_sz, int *out_sz, int *bucket_map, float target) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C::NWIN) {
        return;
    }
    auto cur_in_sz = in_sz + idx * C::BUCKETS_PER_WIN;
    auto cur_out_sz = out_sz + idx * C::BUCKETS_PER_WIN;
    auto cur_map = bucket_map + idx * (C::BUCKETS_PER_WIN - 1);
    int empty_cnt = 0;
    for (int i = 0; i < C::BUCKETS_PER_WIN - 1; i++) {
        if (cur_in_sz[i] == cur_in_sz[i + 1]) {
            empty_cnt++;
        }
    }
    cur_out_sz[0] = cur_in_sz[0];
    int cnt = 1;
    for (int i = 0; i < C::BUCKETS_PER_WIN - 1; i++) {
        cur_map[i] = cnt - 1;
        int d = cur_in_sz[i + 1] - cur_in_sz[i];
        if (d == 0) {
            continue;
        }
        int k = lroundf(d / target);
        k = CUB_MIN(k, empty_cnt + 1);
        if (k <= 1) {
            cur_out_sz[cnt++] = cur_in_sz[i + 1];
            continue;
        }
        int off = 0;
        for (int j = 0; j < k; j++) {
            off += d / k + (d % k > j);
            cur_out_sz[cnt + j] = cur_in_sz[i] + off;
        }
        cnt += k;
        empty_cnt -= k - 1;
    }
    for (; cnt < C::BUCKETS_PER_WIN; cnt++) {
        cur_out_sz[cnt] = cur_out_sz[cnt - 1];
    }
}

template<typename C>
void ec_multiexp_scan_mem_size(size_t n, size_t *temp_size, size_t *out_size) {
    size_t sort_temp_size;
    size_t sort_len = n * C::NWIN;
    size_t key_size = sort_len * sizeof(uint32_t);
    size_t value_size = sort_len * sizeof(int);
    CubDebug((cub::DeviceRadixSort::SortPairs<uint32_t, int>(nullptr, sort_temp_size, nullptr, nullptr, nullptr, (int *)nullptr, sort_len, 0, C::SORT_BITS)));
    *temp_size = key_size * 2 + std::max(value_size + sort_temp_size, C::BUCKET_SZ_LEN * sizeof(int)); // keys + max(cub sort temp, output of scan_sz)
    *out_size = C::BUCKET_SZ_LEN * sizeof(int) + C::BUCKET_MAP_LEN * sizeof(int) + value_size; // sz + map + idx
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
    auto keys1 = (uint32_t *)temp;
    auto keys2 = keys1 + idx_size;
    auto out_temp = (int *)(keys2 + idx_size);
    auto sort_temp = (void *)(out_temp + idx_size);
    size_t sort_temp_size = (char *)temp + temp_size - (char *)sort_temp;

    // convert from montegomery form
    int nblocks = CUB_QUOTIENT_CEILING(n, C::ELTS_PER_BLOCK);
    ec_scalar_from_monty<Fr><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(scalars, n);
    CubDebug(cudaGetLastError());

    // extract bucket id
    nblocks = CUB_QUOTIENT_CEILING(idx_size, C::THREADS_PER_BLOCK);
    ec_multiexp_scan_idx<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(scalars, keys1, out_temp, n);
    CubDebug(cudaGetLastError());

    // sort so that items with same window & id are grouped together
    CubDebug(cub::DeviceRadixSort::SortPairs(sort_temp, sort_temp_size, keys1, keys2, out_temp, out_idx, idx_size, 0, C::SORT_BITS, stream));

    // find bucket sizes
    nblocks = CUB_QUOTIENT_CEILING((n + 1) * C::NWIN, C::THREADS_PER_BLOCK);
    ec_multiexp_scan_sz<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(keys2, n, out_temp);
    CubDebug(cudaGetLastError());

    // attempt to balance bucket sizes by splitting large buckets into other empty ones (mainly for the last window)
    nblocks = CUB_QUOTIENT_CEILING(C::NWIN, C::THREADS_PER_BLOCK);
    ec_multiexp_balance<C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(out_temp, out_sz, out_map, (float)n / C::BUCKETS_PER_WIN);
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

// per window parallel prefix sum
template<typename EC, typename C>
__global__ void
ec_multiexp_prefix_sum(const var *in, var *out, int step) {
    int idx = get_idx();
    int win = idx / C::MERGE_ITEMS_PER_WIN;
    int i = idx % C::MERGE_ITEMS_PER_WIN;
    if (win >= C::NWIN) {
        return;
    }
    EC x, y;
    EC::load_jac(x, in + idx * EC::NLIMBS);
    if (i + step < C::MERGE_ITEMS_PER_WIN) {
        EC::load_jac(y, in + (idx + step) * EC::NLIMBS);
        EC::add(x, x, y);
    }
    EC::store_jac(out + idx * EC::NLIMBS, x);
}

// first iteration of per window parallel sum, also handles load-balanced buckets
template<typename EC, typename C>
__global__ void
ec_multiexp_bucket_reduce_first(const var *in, const int *bucket_map, var *out) {
    int idx = get_idx();
    int n = C::BUCKETS_PER_WIN / 2;
    int win = idx / n;
    int i = idx % n;
    if (win >= C::NWIN) {
        return;
    }
    auto cur_map = bucket_map + win * (C::BUCKETS_PER_WIN - 1);
    auto cur_in = in + win * C::MERGE_ITEMS_PER_WIN * EC::NLIMBS;
    EC x, y;
    int j = cur_map[i];
    if (j < C::BUCKETS_PER_WIN - 1) {
        EC::load_jac(x, cur_in + j * ELTS_PER_WARP * EC::NLIMBS);
    } else {
        EC::set_zero(x);
    }
    if (i + n < C::BUCKETS_PER_WIN - 1) {
        int k = cur_map[i + n];
        if (k < C::BUCKETS_PER_WIN - 1) {
            EC::load_jac(y, cur_in + k * ELTS_PER_WARP * EC::NLIMBS);
            EC::add(x, x, y);
        }
    }
    EC::store_jac(out + idx * EC::NLIMBS, x);
}

// following iterations of per window parallel sum
template<typename EC, typename C>
__global__ void
ec_multiexp_bucket_reduce_next(const var *in, var *out, int m) {
    // input size is guaranteed to be power of 2 after the first step
    int n = m / 2;
    int idx = get_idx();
    int win = idx / n;
    int i = idx % n;
    if (win >= C::NWIN) {
        return;
    }
    auto cur_in = in + win * m * EC::NLIMBS;
    EC x, y;
    EC::load_jac(x, cur_in + i * EC::NLIMBS);
    EC::load_jac(y, cur_in + (i + n) * EC::NLIMBS);
    EC::add(x, x, y);
    EC::store_jac(out + idx * EC::NLIMBS, x);
}

// simple parallel sum
template<typename EC, typename C>
__global__ void
ec_multiexp_window_reduce(const var *in, var *out, int m)
{
    int n = (m + 1) / 2;
    int idx = get_idx();
    if (idx >= n) {
        return;
    }

    EC x, y;
    EC::load_jac(x, in + idx * EC::NLIMBS);
    if (idx + n < m) {
        EC::load_jac(y, in + (idx + n) * EC::NLIMBS);
        for (int i = 0; i < n * C::WIN_BITS; i++) {
            EC::dbl(y, y);
        }
        EC::add(x, x, y);
    }
    EC::store_jac(out + idx * EC::NLIMBS, x);
}

template<typename EC, typename C>
void ec_multiexp_pippenger_mem_size(size_t *temp_size, size_t *out_size) {
    *temp_size = C::MERGE_LEN * EC::NLIMBS * sizeof(var) * 2; // output of point merge + temp for non-in-place algorithms
    *out_size = EC::NLIMBS * sizeof(var);
}

template<typename EC, typename C>
void
ec_multiexp_pippenger(const var *pts, const int *bucket_info, var *out, void *temp, size_t off, size_t n, cudaStream_t stream) {
    auto temp1 = (var *)temp;
    auto temp2 = temp1 + C::MERGE_LEN * EC::NLIMBS;

    // each warp handles one (possibly split) bucket
    int nblocks = CUB_QUOTIENT_CEILING((C::BUCKETS_PER_WIN - 1) * C::NWIN, C::WARPS_PER_BLOCK);
    ec_multiexp_point_merge<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(pts, bucket_info, temp1, off, n);
    CubDebug(cudaGetLastError());

    // calculate prefix sum in each window
    // load balanced buckets will also be added together
    for (int i = 1; i < C::MERGE_ITEMS_PER_WIN; i *= 2) {
        nblocks = CUB_QUOTIENT_CEILING(C::MERGE_LEN, C::ELTS_PER_BLOCK);
        ec_multiexp_prefix_sum<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(temp1, temp2, i);
        std::swap(temp1, temp2);
    }
    CubDebug(cudaGetLastError());

    auto bucket_map = bucket_info + C::BUCKET_MAP_OFF;
    nblocks = CUB_QUOTIENT_CEILING(C::BUCKETS_PER_WIN / 2 * C::NWIN, C::ELTS_PER_BLOCK);
    ec_multiexp_bucket_reduce_first<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(temp1, bucket_map, temp2);
    for (int i = C::BUCKETS_PER_WIN / 2; i > 1; i /= 2) {
        nblocks = CUB_QUOTIENT_CEILING(i / 2 * C::NWIN, C::ELTS_PER_BLOCK);
        ec_multiexp_bucket_reduce_next<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(temp2, temp1, i);
        std::swap(temp1, temp2);
    }
    CubDebug(cudaGetLastError());

    for (int i = C::NWIN; i > 1; i = (i + 1) / 2) {
        nblocks = CUB_QUOTIENT_CEILING((i + 1) / 2, C::ELTS_PER_BLOCK);
        auto cur_out = i == 2 ? out : temp2; // write to output buffer on the last iteration
        ec_multiexp_window_reduce<EC, C><<<nblocks, C::THREADS_PER_BLOCK, 0, stream>>>(temp2, cur_out, i);
    }
    CubDebug(cudaGetLastError());
}
