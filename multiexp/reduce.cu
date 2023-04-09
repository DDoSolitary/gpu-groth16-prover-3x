#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <climits>

#include "curves.cu"

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
ec_scalar_from_monty_kernel(var *scalars_, size_t N) {
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

static constexpr size_t threads_per_block = 256;
static constexpr size_t elts_per_block = threads_per_block / CUB_PTX_WARP_THREADS * ELTS_PER_WARP;

template<typename EC>
void
ec_scalar_from_monty(var *scalars, size_t N) {
    size_t nblocks = (N + elts_per_block - 1) / elts_per_block;
    ec_scalar_from_monty_kernel<typename EC::group_type><<<nblocks, threads_per_block>>>(scalars, N);
    CubDebug(cudaGetLastError());
}

template< typename EC, int C, int R >
void
ec_reduce_straus(cudaStream_t &strm, var *out, const var *multiples, const var *scalars, size_t N)
{
    CubDebug(cudaStreamCreate(&strm));

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

static inline double as_mebibytes(size_t n) {
    return n / (long double)(1UL << 20);
}

void print_meminfo(size_t allocated) {
    size_t free_mem, dev_mem;
    CubDebug(cudaMemGetInfo(&free_mem, &dev_mem));
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated,
            as_mebibytes(free_mem),
            100.0 * free_mem / dev_mem);
}

struct CudaFree {
    void operator()(var *mem) { CubDebug(cudaFree(mem)); }
};
typedef std::unique_ptr<var, CudaFree> var_ptr;

var_ptr
allocate_memory(size_t nbytes, int dbg = 0) {
    var *mem = nullptr;
    CubDebug(cudaMalloc(&mem, nbytes));
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg)
        print_meminfo(nbytes);
    return var_ptr(mem);
}

struct CudaFreeHost {
    void operator()(var *mem) { CubDebug(cudaFreeHost(mem)); }
};

std::unique_ptr<var, CudaFreeHost>
allocate_host_memory(size_t nbytes) {
    var *mem = nullptr;
    CubDebug(cudaHostAlloc(&mem, nbytes, cudaHostAllocDefault));
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough host memory\n");
        abort();
    }
    return std::unique_ptr<var, CudaFreeHost>(mem);
}

var_ptr read_file_chunked(FILE *f, size_t n) {
    auto dev_buf = allocate_memory(n);
    auto dev_ptr = (char *)dev_buf.get();
    auto bufsz = std::min(n, (size_t)INT32_MAX); // Iluvatar's cudaMemcpy fails when size is above this
    auto host_buf = allocate_host_memory(bufsz);
    auto host_ptr = (char *)host_buf.get();
    for (size_t off = 0; off < n; off += bufsz) {
        auto sz = std::min(n, off + bufsz) - off;
        if (fread(host_ptr, sz, 1, f) < 1) {
            fprintf(stderr, "Failed to read input\n");
            abort();
        }
        CubDebug(cudaMemcpy(dev_ptr + off, host_ptr, sz, cudaMemcpyHostToDevice));
    }
    return dev_buf;
}

var_ptr
load_scalars(size_t n, FILE *inputs)
{
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;

    return read_file_chunked(inputs, total_bytes);
}

template< typename EC >
var_ptr
load_points_affine(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;

    return read_file_chunked(inputs, total_aff_bytes);
}
