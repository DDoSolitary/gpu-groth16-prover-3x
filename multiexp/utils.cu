#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

void
print_meminfo(size_t allocated) {
    size_t free_mem, dev_mem;
    CubDebug(cudaMemGetInfo(&free_mem, &dev_mem));
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated,
            free_mem / 0x1p20,
            100.0 * free_mem / dev_mem);
}

template<typename T = var>
std::shared_ptr<T>
allocate_memory(size_t n, int dbg = 1) {
    T *mem = nullptr;
    CubDebug(cudaMalloc(&mem, n));
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg) {
        print_meminfo(n);
    }
    return std::shared_ptr<T>(mem, [](void *p) { CubDebug(cudaFree(p)); });
}

template<typename T = var>
std::shared_ptr<T>
allocate_host_memory(size_t n) {
    T *mem = nullptr;
    CubDebug(cudaHostAlloc(&mem, n, cudaHostAllocDefault));
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough host memory\n");
        abort();
    }
    return std::shared_ptr<T>(mem, [](void *p) { CubDebug(cudaFreeHost(p)); });
}

std::shared_ptr<var>
read_file_chunked(FILE *f, size_t n) {
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

std::shared_ptr<var>
load_scalars(size_t n, FILE *inputs)
{
    return read_file_chunked(inputs, n * ELT_BYTES);
}

template< typename EC >
std::shared_ptr<var>
load_points_affine(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    return read_file_chunked(inputs, n * FF::DEGREE * ELT_BYTES * 2);
}
