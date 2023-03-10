#pragma once

#include <iostream>
#include <cstdint>

namespace cuFIXNUM {

// TODO: Copy over functionality and documentation from IntmodVector.
template< typename fixnum >
class fixnum_array {
public:
    typedef std::uint8_t byte;

    static fixnum_array *create(size_t nelts);
    template< typename T >
    static fixnum_array *create(size_t nelts, T init);
    // NB: If bytes_per_elt doesn't divide len, the last len % bytes_per_elt
    // bytes are *dropped*.
    static fixnum_array *create(const byte *data, size_t total_bytes, size_t bytes_per_elt);

    fixnum_array *rotate(int i);
    fixnum_array *rotations(int ntimes);
    fixnum_array *repeat(int ntimes);
    const byte *get_cptr_host() const;

    ~fixnum_array();

    int length() const;

    int set(int idx, const byte *data, size_t len);
    size_t retrieve_into(byte *dest, size_t dest_space, int idx) const;
    void retrieve_all(byte *dest, size_t dest_space, int *nelts) const;

    template< template <typename> class Func, typename... Args >
    static void map(Args... args);

private:
    enum { MEM_UNINIT = -1, MEM_HOST = 0, MEM_DEV = 1 };

    fixnum *ptr_dev, *ptr_host;
    int mem_loc;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);

    void fetch_from_dev() const;
    void fetch_to_dev() const;
    byte *get_ptr_host();
};

template< typename fixnum >
std::ostream &
operator<<(std::ostream &os, const fixnum_array<fixnum> *fn_arr);

} // End namespace cuFIXNUM

#include "fixnum_array.cu"
