#include <cassert>
#include <cstdio>
#include <fstream>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <omp.h>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/knowledge_commitment/knowledge_commitment.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>

#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>
using namespace libff;
using namespace libsnark;

template<typename T>
class vec_uninit : public std::vector<T> {
  static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");
public:
  vec_uninit() = default;

  vec_uninit(size_t n) {
    this->_M_impl._M_start = this->_M_allocate(n);
    this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = this->_M_impl._M_start + n;
  }

  void resize_uninit(size_t n) {
    if (n >= this->size()) {
      this->reserve(n);
      this->_M_impl._M_finish = this->_M_impl._M_start + n;
    } else {
      this->resize(n);
    }
  }
};

// be careful not to trigger re-allocation
template<typename T>
class vec_ptr : public std::vector<T> {
public:
  vec_ptr(void *p, size_t n) {
    this->_M_impl._M_start = reinterpret_cast<T *>(p);
    this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = this->_M_impl._M_start + n;
  }

  ~vec_ptr() {
    this->_M_impl._M_start = this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = nullptr;
  }
};

const size_t num_limbs = 12;

template<typename ppT>
void write_fq(FILE* output, const Fq<ppT> &x) {
  fwrite((void *) x.mont_repr.data, num_limbs * sizeof(mp_size_t), 1, output);
}

template<typename ppT>
void write_fr(FILE* output, const Fr<ppT> &x) {
  fwrite((void *) x.mont_repr.data, num_limbs * sizeof(mp_size_t), 1, output);
}

template<typename ppT>
void write_fqe(FILE* output, const Fqe<ppT> &x) {
  std::vector<Fq<ppT>> v = x.all_base_field_elements();
  size_t deg = Fqe<ppT>::extension_degree();
  for (size_t i = 0; i < deg; ++i) {
    write_fq<ppT>(output, v[i]);
  }
}

template<typename ppT>
void write_g1(FILE* output, G1<ppT> g) {
  if (g.is_zero())  {
    write_fq<ppT>(output, Fq<ppT>::zero());
    write_fq<ppT>(output, Fq<ppT>::zero());
    return;
  }

  g.to_affine_coordinates();
  write_fq<ppT>(output, g.X());
  write_fq<ppT>(output, g.Y());
}

template<typename ppT>
void write_g2(FILE* output, G2<ppT> g) {
  if (g.is_zero())  {
    write_fqe<ppT>(output, Fqe<ppT>::zero());
    write_fqe<ppT>(output, Fqe<ppT>::zero());
    return;
  }

  g.to_affine_coordinates();
  write_fqe<ppT>(output, g.X());
  write_fqe<ppT>(output, g.Y());
}

template<typename ppT>
void read_fq(FILE* input, Fq<ppT> &x) {
  fread((void *) x.mont_repr.data, num_limbs * sizeof(mp_size_t), 1, input);
}

template<typename ppT>
void read_fq(const void* input, Fq<ppT> &x) {
  memcpy((void *) x.mont_repr.data, input, num_limbs * sizeof(mp_size_t));
}

template<typename ppT>
void read_fr(FILE* input, Fr<ppT> &x) {
  fread((void *) x.mont_repr.data, num_limbs * sizeof(mp_size_t), 1, input);
}

template<typename ppT>
void read_fr(const void* input, Fq<ppT> &x) {
  memcpy((void *) x.mont_repr.data, input, num_limbs * sizeof(mp_size_t));
}

template<typename ppT>
void read_g1(FILE* input, G1<ppT> &g) {
  read_fq<ppT>(input, g.X_);
  read_fq<ppT>(input, g.Y_);
  if (g.Y_.is_zero()) {
    g = G1<ppT>::zero();
  } else {
    g.Z_ = Fq<ppT>::one();
  }
}

template<typename ppT>
void read_g1(const void* input, G1<ppT> &g) {
  read_fq<ppT>(input, g.X_);
  read_fq<ppT>(static_cast<const char *>(input) + sizeof(Fq<ppT>), g.Y_);
  if (g.Y_.is_zero()) {
    g = G1<ppT>::zero();
  } else {
    g.Z_ = Fq<ppT>::one();
  }
}

template<typename ppT>
void read_fqe(FILE* input, Fqe<ppT> &x) {
  fread(&x, sizeof(Fqe<ppT>), 1, input);
}

template<typename ppT>
void read_fqe(const void* input, Fqe<ppT> &x) {
  memcpy(&x, input, sizeof(Fqe<ppT>));
}

template<typename ppT>
G2<ppT> read_g2(FILE* input, G2<ppT> &g) {
  read_fqe<ppT>(input, g.X_);
  read_fqe<ppT>(input, g.Y_);
  if (g.Y_.is_zero()) {
    g = G2<ppT>::zero();
  } else {
    g.Z_ = Fqe<ppT>::one();
  }
}

template<typename ppT>
void read_g2(const void* input, G2<ppT> &g) {
  read_fqe<ppT>(input, g.X_);
  read_fqe<ppT>(static_cast<const char *>(input) + sizeof(Fqe<ppT>), g.Y_);
  if (g.Y_.is_zero()) {
    g = G2<ppT>::zero();
  } else {
    g.Z_ = Fqe<ppT>::one();
  }
}

size_t read_size_t(FILE* input) {
  size_t n;
  fread((void *) &n, sizeof(size_t), 1, input);
  return n;
}

void write_size_t(FILE* output, size_t n) {
  fwrite((void *) &n, sizeof(size_t), 1, output);
}
