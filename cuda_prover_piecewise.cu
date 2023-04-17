#include <string>
#include <chrono>
#include <thread>
#include <map>
#include <functional>

#define NDEBUG 1
#define CUB_STDERR

#include <prover_reference_functions.hpp>

#include "multiexp/reduce.cu"

// This is where all the FFTs happen

// template over the bundle of types and functions.
// Overwrites ca!
template <typename B>
typename B::vector_Fr *compute_H(size_t d, typename B::vector_Fr *ca,
                                 typename B::vector_Fr *cb,
                                 typename B::vector_Fr *cc) {
  auto domain = B::get_evaluation_domain(d + 1);

  B::domain_iFFT(domain, ca);
  B::domain_iFFT(domain, cb);

  B::domain_cosetFFT(domain, ca);
  B::domain_cosetFFT(domain, cb);

  // Use ca to store H
  auto H_tmp = ca;

  size_t m = B::domain_get_m(domain);
  // for i in 0 to m: H_tmp[i] *= cb[i]
  B::vector_Fr_muleq(H_tmp, cb, m);

  B::domain_iFFT(domain, cc);
  B::domain_cosetFFT(domain, cc);

  m = B::domain_get_m(domain);

  // for i in 0 to m: H_tmp[i] -= cc[i]
  B::vector_Fr_subeq(H_tmp, cc, m);

  B::domain_divide_by_Z_on_coset(domain, H_tmp);

  B::domain_icosetFFT(domain, H_tmp);

  m = B::domain_get_m(domain);
  typename B::vector_Fr *H_res = B::vector_Fr_zeros(m + 1);
  B::vector_Fr_copy_into(H_tmp, H_res, m);
  return H_res;
}

static size_t read_size_t(FILE* input) {
  size_t n;
  fread((void *) &n, sizeof(size_t), 1, input);
  return n;
}

template< typename B >
struct ec_type;

template<>
struct ec_type<mnt4753_libsnark> {
    typedef ECp_MNT4 ECp;
    typedef ECp2_MNT4 ECpe;
};

template<>
struct ec_type<mnt6753_libsnark> {
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;
};


void
check_trailing(FILE *f, const char *name) {
    long bytes_remaining = 0;
    while (fgetc(f) != EOF)
        ++bytes_remaining;
    if (bytes_remaining > 0)
        fprintf(stderr, "!! Trailing characters in \"%s\": %ld\n", name, bytes_remaining);
}


static inline auto now() -> decltype(std::chrono::high_resolution_clock::now()) {
    return std::chrono::high_resolution_clock::now();
}

template<typename T>
void
print_time(T &t1, const char *str) {
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%s: %ld ms\n", str, tim);
    t1 = t2;
}

template<typename B, typename C>
void run_prover(
        const char *params_path,
        const char *input_path,
        const char *output_path)
{
    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;

    cudaStream_t sA, sB1, sB2, sL;
    CubDebug(cudaStreamCreate(&sA));
    CubDebug(cudaStreamCreate(&sB1));
    CubDebug(cudaStreamCreate(&sB2));
    CubDebug(cudaStreamCreate(&sL));

    B::init_public_params();

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");

    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    printf("d = %zu, m = %zu\n", d, m);

    size_t A_pts_size = ECp::NBYTES_AFF * (m + 1);
    auto A_pts = allocate_memory(A_pts_size);
    auto A_pts_h = allocate_host_memory(A_pts_size);
    size_t B1_pts_size = ECp::NBYTES_AFF * (m + 1);
    auto B1_pts = allocate_memory(B1_pts_size);
    auto B1_pts_h = allocate_host_memory(B1_pts_size);
    size_t B2_pts_size = ECpe::NBYTES_AFF * (m + 1);
    auto B2_pts = allocate_memory(B2_pts_size);
    auto B2_pts_h = allocate_host_memory(B2_pts_size);
    size_t L_pts_size = ECp::NBYTES_AFF * (m - 1);
    auto L_pts = allocate_memory(L_pts_size);
    auto L_pts_h = allocate_host_memory(L_pts_size);
    size_t H_pts_size = ECp::NBYTES_AFF * d;
    auto H_pts_h = allocate_host_memory(H_pts_size);

    auto input_w_size = (m + 1) * ELT_BYTES;
    auto input_others_size = ((d + 1) * 3 + 1) * ELT_BYTES;
    auto inputs_h = allocate_host_memory<char>(input_w_size + input_others_size);
    auto w = allocate_memory(input_w_size);

    size_t scan_temp_size, scan_out_size;
    ec_multiexp_scan_mem_size<C>(m + 1, &scan_temp_size, &scan_out_size);
    size_t temp_size_G1, out_size_G1;
    ec_multiexp_pippenger_mem_size<ECp, C>(&temp_size_G1, &out_size_G1);
    size_t temp_size_G2, out_size_G2;
    ec_multiexp_pippenger_mem_size<ECpe, C>(&temp_size_G2, &out_size_G2);

    auto scan_temp = allocate_memory<void>(scan_temp_size);
    auto scan_out = allocate_memory<int>(scan_out_size);
    auto temp_A = allocate_memory<void>(temp_size_G1);
    auto out_A = allocate_memory<var>(out_size_G1);
    auto temp_B1 = allocate_memory<void>(temp_size_G1);
    auto out_B1 = allocate_memory<var>(out_size_G1);
    auto temp_B2 = allocate_memory<void>(temp_size_G2);
    auto out_B2 = allocate_memory<var>(out_size_G2);
    auto temp_L = allocate_memory<void>(temp_size_G1);
    auto out_L = allocate_memory<var>(out_size_G1);

    auto out_A_h = allocate_host_memory(ECp::NBYTES);
    auto out_B1_h = allocate_host_memory(ECp::NBYTES);
    auto out_B2_h = allocate_host_memory(ECpe::NBYTES);
    auto out_L_h = allocate_host_memory(ECp::NBYTES);

    print_time(t, "alloc device mem");

    auto params = B::alloc_params(d, m);
    fread(A_pts_h.get(), A_pts_size, 1, params_file);
    CubDebug(cudaMemcpyAsync(A_pts.get(), A_pts_h.get(), A_pts_size, cudaMemcpyHostToDevice));
    // B::read_params_A(params, A_pts_h.get());
    fread(B1_pts_h.get(), B1_pts_size, 1, params_file);
    CubDebug(cudaMemcpyAsync(B1_pts.get(), B1_pts_h.get(), B1_pts_size, cudaMemcpyHostToDevice));
    B::read_params_B1(params, B1_pts_h.get());
    fread(B2_pts_h.get(), B2_pts_size, 1, params_file);
    CubDebug(cudaMemcpyAsync(B2_pts.get(), B2_pts_h.get(), B2_pts_size, cudaMemcpyHostToDevice));
    // B::read_params_B2(params, B2_pts_h.get());
    fread(L_pts_h.get(), L_pts_size, 1, params_file);
    CubDebug(cudaMemcpyAsync(L_pts.get(), L_pts_h.get(), L_pts_size, cudaMemcpyHostToDevice));
    // B::read_params_L(params, L_pts_h.get());
    fread(H_pts_h.get(), H_pts_size, 1, params_file);
    B::read_params_H(params, H_pts_h.get());
    fclose(params_file);
    print_time(t, "load params");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    fread(inputs_h.get(), input_w_size, 1, inputs_file);
    CubDebug(cudaMemcpyAsync(w.get(), inputs_h.get(), input_w_size, cudaMemcpyHostToDevice));
    fread(inputs_h.get() + input_w_size, input_others_size, 1, inputs_file);
    fclose(inputs_file);
    auto inputs = B::make_input(inputs_h.get(), d, m);
    print_time(t, "load inputs");

    auto t_gpu = t;

    ec_multiexp_scan<typename ECp::group_type, C>(w.get(), scan_out.get(), m + 1, scan_temp.get(), scan_temp_size, nullptr);
    ec_multiexp_pippenger<ECp, C>(A_pts.get(), scan_out.get(), out_A.get(), temp_A.get(), 0, m + 1, sA);
    ec_multiexp_pippenger<ECp, C>(B1_pts.get(), scan_out.get(), out_B1.get(), temp_B1.get(), 0, m + 1, sB1);
    ec_multiexp_pippenger<ECpe, C>(B2_pts.get(), scan_out.get(), out_B2.get(), temp_B2.get(), 0, m + 1, sB2);
    ec_multiexp_pippenger<ECp, C>(L_pts.get(), scan_out.get(), out_L.get(), temp_L.get(), 2, m - 1, sL);
    CubDebug(cudaMemcpyAsync(out_A_h.get(), out_A.get(), ECp::NBYTES, cudaMemcpyDeviceToHost, sA));
    CubDebug(cudaMemcpyAsync(out_B1_h.get(), out_B1.get(), ECp::NBYTES, cudaMemcpyDeviceToHost, sB1));
    CubDebug(cudaMemcpyAsync(out_B2_h.get(), out_B2.get(), ECpe::NBYTES, cudaMemcpyDeviceToHost, sB2));
    CubDebug(cudaMemcpyAsync(out_L_h.get(), out_L.get(), ECp::NBYTES, cudaMemcpyDeviceToHost, sL));
    print_time(t_gpu, "gpu launch");

    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    typename B::vector_G1 *H;
    typename B::vector_Fr *coefficients_for_H;
    G1 *evaluation_Ht;
    std::thread cpu1_thread([&]() {
        auto t_cpu1 = now();
        H = B::params_H(params);
        coefficients_for_H =
            compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
        print_time(t_cpu1, "cpu fft H");
        evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);
        print_time(t_cpu1, "cpu multiexp H");
    });

    CubDebug(cudaStreamSynchronize(sA));
    G1 *evaluation_At = B::read_pt_ECp(out_A_h.get());
    CubDebug(cudaStreamSynchronize(sB1));
    G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1_h.get());
    CubDebug(cudaStreamSynchronize(sB2));
    G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2_h.get());
    CubDebug(cudaStreamSynchronize(sL));
    G1 *evaluation_Lt = B::read_pt_ECp(out_L_h.get());
    print_time(t, "gpu e2e");

    cpu1_thread.join();
    print_time(t, "cpu 1 wait");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");

    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

    CubDebug(cudaStreamDestroy(sA));
    CubDebug(cudaStreamDestroy(sB1));
    CubDebug(cudaStreamDestroy(sB2));
    CubDebug(cudaStreamDestroy(sL));

    B::delete_vector_G1(H);

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);
    B::delete_groth16_params(params);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

int main(int, char **argv) {
  setbuf(stdout, NULL);
  std::string curve(argv[1]);
  std::string mode(argv[2]);

  const char *params_path = argv[3];

  if (mode == "compute") {
      const char *input_path = argv[4];
      const char *output_path = argv[5];

      using config = ec_multiexp_config<MULTIEXP_WIN_BITS, 753, MULTIEXP_BLOCK_SIZE>;

      if (curve == "MNT4753") {
          run_prover<mnt4753_libsnark, config>(params_path, input_path, output_path);
      } else if (curve == "MNT6753") {
          run_prover<mnt6753_libsnark, config>(params_path, input_path, output_path);
      }
  }

  return 0;
}
