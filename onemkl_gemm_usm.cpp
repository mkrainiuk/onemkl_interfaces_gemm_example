#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <list>
#include <map>
#include <type_traits>
#include <chrono>

#include <CL/sycl.hpp>

#include "oneapi/mkl.hpp"


//
// helper functions
//
template <typename fp>
fp set_fp_value(fp arg1) { return arg1;}

template <typename fp> fp rand_scalar() { return fp(std::rand()) / fp(RAND_MAX) - fp(0.5); }

template <typename fp> void rand_matrix(fp *M, oneapi::mkl::transpose trans, int m, int n, int ld)
{

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename fp>
int LD(oneapi::mkl::transpose trans, int m, int n)
{

    int new_ld, LD_OFFSET = 64 / sizeof(fp);

    new_ld = (trans == oneapi::mkl::transpose::nontrans) ? m : n;
    new_ld = (new_ld + LD_OFFSET - 1) / LD_OFFSET * LD_OFFSET;
	new_ld = (new_ld * sizeof(fp)) % 512 ? new_ld : new_ld + LD_OFFSET;

    return new_ld;
}


template <typename fp>
void run_gemm(const cl::sycl::device &dev, oneapi::mkl::transpose transA, oneapi::mkl::transpose transB,
              int m, int n, int k, int ldA, int ldB, int ldC, int nb_it) {

    ldA = (ldA < 0) ? LD<fp>(transA, m, k) : ldA;
    ldB = (ldB < 0) ? LD<fp>(transB, k, n) : ldB;
    ldC = (ldC < 0) ? LD<fp>(oneapi::mkl::transpose::N, m, n) : ldC;

    int sizea, sizeb, sizec = ldC * n;
    int i;
    double diff, min, max, mean;

    // set scalar fp value
    fp alpha = set_fp_value(fp(1.0));
    fp beta  = set_fp_value(fp(0.0));

    // prepare matrix data
    fp *A, *B, *C;

    // Catch asynchronous exceptions
    auto exception_handler = [] (cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(cl::sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                << e.what() << std::endl;
            }
        }
    };

    cl::sycl::context cxt(dev);

    cl::sycl::queue queue(cxt, dev, exception_handler);

    cl::sycl::event gemm_done;
    std::vector<cl::sycl::event> gemm_dependencies;
    sizea = (transA == oneapi::mkl::transpose::nontrans) ? ldA * k : ldA * m;
    sizeb = (transB == oneapi::mkl::transpose::nontrans) ? ldB * n : ldB * k;

    A = (fp *)malloc_shared(sizea * sizeof(fp), queue);
    B = (fp *)malloc_shared(sizeb * sizeof(fp), queue);
    C = (fp *)malloc_shared(sizec * sizeof(fp), queue);

    if (!A || !B || !C )
        throw std::runtime_error("Failed to allocate USM memory.");

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, m, n, ldC);

    try {
        auto start = std::chrono::high_resolution_clock::now();
#ifdef GEMM_CUBLAS
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#elif defined GEMM_MKL
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#elif defined GEMM_MKL_GPU
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#else
        gemm_done = oneapi::mkl::blas::column_major::gemm(queue, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#endif
        gemm_done.wait();
        auto stop = std::chrono::high_resolution_clock::now();
        long long totalTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        diff = double(totalTime) / 1000000.0;
    }
    catch(cl::sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                  << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    min = diff;
    max = 0.0;
    mean = 0.0;

    double nops = 2.0 * double(m) * double(n) * double(k);
    for (i = 0; i < nb_it; i++) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
#ifdef GEMM_CUBLAS
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#elif defined GEMM_MKL
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#elif defined GEMM_MKL_GPU
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#else
        gemm_done = oneapi::mkl::blas::column_major::gemm(queue, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
#endif
            gemm_done.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            long long totalTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            diff = double(totalTime) / 1000000.0;
        }
        catch(cl::sycl::exception const& e) {
            std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                      << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
        }
        mean += diff;
        if (min > diff) min = diff;
        if (max < diff) max = diff;
    }

    mean = mean / (double) nb_it;

    double nmem = (double(m) * double(n) + double(m) * double(k) + double(n) * double(k)) * sizeof(fp);
    if ((std::is_same<fp, std::complex<float>>::value) || (std::is_same<fp, std::complex<double>>::value))
        nops *= 4.0;

    printf("%lf (bw),%lf (gflopmax),%lf (gflopmean),%g (smin),%g (smean)\n", nmem / (1e9 * min), nops / (1e9 * min), nops / (1e9 * mean), min, mean);


    free(A, queue);
    free(B, queue);
    free(C, queue);

}


int main (int argc, char ** argv) {

#ifdef RUN_ON_CPU
    cl::sycl::device dev = cl::sycl::device(cl::sycl::cpu_selector());
    if (dev.is_gpu()) printf("Running on CPU device\n");
#else
    cl::sycl::device dev = cl::sycl::device(cl::sycl::gpu_selector());
    if (dev.is_gpu()) printf("Running on GPU device\n");
#endif

    bool is_level0 = dev.get_info<cl::sycl::info::device::opencl_c_version>().empty();
    if (is_level0) printf("DPC++ running with Level0 backend\n");
    else printf("DPC++ running with OpenCL backend\n");

    int m = 10, n = 10, k = 10, lda = -1, ldb = -1, ldc = -1, nb_it = 500;
    oneapi::mkl::transpose ta = oneapi::mkl::transpose::N, tb = oneapi::mkl::transpose::N;
    char data_type = 'D';

    if (argc > 1) {
        data_type = argv[1][0];
    }

    if (argc > 3) {
        ta = (argv[2][0] == 'N') ? oneapi::mkl::transpose::N : ((argv[2][0] == 'T') ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::C);
        tb = (argv[3][0] == 'N') ? oneapi::mkl::transpose::N : ((argv[3][0] == 'T') ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::C);
    }

    if (argc > 6) {
        m = atoi(argv[4]);
        n = atoi(argv[5]);
        k = atoi(argv[6]);
    }


    if (argc > 7) nb_it = atoi(argv[7]);

    if (argc > 10) {
        lda = atoi(argv[8]);
        ldb = atoi(argv[9]);
        ldc = atoi(argv[10]);
    }

    printf("%cGEMM,%d,%d,%d,%d,%d,%d,%d,%d,",
           data_type, (int) ta, (int) tb, m, n, k, lda, ldb, ldc);

    if (data_type == 'S')
        run_gemm<float>(dev, ta, tb, m, n, k, lda, ldb, ldc, nb_it);
    else if (data_type == 'D')
        run_gemm<double>(dev, ta, tb, m, n, k, lda, ldb, ldc, nb_it);
    else {
        printf("Error wrong data type\n");
        return 1;
    }

    return 0;

}
