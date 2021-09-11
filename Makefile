CXX = clang++
COPT = -fsycl

mklcpu: run_mklcpu

cublas: run_cublas

mklgpu: run_mklgpu

onemkl_%_usm.out: onemkl_%_usm.o
	$(CXX) $(COPT) $^ -o $@ -L${ONEMKL}/lib -lonemkl -ldl

onemkl_%_usm_mklcpu.out: onemkl_%_usm_mklcpu.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_mklcpu.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lOpenCL -ldl -lm -lpthread

onemkl_%_usm_mklgpu.out: onemkl_%_usm_mklgpu.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_mklgpu.a ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lOpenCL -ldl -lm -lpthread

onemkl_%_usm_cublas.out: onemkl_%_usm_cublas.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_cublas.a -lcublas -lcuda -ldl

%.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -c $^ -o $@

%_mklcpu.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DRUN_ON_CPU -DGEMM_MKL -c $^ -o $@

%_mklgpu.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DGEMM_MKL_GPU -c $^ -o $@

%_cublas.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DGEMM_CUBLAS -c $^ -o $@

run_mklcpu: onemkl_gemm_usm_mklcpu.out
	for n in 500 1000 2000 4000 8000; do ./onemkl_gemm_usm_mklcpu.out S N N $$n $$n $$n 100 $$n $$n $$n; done

run_cublas: onemkl_gemm_usm_cublas.out
	for n in 500 1000 2000 4000 8000; do ./onemkl_gemm_usm_cublas.out S N N $$n $$n $$n 100 $$n $$n $$n; done

run_mklgpu: onemkl_gemm_usm_mklgpu.out
	for n in 500 1000 2000 4000 8000; do ./onemkl_gemm_usm_mklgpu.out S N N $$n $$n $$n 100 $$n $$n $$n; done

clean:
	rm -rf *.o *.out

