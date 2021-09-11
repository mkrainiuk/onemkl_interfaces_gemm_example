CXX = dpcpp
COPT = -fsycl

mklcpu: run_mklcpu

cublas: run_cublas

mklgpu: run_mklgpu

onemkl_gemm_usm.out: onemkl_gemm_usm.o
	$(CXX) $(COPT) $^ -o $@ -L${ONEMKL}/lib -lonemkl -ldl

onemkl_gemm_usm_mklcpu.out: onemkl_gemm_usm_mklcpu.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_mklcpu.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lOpenCL -ldl -lm -lpthread

onemkl_gemm_usm_mklgpu.out: onemkl_gemm_usm_mklgpu.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_mklgpu.a ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lOpenCL -ldl -lm -lpthread

onemkl_gemm_usm_cublas.out: onemkl_gemm_usm_cublas.o
	$(CXX) $(COPT) $^ -o $@ ${ONEMKL}/lib/libonemkl_blas_cublas.a -lcublas -ldl

%.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -c $^ -o $@

%_mklcpu.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DRUN_ON_CPU -DGEMM_MKL -c $^ -o $@

%_mklgpu.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DGEMM_MKL_GPU -c $^ -o $@

%_cublas.o: %.cpp
	$(CXX) $(COPT) -I${ONEMKL}/include -DGEMM_CUBLAS -c $^ -o $@

run_mklcpu: onemkl_gemm_usm_mklcpu.out
	for n in 500 1000 2000 4000 8000; do ./$^ S N N $$n $$n $$n 100 $$n $$n $$n; done

run_cublas: onemkl_gemm_usm_cublas.out
	for n in 500 1000 2000 4000 8000; do ./$^ S N N $$n $$n $$n 100 $$n $$n $$n; done

run_mklgpu: onemkl_gemm_usm_mklgpu.out
	for n in 500 1000 2000 4000 8000; do ./$^ S N N $$n $$n $$n 100 $$n $$n $$n; done

clean:
	rm -rf *.o *.out

