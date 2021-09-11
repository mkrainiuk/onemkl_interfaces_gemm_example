# onemkl_interfaces_gemm_example

Repo contains examples for [oneAPI Math Kernel Library Interfaces project](https://github.com/oneapi-src/oneMKL)

## Reproducing SC21 paper results
1. Build oneMKL project for MKLCPU, MKLGPU, and CUBLAS backends (see build instructions [here](https://github.com/oneapi-src/oneMKL#build-setup)).
2. Make sure you source required compilers (Intel DPC++ for MKLCPU/MKLGPU backends and DPC++ with CUDA backend for CUBLAS backend) and third-party libraries (Intel oneMKL and NVIDIA cuBLAS)
3. Set `ONEMKL` path to oneMKL interfaces header files and libraries (GNU Makefile from the repo expects `$ONEMKL/include` path with oneMKL header files and `$ONEMKL/lib` with libraries)
4. Use GNU Makefile from the repo to build and run GEMM benchmark as shown below:
```
# build and run GEMM with MKLCPU backend
> make mklcpu

# build and run GEMM with CUBLAS backend
> make cublas

# build and run GEMM with MKLGPU backend
> make mklgpu

# clean the repo
> make clean
```
