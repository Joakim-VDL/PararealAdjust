#pragma once

#include "Timer.hpp"

#ifdef __CUDACC__
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif

struct GPU_handle
{
#ifdef __CUDACC__
    cublasHandle_t cublas_handle;
#endif

    GPU_handle()
    {
#ifdef __CUDACC__
        cublasCreate_v2(&cublas_handle);
#endif
    }

    ~GPU_handle()
    {
#ifdef __CUDACC__
        cublasDestroy(cublas_handle);
#endif
    }
};

namespace LeXInt
{
#ifdef __CUDACC__

// Set y = x
__global__ void copy_CUDA(const double* __restrict__ x, double* __restrict__ y, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        y[ii] = x[ii];
    }
}

// ones(y) = (y[0:N] =) 1.0
__global__ void ones_CUDA(double* __restrict__ x, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        x[ii] = 1.0;
    }
}

// eigen_ones(y): y[:] = 0, y[0] = 1
__global__ void eigen_ones_CUDA(double* __restrict__ x, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        x[ii] = 0.0;
    }
    if (ii == 0)
    {
        x[0] = 1.0;
    }
}

// y = ax
__global__ void axpby_CUDA(double a, const double* __restrict__ x, double* __restrict__ y, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        y[ii] = (a * x[ii]);
    }
}

// z = ax + by
__global__ void axpby_CUDA(double a, const double* __restrict__ x,
                           double b, const double* __restrict__ y,
                           double* __restrict__ z, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        z[ii] = (a * x[ii]) + (b * y[ii]);
    }
}

// w = ax + by + cz
__global__ void axpby_CUDA(double a, const double* __restrict__ x,
                           double b, const double* __restrict__ y,
                           double c, const double* __restrict__ z,
                           double* __restrict__ w, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        w[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]);
    }
}

// v = ax + by + cz + dw
__global__ void axpby_CUDA(double a, const double* __restrict__ x,
                           double b, const double* __restrict__ y,
                           double c, const double* __restrict__ z,
                           double d, const double* __restrict__ w,
                           double* __restrict__ v, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        v[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]) + (d * w[ii]);
    }
}

// u = ax + by + cz + dw + ev
__global__ void axpby_CUDA(double a, const double* __restrict__ x,
                           double b, const double* __restrict__ y,
                           double c, const double* __restrict__ z,
                           double d, const double* __restrict__ w,
                           double e, const double* __restrict__ v,
                           double* __restrict__ u, size_t N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < N)
    {
        u[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]) + (d * w[ii]) + (e * v[ii]);
    }
}

#endif
}
