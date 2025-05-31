#pragma once

#include <iostream>
#include "Kernels.hpp"
#include "functions.hpp"

namespace LeXInt
{
    int threadsPerBlock = 256;

    // l2 norm (unchanged, as GPU version is commented out)
    double l2norm(const double* x, size_t N, bool GPU, GPU_handle& cublas_handle)
    {
        double norm;
        if (GPU)
        {
            // cublasDnrm2(cublas_handle.cublas_handle, N, x, 1, &norm);
        }
        else
        {
            norm = l2norm_Cpp(x, N);
        }
        return norm;
    }

    // Set x = y
    void copy(const double* x, double* y, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            copy_CUDA<<<numBlocks, threadsPerBlock>>>(x, y, N);
        }
        else
        {
            copy_Cpp(x, y, N);
        }
    }

    // ones(y) = (y[0:N] =) 1.0
    void ones(double* x, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            ones_CUDA<<<numBlocks, threadsPerBlock>>>(x, N);
        }
        else
        {
            ones_Cpp(x, N);
        }
    }

    // eigen_ones(y): y[:] = 0, y[0] = 1
    void eigen_ones(double* x, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            eigen_ones_CUDA<<<numBlocks, threadsPerBlock>>>(x, N);
        }
        else
        {
            eigen_ones_Cpp(x, N);
        }
    }

    // y = ax
    void axpby(double a, const double* x, double* y, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, y, N);
        }
        else
        {
            axpby_Cpp(a, x, y, N);
        }
    }

    // z = ax + by
    void axpby(double a, const double* x,
               double b, const double* y,
               double* z, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, z, N);
        }
        else
        {
            axpby_Cpp(a, x, b, y, z, N);
        }
    }

    // w = ax + by + cz
    void axpby(double a, const double* x,
               double b, const double* y,
               double c, const double* z,
               double* w, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, w, N);
        }
        else
        {
            axpby_Cpp(a, x, b, y, c, z, w, N);
        }
    }

    // v = ax + by + cz + dw
    void axpby(double a, const double* x,
               double b, const double* y,
               double c, const double* z,
               double d, const double* w,
               double* v, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, d, w, v, N);
        }
        else
        {
            axpby_Cpp(a, x, b, y, c, z, d, w, v, N);
        }
    }

    // u = ax + by + cz + dw + ev
    void axpby(double a, const double* x,
               double b, const double* y,
               double c, const double* z,
               double d, const double* w,
               double e, const double* v,
               double* u, size_t N, bool GPU)
    {
        if (GPU)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, d, w, e, v, u, N);
        }
        else
        {
            axpby_Cpp(a, x, b, y, c, z, d, w, e, v, u, N);
        }
    }
}
