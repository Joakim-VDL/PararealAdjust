#pragma once

#include <iostream>

//? ----------------------------------------------------------
//?
//? Description:
//?     Kernels and functions are "unified" for proper usage
//?     depending on whether GPU support is activated or not. 
//?
//? ----------------------------------------------------------

#include "Kernels.hpp"
#include "functions.hpp"

namespace LeXInt
{
    int threadsPerBlock = 128;
    //?l2 norm
    double l2norm(double *x, size_t N, bool GPU, GPU_handle& cublas_handle)
    {
        double norm;

        if (GPU == true)
        {
            //* CUDA
            //! Does not work and not used so commented.
            // cublasDnrm2(cublas_handle.cublas_handle, N, x, 1, &norm);
        }
        else
        {
            //* C++
            norm = l2norm_Cpp(x, N);
        }

        return norm;
    }

    //? Set x = y
    void copy(double *x, double *y, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            copy_CUDA<<<numBlocks, threadsPerBlock>>>(x, y, N);
        }
        else
        {
            //* C++
            copy_Cpp(x, y, N);
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    void ones(double *x, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            ones_CUDA<<<numBlocks, threadsPerBlock>>>(x, N);
        }
        else
        {
            //* C++
            ones_Cpp(x, N);
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    void eigen_ones(double *x, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            eigen_ones_CUDA<<<numBlocks, threadsPerBlock>>>(x, N);
        }
        else
        {
            //* C++
            eigen_ones_Cpp(x, N);
        }
    }


    //? y = ax
    void axpby(double a, double *x, 
                         double *y, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, y, N);
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, y, N);
        }
    }


    //? z = ax + by
    void axpby(double a, double *x, 
               double b, double *y, 
                         double *z, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, z, N);
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, b, y, z, N);
        }
    }

    //? w = ax + by + cz
    void axpby(double a, double *x, 
               double b, double *y, 
               double c, double *z, 
                         double *w, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, w, N);
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, b, y, c, z, w, N);
        }
    }

    //? v = ax + by + cz + dw
    void axpby(double a, double *x, 
               double b, double *y, 
               double c, double *z, 
               double d, double *w, 
                         double *v, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, d, w, v, N);
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, b, y, c, z, d, w, v, N);
        }
    }

        //? u = ax + by + cz + dw + ev
    void axpby(double a, double *x, 
               double b, double *y, 
               double c, double *z, 
               double d, double *w, 
               double e, double *v,
                         double *u, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            //* CUDA
            axpby_CUDA<<<numBlocks, threadsPerBlock>>>(a, x, b, y, c, z, d, w, e, v, u, N);
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, b, y, c, z, d, w, e, v, u, N);
        }
    }
}