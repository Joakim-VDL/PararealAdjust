#pragma once

#include "Problems.hpp"

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

// Device inline PBC for maximum performance
__device__ __forceinline__ int PBC_dev(int ii, int jj, int N)
{
    ii = (ii + N) % N;
    jj = (jj + N) % N;
    return N * ii + jj;
}

__global__ void Dif_Adv_2D(
    int N, double dx, double dy, double velocity,
    const double* __restrict__ input,
    double* __restrict__ output)
{
    int ii = threadIdx.y + blockIdx.y * blockDim.y;
    int jj = threadIdx.x + blockIdx.x * blockDim.x;

    if (ii >= N || jj >= N) return;

    const int idx      = PBC_dev(ii,     jj,     N);
    const int idx_p1j  = PBC_dev(ii,     jj + 1, N);
    const int idx_m1j  = PBC_dev(ii,     jj - 1, N);
    const int idx_ip1  = PBC_dev(ii + 1, jj,     N);
    const int idx_im1  = PBC_dev(ii - 1, jj,     N);
    const int idx_jp2  = PBC_dev(ii,     jj + 2, N);
    const int idx_im2  = PBC_dev(ii + 2, jj,     N);

    output[idx] =
        // Diffusion
        (input[idx_p1j] - 4.0 * input[idx] + input[idx_m1j]) / (dx * dx)
        + (input[idx_ip1] + input[idx_im1]) / (dy * dy)
        // Advection x
        + velocity / dx *
            (-2.0 / 6.0 * input[idx_m1j]
             -3.0 / 6.0 * input[idx]
             +6.0 / 6.0 * input[idx_p1j]
             -1.0 / 6.0 * input[idx_jp2])
        // Advection y
        + velocity / dy *
            (-2.0 / 6.0 * input[idx_im1]
             -3.0 / 6.0 * input[idx]
             +6.0 / 6.0 * input[idx_ip1]
             -1.0 / 6.0 * input[idx_im2]);
}

#endif

struct RHS_Dif_Adv_2D : public Problems_2D
{
    RHS_Dif_Adv_2D(int _N, double _dx, double _dy, double _velocity)
        : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output, bool GPU)
    {
        if (GPU)
        {
#ifdef __CUDACC__
            int num_threads = 16;
            dim3 threads(num_threads, num_threads);
            dim3 blocks((N + num_threads - 1) / num_threads, (N + num_threads - 1) / num_threads);
            Dif_Adv_2D<<<blocks, threads>>>(N, dx, dy, velocity, input, output);
#endif
        }
        else
        {
            // Flat OpenMP parallel loop for best CPU performance
#pragma omp parallel for collapse(2)
            for (int ii = 0; ii < N; ++ii)
            {
                for (int jj = 0; jj < N; ++jj)
                {
                    const int idx      = PBC(ii,     jj,     N);
                    const int idx_p1j  = PBC(ii,     jj + 1, N);
                    const int idx_m1j  = PBC(ii,     jj - 1, N);
                    const int idx_ip1  = PBC(ii + 1, jj,     N);
                    const int idx_im1  = PBC(ii - 1, jj,     N);
                    const int idx_jp2  = PBC(ii,     jj + 2, N);
                    const int idx_im2  = PBC(ii + 2, jj,     N);

                    output[idx] =
                        (input[idx_p1j] - 4.0 * input[idx] + input[idx_m1j]) / (dx * dx)
                        + (input[idx_ip1] + input[idx_im1]) / (dy * dy)
                        + velocity / dx *
                            (-2.0 / 6.0 * input[idx_m1j]
                             -3.0 / 6.0 * input[idx]
                             +6.0 / 6.0 * input[idx_p1j]
                             -1.0 / 6.0 * input[idx_jp2])
                        + velocity / dy *
                            (-2.0 / 6.0 * input[idx_im1]
                             -3.0 / 6.0 * input[idx]
                             +6.0 / 6.0 * input[idx_ip1]
                             -1.0 / 6.0 * input[idx_im2]);
                }
            }
        }
    }

    ~RHS_Dif_Adv_2D() {}
};

//? ====================================================================================== ?//
