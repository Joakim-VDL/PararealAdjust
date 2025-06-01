#pragma once

#include "Problems.hpp"

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

//? ====================================================================================== ?//

#ifdef __CUDACC__

// Device periodic boundary condition for a single index
__device__ __forceinline__ int PBC_dev(int ii, int N)
{
    return (ii + N) % N;
}

// Shared memory optimised 2D advection-diffusion kernel
__global__ void Dif_Adv_2D_Optimised(
    int N, double dx, double dy, double velocity,
    const double* __restrict__ input,
    double* __restrict__ output)
{
    constexpr int TILE = 64; // Block size, find the max for your GPU 
    // compile with increasing powers of 2 until you get 'uses too much shared data' 

    // Shared memory tile with halo (2 extra cells on each side)
    __shared__ double tile[TILE + 4][TILE + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ii = blockIdx.y * TILE + ty;
    int jj = blockIdx.x * TILE + tx;

    // Load shared memory tile (including halo)
    for (int dy_off = 0; dy_off < TILE + 4; dy_off += TILE)
    {
        for (int dx_off = 0; dx_off < TILE + 4; dx_off += TILE)
        {
            int local_y = ty + dy_off;
            int local_x = tx + dx_off;
            int global_y = PBC_dev(ii + dy_off - 2, N);
            int global_x = PBC_dev(jj + dx_off - 2, N);
            if (local_y < TILE + 4 && local_x < TILE + 4)
                tile[local_y][local_x] = input[global_y * N + global_x];
        }
    }
    __syncthreads();

    // Only compute if inside the real tile (not in the halo)
    if (ii < N && jj < N && tx < TILE && ty < TILE)
    {
        int lx = tx + 2;
        int ly = ty + 2;

        double val_c   = tile[ly][lx];
        double val_xp1 = tile[ly][lx + 1];
        double val_xm1 = tile[ly][lx - 1];
        double val_xp2 = tile[ly][lx + 2];
        double val_yp1 = tile[ly + 1][lx];
        double val_ym1 = tile[ly - 1][lx];
        double val_yp2 = tile[ly + 2][lx];

        output[ii * N + jj] =
            // Diffusion
            (val_xp1 - 4.0 * val_c + val_xm1) / (dx * dx)
            + (val_yp1 + val_ym1) / (dy * dy)
            // Advection x
            + velocity / dx *
                (-2.0 / 6.0 * val_xm1
                 -3.0 / 6.0 * val_c
                 +6.0 / 6.0 * val_xp1
                 -1.0 / 6.0 * val_xp2)
            // Advection y
            + velocity / dy *
                (-2.0 / 6.0 * val_ym1
                 -3.0 / 6.0 * val_c
                 +6.0 / 6.0 * val_yp1
                 -1.0 / 6.0 * val_yp2);
    }
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
            constexpr int TILE = 16;
            dim3 threads(TILE, TILE);
            dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
            Dif_Adv_2D_Optimised<<<blocks, threads>>>(N, dx, dy, velocity, input, output);
#endif
        }
        else
        {
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
