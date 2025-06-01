#pragma once

#include "Problems.hpp"
// #include "error_check.hpp"

using namespace std;

//! This function has 2 vector reads and writes.

//? ====================================================================================== ?//

#ifdef __CUDACC__

__global__ void Dif_Adv_2D_Optimised(int N, double dx, double dy, double velocity, double* input, double* output)
{
    // Increased block size
    const int TILE_SIZE = 16;
    __shared__ double tile[TILE_SIZE + 4][TILE_SIZE + 4]; // +4 to cover stencil reach

    int ii = blockIdx.y * TILE_SIZE + threadIdx.y;
    int jj = blockIdx.x * TILE_SIZE + threadIdx.x;

    int local_ii = threadIdx.y + 2; // +2 for halo
    int local_jj = threadIdx.x + 2;

    // Load data into shared memory, including halos
    if (ii < N && jj < N)
        tile[local_ii][local_jj] = input[PBC(ii, jj, N)];
    
    // Halo for boundaries
    if (threadIdx.y < 2 && ii >= 2)
        tile[local_ii - 2][local_jj] = input[PBC(ii - 2, jj, N)];
    if (threadIdx.y >= TILE_SIZE - 2 && ii + 2 < N)
        tile[local_ii + 2][local_jj] = input[PBC(ii + 2, jj, N)];
    if (threadIdx.x < 2 && jj >= 2)
        tile[local_ii][local_jj - 2] = input[PBC(ii, jj - 2, N)];
    if (threadIdx.x >= TILE_SIZE - 2 && jj + 2 < N)
        tile[local_ii][local_jj + 2] = input[PBC(ii, jj + 2, N)];

    // Diagonal halos
    if (threadIdx.x < 2 && threadIdx.y < 2 && ii >= 2 && jj >= 2)
        tile[local_ii - 2][local_jj - 2] = input[PBC(ii - 2, jj - 2, N)];
    if (threadIdx.x >= TILE_SIZE - 2 && threadIdx.y < 2 && ii >= 2 && jj + 2 < N)
        tile[local_ii - 2][local_jj + 2] = input[PBC(ii - 2, jj + 2, N)];
    if (threadIdx.x < 2 && threadIdx.y >= TILE_SIZE - 2 && ii + 2 < N && jj >= 2)
        tile[local_ii + 2][local_jj - 2] = input[PBC(ii + 2, jj - 2, N)];
    if (threadIdx.x >= TILE_SIZE - 2 && threadIdx.y >= TILE_SIZE - 2 && ii + 2 < N && jj + 2 < N)
        tile[local_ii + 2][local_jj + 2] = input[PBC(ii + 2, jj + 2, N)];

    __syncthreads();

    if ((ii < N) && (jj < N))
    {
        // Read from shared memory instead of global
        output[N * ii + jj] =
              (tile[local_ii][local_jj + 1] - (4.0 * tile[local_ii][local_jj]) + tile[local_ii][local_jj - 1])/(dx*dx)
            + (tile[local_ii + 1][local_jj] + tile[local_ii - 1][local_jj])/(dy*dy)
            + velocity/dx * (- 2.0/6.0 * tile[local_ii][local_jj - 1]
                             - 3.0/6.0 * tile[local_ii][local_jj]
                             + 6.0/6.0 * tile[local_ii][local_jj + 1]
                             - 1.0/6.0 * tile[local_ii][local_jj + 2])
            + velocity/dy * (- 2.0/6.0 * tile[local_ii - 1][local_jj]
                             - 3.0/6.0 * tile[local_ii][local_jj]
                             + 6.0/6.0 * tile[local_ii + 1][local_jj]
                             - 1.0/6.0 * tile[local_ii + 2][local_jj]);
    }
}

#endif

struct RHS_Dif_Adv_2D:public Problems_2D
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Dif_Adv_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void operator()(double* input, double* output, bool GPU)
    {
        if (GPU == true)
        {
            int num_threads = 32;
            dim3 threads(num_threads, num_threads);
            dim3 blocks((N + num_threads - 1)/num_threads, (N + num_threads - 1)/num_threads);
            
            Dif_Adv_2D_Optimised<<<blocks, threads>>>(N, dx, dy, velocity, input, output);
        }
        else
        {
            int num_threads = 32;

            #pragma omp parallel for collapse(2)
            for (int blockIdxx = 0; blockIdxx < (N + num_threads - 1)/num_threads; blockIdxx++)
            {
                for (int blockIdxy = 0; blockIdxy < (N + num_threads - 1)/num_threads; blockIdxy++)
                {
                    for (int threadIdxx = 0; threadIdxx < num_threads; threadIdxx++)
                    {
                        for (int threadIdxy = 0; threadIdxy < num_threads; threadIdxy++)
                        {
                            int ii = (blockIdxx * num_threads) + threadIdxx;
                            int jj = (blockIdxy * num_threads) + threadIdxy;

                            if ((ii < N) && (jj < N))
                            {
                                                    //? Diffusion
                                output[N*ii + jj] =   (input[PBC(ii, jj + 1, N)] - (4.0 * input[PBC(ii, jj, N)]) + input[PBC(ii, jj - 1, N)])/(dx*dx)
                                                    + (input[PBC(ii + 1, jj, N)] + input[PBC(ii - 1, jj, N)])/(dy*dy)
                                                    
                                                    //? Advection
                                                    + velocity/dx 
                                                    * (- 2.0/6.0 * input[PBC(ii, jj - 1, N)]
                                                    - 3.0/6.0 * input[PBC(ii, jj, N)]
                                                    + 6.0/6.0 * input[PBC(ii, jj + 1, N)]
                                                    - 1.0/6.0 * input[PBC(ii, jj + 2, N)])
                                                    + velocity/dy
                                                    * (- 2.0/6.0 * input[PBC(ii - 1, jj, N)]
                                                    - 3.0/6.0 * input[PBC(ii, jj, N)]
                                                    + 6.0/6.0 * input[PBC(ii + 1, jj, N)]
                                                    - 1.0/6.0 * input[PBC(ii + 2, jj, N)]);
                            }
                        }
                    }
                }
            }
            
        }
    }

    //! Destructor
    ~RHS_Dif_Adv_2D() {}
};

//? ====================================================================================== ?//