#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"
#include "Dif_Adv_2D.hpp"
#include "Explicit.hpp"

using namespace std;

// Restrict qualifiers for compiler optimizations
#if defined(__GNUC__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

int main(int argc, char** argv) {
    LeXInt::timer time_loop;
    time_loop.start();

    // Parameter validation
    if (argc < 8) {
        cerr << "Usage: " << argv[0] << " <index> <n_cfl> <tol> <steps> <integrator> <output> <gpu>\n";
        return 1;
    }

    // Parse parameters
    const int index = atoi(argv[1]);
    const double n_cfl = atof(argv[2]);
    const double tol = atof(argv[3]);
    const int num_time_steps = atoi(argv[4]);
    const string integrator = argv[5];
    const int output_cycle = atoi(argv[6]);
    const bool GPU_access = atoi(argv[7]);

    // OpenMP setup
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    const size_t cache_line_size = 64; // Adjust based on target architecture

    // Grid parameters
    const long long n = 1LL << index;
    const long long N = n * n;
    constexpr double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
    const double dx = (xmax - xmin) / n;
    const double dy = (ymax - ymin) / n;
    const double velocity = 10.0;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    // CFL calculations
    const double dif_cfl = 1.0 / (2.0 * (inv_dx2 + inv_dy2));
    const double adv_cfl = min(dx, dy) / velocity;
    const double dt = n_cfl * min(dif_cfl, adv_cfl);

    // Timing variables
    double time = 0, writeTime = 0, setupTime = 0;
    int time_steps = 0, iters_total = 0;

    cout << "\nN=" << N << ", dt=" << dt << ", threads=" << num_threads 
         << "\nCFL: " << min(dif_cfl, adv_cfl) << "\n\n";

    // Initial condition with cache-friendly layout
    vector<double, aligned_allocator<double, cache_line_size>> u_init(N);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ++ii) {
        const double x = xmin + ii * dx;
        for (int jj = 0; jj < n; ++jj) {
            const double y = ymin + jj * dy;
            const double r2 = (x + 0.5)*(x + 0.5) + (y + 0.5)*(y + 0.5);
            u_init[n*ii + jj] = 1.0 + 10.0 * exp(-r2 / 0.02);
        }
    }

    // Aligned memory allocation
    auto alloc = aligned_allocator<double, cache_line_size>();
    unique_ptr<double[], decltype(alloc)> u(alloc.allocate(N), alloc);
    unique_ptr<double[], decltype(alloc)> u_sol(alloc.allocate(N), alloc);
    unique_ptr<double[], decltype(alloc)> u_temp(nullptr, alloc);

    size_t temp_size = 0;
    if (integrator == "RK2") temp_size = 2 * N;
    else if (integrator == "RK4") temp_size = 4 * N;
    if (temp_size > 0) u_temp.reset(alloc.allocate(temp_size));

    copy(u_init.begin(), u_init.end(), u.get());

    // GPU setup
#ifdef __CUDACC__
    double *u_D = nullptr, *u_sol_D = nullptr, *u_temp_D = nullptr;
    cudaStream_t stream;
    if (GPU_access) {
        cudaStreamCreate(&stream);
        cudaMallocAsync(&u_D, N*sizeof(double), stream);
        cudaMallocAsync(&u_sol_D, N*sizeof(double), stream);
        if (temp_size > 0) cudaMallocAsync(&u_temp_D, temp_size*sizeof(double), stream);
        cudaMemcpyAsync(u_D, u.get(), N*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
#endif

    if (output_cycle > 0) system("mkdir -p ./movie/");
    setupTime = time_loop.stop();

    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);
    
    // Precompute RHS parameters
    const double adv_coeff = velocity * dt;
    const double diff_coeff = dt * (inv_dx2 + inv_dy2);

    // Main simulation loop
    for (; time_steps < num_time_steps; ++time_steps) {
        if (GPU_access) {
#ifdef __CUDACC__
            if (integrator == "Explicit_Euler") {
                explicit_Euler(RHS, u_D, u_sol_D, u_temp_D, dt, N, stream);
            } else if (integrator == "RK2") {
                RK2(RHS, u_D, u_sol_D, u_temp_D, dt, N, stream);
            } else if (integrator == "RK4") {
                RK4(RHS, u_D, u_sol_D, u_temp_D, dt, N, stream);
            }
            cudaStreamSynchronize(stream);
#endif
        } else {
            // CPU optimized path with SIMD
            if (integrator == "Explicit_Euler") {
                #pragma omp parallel for simd schedule(static) aligned(u, u_sol: cache_line_size)
                for (long long i = 0; i < N; ++i) {
                    u_sol[i] = u[i] + dt * RHS(u[i]);
                }
            } else if (integrator == "RK2") {
                // RK2 optimized implementation
                #pragma omp parallel for simd schedule(static) aligned(u, u_sol, u_temp: cache_line_size)
                for (long long i = 0; i < N; ++i) {
                    const double k1 = RHS(u[i]);
                    u_temp[i] = u[i] + 0.5 * dt * k1;
                }
                #pragma omp parallel for simd schedule(static) aligned(u_temp, u_sol: cache_line_size)
                for (long long i = 0; i < N; ++i) {
                    const double k2 = RHS(u_temp[i]);
                    u_sol[i] = u[i] + dt * k2;
                }
            }
        }

        time += dt;
        
        // Pointer swap optimized
        if (GPU_access) {
#ifdef __CUDACC__
            std::swap(u_D, u_sol_D);
#else
            swap(u, u_sol);
#endif
        }

        // Output handling
        if (output_cycle > 0 && (time_steps % output_cycle) == 0) {
            const auto write_start = time_loop.stop();
#ifdef __CUDACC__
            if (GPU_access) {
                cudaMemcpyAsync(u.get(), u_D, N*sizeof(double), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
#endif
            ofstream data("./movie/" + to_string(time_steps) + ".bin", ios::binary);
            data.write(reinterpret_cast<char*>(u.get()), N*sizeof(double));
            writeTime += time_loop.stop() - write_start;
        }

        if (time_steps % 100 == 0) {
            cout << "Step: " << time_steps << "\tTime: " 
                 << time_loop.stop() - setupTime - writeTime << "s\n";
        }
    }

    // Final output
    const auto final_write_start = time_loop.stop();
    const string output_dir = "./" + integrator + "/cores_" + to_string(num_threads);
    system(("mkdir -p " + output_dir).c_str());

#ifdef __CUDACC__
    if (GPU_access) {
        cudaMemcpyAsync(u.get(), u_D, N*sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
#endif

    ofstream params(output_dir + "/params.bin", ios::binary);
    params.write(reinterpret_cast<const char*>(&time), sizeof(time));
    params.write(reinterpret_cast<const char*>(&dt), sizeof(dt));
    params.write(reinterpret_cast<const char*>(&N), sizeof(N));

    ofstream final_data(output_dir + "/final.bin", ios::binary);
    final_data.write(reinterpret_cast<char*>(u.get()), N*sizeof(double));
    writeTime += time_loop.stop() - final_write_start;

    // Cleanup
#ifdef __CUDACC__
    if (GPU_access) {
        cudaFreeAsync(u_D, stream);
        cudaFreeAsync(u_sol_D, stream);
        if (u_temp_D) cudaFreeAsync(u_temp_D, stream);
        cudaStreamDestroy(stream);
    }
#endif

    time_loop.stop();

    // Final report
    cout << "\n==============================================\n"
         << "Total Time: " << fixed << setprecision(2) << time_loop.total() << "s\n"
         << "Compute: " << time_loop.total() - setupTime - writeTime << "s\n"
         << "I/O: " << writeTime << "s\n"
         << "Throughput: " << 1e-6 * N * time_steps / (time_loop.total() - writeTime) 
         << " MCell/s\n"
         << "==============================================\n";

    return 0;
}