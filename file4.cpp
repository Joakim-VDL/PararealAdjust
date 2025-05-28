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

int main(int argc, char** argv) {
    // Initialize timers
    LeXInt::timer time_loop;
    time_loop.start();

    // Parameter parsing with better error checking
    if (argc < 8) {
        cerr << "Usage: " << argv[0] << " <index> <n_cfl> <tol> <num_time_steps> <integrator> <output_cycle> <GPU_access>\n";
        return 1;
    }

    const int index = atoi(argv[1]);
    const double n_cfl = atof(argv[2]);
    const double tol = atof(argv[3]);
    const int num_time_steps = atoi(argv[4]);
    const string integrator = argv[5];
    const int output_cycle = atoi(argv[6]);
    const bool GPU_access = atoi(argv[7]);

    // OpenMP setup with better thread management
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    omp_set_num_threads(num_threads);

    // Grid setup with optimized memory layout
    const long long n = 1LL << index;
    const long long N = n * n;
    constexpr double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
    const double dx = (xmax - xmin) / n;
    const double dy = (ymax - ymin) / n;
    const double velocity = 10.0;

    // CFL conditions with better numerical stability
    const double dif_cfl = (dx * dx * dy * dy) / (2.0 * (dx * dx + dy * dy));
    const double adv_cfl = min(dx, dy) / velocity;
    const double dt = n_cfl * min(dif_cfl, adv_cfl);

    // Timing and tracking variables
    double time = 0;
    int time_steps = 0;
    int iters_total = 0;
    double writeTime = 0;
    double setupTime = 0;
    double startTime = 0;

    cout << "\nN = " << N << ", tol = " << tol << ", Time steps = " << num_time_steps << endl;
    cout << "N_cfl = " << n_cfl << ", CFL: " << min(dif_cfl, adv_cfl) << ", dt = " << dt << endl << endl;

    // Optimized initial condition setup
    vector<double> u_init(N);
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ++ii) {
        const double x = xmin + ii * dx;
        for (int jj = 0; jj < n; ++jj) {
            const double y = ymin + jj * dy;
            const double r2 = (x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5);
            u_init[n * ii + jj] = 1.0 + 10.0 * exp(-r2 / 0.02);
        }
    }

    // Memory allocation with better alignment
    unique_ptr<double[]> u(new double[N]);
    unique_ptr<double[]> u_sol(new double[N]);
    unique_ptr<double[]> u_temp;

    size_t temp_size = 0;
    if (integrator == "RK2") {
        temp_size = 2 * N;
    } else if (integrator == "RK4") {
        temp_size = 4 * N;
    }
    if (temp_size > 0) {
        u_temp.reset(new double[temp_size]);
    }

    copy(u_init.begin(), u_init.end(), u.get());

    // GPU setup with better error handling
    #ifdef __CUDACC__
    double *u_D = nullptr, *u_sol_D = nullptr, *u_temp_D = nullptr;
    if (GPU_access) {
        cudaError_t err;
        if ((err = cudaMalloc(&u_D, N * sizeof(double))) {
            cerr << "CUDA error (u_D): " << cudaGetErrorString(err) << endl;
            return 1;
        }
        if ((err = cudaMalloc(&u_sol_D, N * sizeof(double)))) {
            cerr << "CUDA error (u_sol_D): " << cudaGetErrorString(err) << endl;
            cudaFree(u_D);
            return 1;
        }
        if (temp_size > 0) {
            if ((err = cudaMalloc(&u_temp_D, temp_size * sizeof(double)))) {
                cerr << "CUDA error (u_temp_D): " << cudaGetErrorString(err) << endl;
                cudaFree(u_D);
                cudaFree(u_sol_D);
                return 1;
            }
        }
        cudaMemcpy(u_D, u.get(), N * sizeof(double), cudaMemcpyHostToDevice);
    }
    #endif

    // Create output directory if needed
    if (output_cycle > 0) {
        system("mkdir -p ./movie/");
    }

    setupTime = time_loop.stop();

    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);
    cout << "\nN=" << N << ", dt=" << dt
         << ", using " << integrator << " on "
         << (GPU_access ? "GPU" : "CPU") << " with " << num_threads << " threads\n\n";

    // Main simulation loop with optimized structure
    for (; time_steps < num_time_steps; ++time_steps) {
        if (GPU_access) {
            #ifdef __CUDACC__
            if (integrator == "Explicit_Euler") {
                explicit_Euler(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            } else if (integrator == "RK2") {
                RK2(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            } else if (integrator == "RK4") {
                RK4(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            }
            cudaDeviceSynchronize();
            #endif
        } else {
            if (integrator == "Explicit_Euler") {
                explicit_Euler(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            } else if (integrator == "RK2") {
                RK2(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            } else if (integrator == "RK4") {
                RK4(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            }
        }

        time += dt;
        iters_total += iters;

        // Progress reporting
        if (time_steps % 100 == 0) {
            cout << "Time step: " << time_steps 
                 << ", Simulation time: " << time_loop.stop()-setupTime-writeTime << "s\n";
        }

        // Output for visualization
        if (output_cycle > 0 && time_steps % output_cycle == 0) {
            startTime = time_loop.stop();
            ofstream data("./movie/" + to_string(time_steps) + ".txt");
            data.precision(16);

            if (GPU_access) {
                #ifdef __CUDACC__
                cudaMemcpy(u.get(), u_D, N * sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            }

            for (long long i = 0; i < N; ++i) {
                data << u[i] << '\n';
            }
            writeTime += time_loop.stop() - startTime;
        }

        // Buffer swap optimized
        if (GPU_access) {
            #ifdef __CUDACC__
            swap(u_D, u_sol_D);
            #endif
        } else {
            swap(u, u_sol);
        }
    }

    // Final output with better organization
    startTime = time_loop.stop();
    const string output_dir = "./" + integrator + "/cores_" + to_string(num_threads);
    system(("mkdir -p " + output_dir).c_str());

    ofstream params(output_dir + "/Parameters.txt");
    params << fixed << setprecision(6)
           << "Grid points: " << N << "\n"
           << "Step size: " << dt << "\n"
           << "Tolerance: " << tol << "\n"
           << "Simulation time: " << time << "\n"
           << "Time steps: " << time_steps << "\n"
           << "Threads: " << num_threads << "\n"
           << "Total iterations: " << iters_total << "\n"
           << "Runtime (s): " << time_loop.total() << "\n";

    if (GPU_access) {
        #ifdef __CUDACC__
        cudaMemcpy(u.get(), u_D, N * sizeof(double), cudaMemcpyDeviceToHost);
        #endif
    }

    ofstream final_data(output_dir + "/dt_cfl_" + to_string(n_cfl) + "_data.txt");
    final_data.precision(16);
    for (long long i = 0; i < N; ++i) {
        final_data << u[i] << '\n';
    }
    writeTime += time_loop.stop() - startTime;

    // Cleanup
    #ifdef __CUDACC__
    if (GPU_access) {
        cudaFree(u_D);
        cudaFree(u_sol_D);
        if (u_temp_D) cudaFree(u_temp_D);
    }
    #endif

    time_loop.stop();

    // Enhanced timing output
    cout << "\n==================================================\n"
         << "Simulation time            : " << fixed << setprecision(2) << time << "\n"
         << "Total time steps           : " << time_steps << "\n"
         << "Total iterations           : " << iters_total << "\n"
         << "Setup time (s)             : " << setupTime << "\n"
         << "Computation time (s)       : " << time_loop.total()-setupTime-writeTime << "\n"
         << "I/O time (s)               : " << writeTime << "\n"
         << "Total runtime (s)          : " << time_loop.total() << "\n"
         << "Threads used               : " << num_threads << "\n"
         << "==================================================\n\n";

    return 0;
}