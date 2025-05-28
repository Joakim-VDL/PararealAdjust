#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <iostream>
#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"
#include "Dif_Adv_2D.hpp"
#include "Explicit.hpp"

using namespace std;

int main(int argc, char** argv) {
    LeXInt::timer total_timer;
    total_timer.start();

    // Parameters
    const int index = atoi(argv[1]);
    const double n_cfl = atof(argv[2]);
    const double tol = atof(argv[3]);
    const int num_time_steps = atoi(argv[4]);
    const string integrator = argv[5];
    const int output_cycle = atoi(argv[6]);
    const bool GPU_access = atoi(argv[7]);

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Grid setup
    const long long n = 1LL << index;
    const long long N = n * n;
    constexpr double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
    const double dx = (xmax - xmin) / n;
    const double dy = (ymax - ymin) / n;
    const double velocity = 10.0;

    const double dif_cfl = (dx * dx * dy * dy) / (2.0 * (dx * dx + dy * dy));
    const double adv_cfl = min(dx, dy) / velocity;
    const double dt = n_cfl * min(dif_cfl, adv_cfl);

    // Initial condition
    vector<double> u_init(N);
    for (int ii = 0; ii < n; ++ii) {
        const double x = xmin + ii * dx;
        for (int jj = 0; jj < n; ++jj) {
            const double y = ymin + jj * dy;
            const double r2 = (x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5);
            u_init[ii * n + jj] = 1.0 + 10.0 * exp(-r2 / 0.02);
        }
    }

    // Memory allocation
    unique_ptr<double[]> u(new double[N]);
    unique_ptr<double[]> u_sol(new double[N]);
    unique_ptr<double[]> u_temp;

    if (integrator == "RK2") {
        u_temp.reset(new double[2 * N]);
    } 
    else if (integrator == "RK4") {
        u_temp.reset(new double[4 * N]);
    } 
    else if (integrator != "Explicit_Euler") {
        cerr << "Invalid integrator: " << integrator << endl;
        return 1;
    }

    copy(u_init.begin(), u_init.end(), u.get());

    // GPU setup
    #ifdef __CUDACC__
    double *u_D = nullptr, *u_sol_D = nullptr, *u_temp_D = nullptr;
    if (GPU_access) {
        cudaMalloc(&u_D, N * sizeof(double));
        cudaMalloc(&u_sol_D, N * sizeof(double));
        if (integrator == "RK2") {
            cudaMalloc(&u_temp_D, 2 * N * sizeof(double));
        } 
        else if (integrator == "RK4") {
            cudaMalloc(&u_temp_D, 4 * N * sizeof(double));
        }
        cudaMemcpy(u_D, u.get(), N * sizeof(double), cudaMemcpyHostToDevice);
    }
    #else
    if (GPU_access) {
        cerr << "GPU support requested but not compiled with CUDA!" << endl;
        return 1;
    }
    #endif

    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);
    const double setupTime = total_timer.stop();

    cout << "\nStarting simulation with N=" << N << ", dt=" << dt
         << ", using " << integrator << " on "
         << (GPU_access ? "GPU" : "CPU") << "\n\n";

    // Create output directory
    if (output_cycle > 0) {
        if (system("mkdir -p ./movie/") != 0) {
            cerr << "Failed to create movie directory!" << endl;
        }
    }

    // Simulation loop
    double time = 0.0;
    double writeTime = 0.0;
    int time_steps = 0;

    for (; time_steps < num_time_steps; ++time_steps) {
        if (GPU_access) {
            #ifdef __CUDACC__
            if (integrator == "Explicit_Euler") {
                explicit_Euler(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            } 
            else if (integrator == "RK2") {
                RK2(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            } 
            else if (integrator == "RK4") {
                RK4(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            }
            #endif
        } 
        else {
            if (integrator == "Explicit_Euler") {
                explicit_Euler(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            } 
            else if (integrator == "RK2") {
                RK2(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            } 
            else if (integrator == "RK4") {
                RK4(RHS, u.get(), u_sol.get(), u_temp.get(), dt, N, false);
            }
        }

        time += dt;

        if (time_steps % 100 == 0) {
            cout << "Time step: " << time_steps 
                 << ", Simulation time: " << total_timer.stop() - setupTime - writeTime << "s\n";
        }

        if (output_cycle > 0 && time_steps % output_cycle == 0) {
            const double startWrite = total_timer.stop();
            ofstream data("./movie/" + to_string(time_steps) + ".txt");
            data.precision(16);

            if (GPU_access) {
                #ifdef __CUDACC__
                cudaMemcpy(u.get(), u_D, N * sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            }

            for (long long i = 0; i < N; ++i) {
                data << u[i] << "\n";
            }

            writeTime += total_timer.stop() - startWrite;
        }

        if (GPU_access) {
            #ifdef __CUDACC__
            swap(u_D, u_sol_D);
            #endif
        } 
        else {
            swap(u, u_sol);
        }
    }

    // Final output
    const string output_dir = "./" + integrator + "/cores_" + to_string(num_threads);
    if (system(("mkdir -p " + output_dir).c_str()) != 0) {
        cerr << "Failed to create output directory!" << endl;
    }

    ofstream params(output_dir + "/Parameters.txt");
    params << "Grid points: " << N << "\n"
           << "Step size: " << dt << "\n"
           << "Tolerance: " << tol << "\n"
           << "Simulation time: " << time << "\n"
           << "Time steps: " << time_steps << "\n"
           << "Threads: " << num_threads << "\n"
           << "Runtime: " << total_timer.total() << "s\n";

    if (GPU_access) {
        #ifdef __CUDACC__
        cudaMemcpy(u.get(), u_D, N * sizeof(double), cudaMemcpyDeviceToHost);
        #endif
    }

    ofstream final_data(output_dir + "/dt_cfl_" + to_string(n_cfl) + "_data.txt");
    final_data.precision(16);
    for (long long i = 0; i < N; ++i) {
        final_data << u[i] << "\n";
    }

    #ifdef __CUDACC__
    if (GPU_access) {
        cudaFree(u_D);
        cudaFree(u_sol_D);
        if (u_temp_D) cudaFree(u_temp_D);
    }
    #endif

    const double total_time = total_timer.total();
    cout << "\n========================================\n"
         << "  Simulation completed successfully\n"
         << "  Simulation time: " << time << "\n"
         << "  Total number of time steps: " << time_steps << "\n"
         << "  total setup time: " << setupTime << "s\n"
         << "  Total computing time: " << total_time - setupTime - writeTime << "s\n"
         << "  Total writing time: " << writeTime << "s\n"
         << "  Total time: " << total_time << "s\n"
         << "  Threads: " << num_threads << "\n"
         << "========================================\n";

    return 0;
}
