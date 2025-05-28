#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"
#include "Dif_Adv_2D.hpp"
#include "Explicit.hpp"

using namespace std;

int main(int argc, char** argv) {
    LeXInt::timer time_loop;
    time_loop.start();

    // ======================== Parameter Initialization ======================== //
    const int index = atoi(argv[1]);         // Grid size exponent (N = 2^index)
    const double n_cfl = atof(argv[2]);      // CFL multiplier
    const double tol = atof(argv[3]);        // Solver tolerance
    const int num_time_steps = atoi(argv[4]);// Total simulation steps
    const string integrator = argv[5];       // Integration method
    const int output_cycle = atoi(argv[6]);  // Output frequency
    const bool GPU_access = atoi(argv[7]);   // GPU acceleration flag

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // ======================== Grid Setup ======================== //
    const long long n = 1LL << index;        // 2^index grid points per dimension
    const long long N = n * n;               // Total grid points
    constexpr double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
    const double dx = (xmax - xmin) / n;
    const double dy = (ymax - ymin) / n;
    const double velocity = 10.0;

    // CFL conditions
    const double dif_cfl = (dx*dx * dy*dy) / (2.0*(dx*dx + dy*dy));
    const double adv_cfl = min(dx, dy) / velocity;
    const double dt = n_cfl * min(dif_cfl, adv_cfl);

    //time setup
    double time = 0;                                        // Simulation time elapsed                          
    int time_steps = 0;                                     // # time steps
    //int iters = 0;                                          // # of iterations per time step
    //int iters_total = 0;                                    // Total # of iterations during the simulation
    double writeTime = 0;
    double setupTime = 0;
    double startTime = 0;

    // ======================== Memory Allocation ======================== //
    const size_t N_size = N * sizeof(double);
    vector<double> u_init(N);
    
    // Initialize grid and initial condition in a single pass
    for (int ii = 0; ii < n; ++ii) {
        const double x = xmin + ii * dx;
        for (int jj = 0; jj < n; ++jj) {
            const double y = ymin + jj * dy;
            const double r2 = (x + 0.5)*(x + 0.5) + (y + 0.5)*(y + 0.5);
            u_init[ii*n + jj] = 1.0 + 10.0 * exp(-r2 / 0.02);
        }
    }

    // Smart pointers for automatic memory management
    unique_ptr<double[]> u(new double[N]);
    unique_ptr<double[]> u_sol(new double[N]);
    unique_ptr<double[]> u_temp;
    
    // Determine temporary storage size based on integrator
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

    // ======================== GPU Setup ======================== //
    #ifdef __CUDACC__
    double *u_D = nullptr, *u_sol_D = nullptr, *u_temp_D = nullptr;
    if (GPU_access) {
        cudaMalloc(&u_D, N_size);
        cudaMalloc(&u_sol_D, N_size);
        if (integrator == "RK2") {
            cudaMalloc(&u_temp_D, 2 * N_size);
        } 
        else if (integrator == "RK4") {
            cudaMalloc(&u_temp_D, 4 * N_size);
        }
        cudaMemcpy(u_D, u.get(), N_size, cudaMemcpyHostToDevice);
    }
    #else
    if (GPU_access) {
        cerr << "GPU support requested but not compiled with CUDA!" << endl;
        return 1;
    }
    #endif
    setupTime = time_loop.stop();


    // ======================== Simulation Setup ======================== //
    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);
    cout << "\nN=" << N << ", dt=" << dt 
         << ", using " << integrator << " on " 
         << (GPU_access ? "GPU" : "CPU") << "\n\n";

    // ======================== Main Simulation Loop ======================== //
    

    // Create output directory if needed (with error checking)
    if (output_cycle > 0) {
        int result = system("mkdir -p ./movie/");
        if (result != 0) {
            cerr << "Failed to create movie directory!" << endl;
        }
    }

    for (; time_steps < num_time_steps; ++time_steps) {
        // Execute the appropriate integrator
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

        //update time
        time = time + dt;
        time_steps = time_steps + 1;

        // Progress reporting
        if (time_steps % 100 == 0) {
            cout << "Time step: " << time_steps 
                 << ", Simulation time: " << time_loop.stop()-setupTime-writeTime << "s\n";
        }

        // Periodic output
        if (output_cycle > 0 && time_steps % output_cycle == 0) {
            ofstream data("./movie/" + to_string(time_steps) + ".txt");
            data.precision(16);
            
            if (GPU_access) {
                #ifdef __CUDACC__
                cudaMemcpy(u.get(), u_D, N_size, cudaMemcpyDeviceToHost);
                #endif
            }
            
            for (int ii = 0; ii < N; ++ii) {
                data << u[ii] << "\n";
            }
            writeTime = writeTime + time_loop.stop() - startTime;
        }

        // Swap solution pointers
        if (GPU_access) {
            #ifdef __CUDACC__
            swap(u_D, u_sol_D);
            #endif
        } 
        else {
            swap(u, u_sol);
        }
    }
    startTime = time_loop.stop();
    // ======================== Final Output ======================== //
    const string output_dir = "./" + integrator + "/cores_" + to_string(num_threads);
    int dir_result = system(("mkdir -p " + output_dir).c_str());
    if (dir_result != 0) {
        cerr << "Failed to create output directory!" << endl;
    }

    // Write parameters
    ofstream params(output_dir + "/Parameters.txt");
    params << "Grid points: " << N << "\n"
           << "Step size: " << dt << "\n"
           << "Tolerance: " << tol << "\n"
           << "Simulation time: " << time << "\n"
           << "Time steps: " << time_steps << "\n"
           << "Threads: " << num_threads << "\n"
           << "Runtime: " << time_loop.total() << "s\n";
    
    // Write final data
    if (GPU_access) {
        #ifdef __CUDACC__
        cudaMemcpy(u.get(), u_D, N_size, cudaMemcpyDeviceToHost);
        #endif
    }
    
    ofstream final_data(output_dir + "/dt_cfl_" + to_string(n_cfl) + "_data.txt");
    final_data.precision(16);
    for (int ii = 0; ii < N; ++ii) {
        final_data << u[ii] << "\n";
    }
    writeTime = writeTime + time_loop.stop() - startTime;

    // ======================== Cleanup ======================== //
    #ifdef __CUDACC__
    if (GPU_access) {
        cudaFree(u_D);
        cudaFree(u_sol_D);
        if (u_temp_D) cudaFree(u_temp_D);
    }
    #endif

    time_loop.stop();
    // ======================== Performance Summary ======================== //
    cout << "\n========================================\n"
         << "  Simulation completed successfully\n"
         << "  Simulation time: " << time << "\n"
         << "  Total number of time steps: " << time_steps << "\n"
         << "  total setup time: " << setupTime << "s\n"
         << "  Total computing time: " << time_loop.total()-setupTime-writeTime << "s\n"
         << "  Total writing time: " << writeTime << "s\n"
         << "  Total time: " << time_loop.total() << "s\n"
         << "  Threads: " << num_threads << "\n"
         << "========================================\n";

    return 0;
}