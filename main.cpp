#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

//? LeXInt Timer and functions
#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"

//? Problems
#include "Dif_Adv_2D.hpp"

//? Solvers
#include "Explicit.hpp"

using namespace std;

//? ====================================================================================== ?//

int main(int argc, char** argv)
{
    LeXInt::timer time_loop;
    time_loop.start();

    int index = atoi(argv[1]);              // N = 2^index * 2^index
    double n_cfl = atof(argv[2]);           // dt = n_cfl * dt_cfl
    double tol = atof(argv[3]);             // User-specified tolerance
    int num_time_steps = atoi(argv[4]);     // Final simulation time

    string integrator = "Explicit_Euler";   // Integrator
    integrator = argv[5];

    int output_cycle = -1;                   // How often data is written to files (if you want a movie)
    output_cycle = atoi(argv[6]);

    int num_threads;                        // # of OpenMP threads
    #pragma omp parallel
    num_threads = omp_get_num_threads();

    //! Set GPU spport to false
    bool GPU_access = atoi(argv[7]);

    //* Initialise parameters
    long long n = pow(2, index);                          // # grid points (1D)
    long long N = n*n;                                    // # grid points (2D)
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    vector<double> u_init(N);                       // Initial condition

    //* Set up X, Y arrays and initial condition
    for (int ii = 0; ii < n; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/n;
        Y[ii] = ymin + ii*(ymax - ymin)/n;
    }

    //* Initialise additional parameters
    double dx = X[12] - X[11];                              // Grid spacing
    double dy = Y[12] - Y[11];                              // Grid spacing
    double velocity = 10;                                   // Advection speed

    //* Temporal parameters
    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = min(dx/velocity, dy/velocity);         // Advection CFL
    double dt = n_cfl*min(dif_cfl, adv_cfl);                // Step size

    double time = 0;                                        // Simulation time elapsed                          
    int time_steps = 0;                                     // # time steps
    int iters = 0;                                          // # of iterations per time step
    int iters_total = 0;                                    // Total # of iterations during the simulation
    double writeTime = 0;
    double setupTime = 0;
    double startTime = 0;

    cout << endl << "N = " << N << ", tol = " << tol << ", Time steps = " << num_time_steps << endl;
    cout << "N_cfl = " << n_cfl << ", CFL: " << min(dif_cfl, adv_cfl) << ", dt = " << dt << endl << endl;

    //! Diffusion-Advection (+ Sources)
    string problem = "Diff_Adv_2D";
    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity); 

    if (problem == "Diff_Adv_2D")
    {
        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u_init[n*ii + jj] = 1 + 10*exp(-(((X[ii] + 0.5)*(X[ii] + 0.5)) + ((Y[jj] + 0.5)*(Y[jj] + 0.5)))/0.02);
            }
        }
    }
    else
    {
        cout << "Undefined problem! Please check that you have entered the correct problem." << endl;
    } 

    //! Allocate memory on CPU
    size_t N_size = N * sizeof(double);
    double* u = (double*)malloc(N_size);
    copy(u_init.begin(), u_init.end(), u);
    double* u_sol = (double*)malloc(N_size);            //* Solution vector
    double* u_temp;                                     //* Temporary vector(s) for integrators

    if (integrator == "Explicit_Euler")
    {

    }
    else if (integrator == "RK2")
    {
        u_temp = (double*)malloc(2*N_size);
    }
    else if (integrator == "RK4")
    {
        u_temp = (double*)malloc(4*N_size);
    }
    else
    {
        cout << "Incorrect integrator! Please recheck. Terminating simulations ... " << endl << endl;
        return 1;
    }

    
    


    //! Create directories (for movies)
    if (output_cycle > 0 && output_cycle < num_time_steps)
    {
        int sys_value = system(("mkdir -p ./movie/"));
        string directory = "./movie/";
    }

    
    if (GPU_access == true){
        
        //! Allocate memory on GPU

        #ifdef __CUDACC__
        double* u_D;
        cudaMalloc(&u_D, N_size);
        cudaMemcpy(u_D, u, N_size, cudaMemcpyHostToDevice);
        double* u_sol_D;
        cudaMalloc(&u_sol_D, N_size);            //* Solution vector
        double* u_temp_D;                                     //* Temporary vector(s) for integrators

        if (integrator == "Explicit_Euler")
        {

        }
        else if (integrator == "RK2")
        {
            cudaMalloc(&u_temp_D, 2*N_size);
        }
        else if (integrator == "RK4")
        {
            cudaMalloc(&u_temp_D, 4*N_size);
        }
        else
        {
            cout << "Incorrect integrator! Please recheck. Terminating simulations ... " << endl << endl;
            return 1;
        }
        #else
        ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
        exit(1);
        #endif

        setupTime = time_loop.stop();

        cout << "Running the 2D diffusion--advection problem on the GPU with the " << integrator << " integrator." << endl << endl;
        for (int nn = 0; nn < num_time_steps; nn++)
        {

            //? ------------- List of integrators ------------- ?//

            if (integrator == "Explicit_Euler")
            {
                explicit_Euler(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            }
            else if (integrator == "RK2")
            {
                RK2(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            }
            else if (integrator == "RK4")
            {
                RK4(RHS, u_D, u_sol_D, u_temp_D, dt, N, true);
            }
            else
            {
                cout << "Incorrect integrator! Please recheck. Terminating simulations ... " << endl << endl;
                return 1;
            }

            //? ----------------------------------------------- ?//

            //* Update variables
            time = time + dt;
            time_steps = time_steps + 1;
            iters_total = iters_total + iters;

            //? Update solution
            LeXInt::copy(u_sol_D, u_D, N, true);
            

            if (time_steps % 100 == 0)
            {
                // double norm = l1norm_Cpp(u_sol, N);
                // cout << "l1 norm of u   : " << norm << endl;
                cout << endl << "Time step      : " << time_steps << endl;
                cout << "Simulation time: " << time_loop.stop()-setupTime-writeTime << endl << endl;
            }

            //! Write data to files (for movies)
            if (time_steps % output_cycle == 0 && output_cycle > 0)
            {
                startTime = time_loop.stop();
                cudaMemcpy(u, u_D, N_size, cudaMemcpyDeviceToHost);
                cout << "Writing data to files at the " << time_steps << "th time step" << endl;
                string output_data = "./movie/" +  to_string(time_steps) + ".txt";
                ofstream data;
                data.open(output_data); 
                data.precision(16);
                for(int ii = 0; ii < N; ii++)
                {
                    data << u[ii] << "\n";
                }
                data.close();
                writeTime = writeTime + time_loop.stop() - startTime;
            }
        }
        cudaMemcpy(u, u_D, N_size, cudaMemcpyDeviceToHost);
    }
    else{
        setupTime = time_loop.stop();

        cout << "Running the 2D diffusion--advection problem on the CPU with the " << integrator << " integrator." << endl << endl;
        for (int nn = 0; nn < num_time_steps; nn++)
        {
            //? ------------- List of integrators ------------- ?//

            if (integrator == "Explicit_Euler")
            {
                explicit_Euler(RHS, u, u_sol, u_temp, dt, N, false);
            }
            else if (integrator == "RK2")
            {
                RK2(RHS, u, u_sol, u_temp, dt, N, false);
            }
            else if (integrator == "RK4")
            {
                RK4(RHS, u, u_sol, u_temp, dt, N, false);
            }
            else
            {
                cout << "Incorrect integrator! Please recheck. Terminating simulations ... " << endl << endl;
                return 1;
            }

            //? ----------------------------------------------- ?//

            //* Update variables
            time = time + dt;
            time_steps = time_steps + 1;
            iters_total = iters_total + iters;

            //? Update solution
            LeXInt::copy(u_sol, u, N, false);

            if (time_steps % 100 == 0)
            {
                // double norm = l1norm_Cpp(u_sol, N);
                // cout << "l1 norm of u   : " << norm << endl;
                cout << endl << "Time step      : " << time_steps << endl;
                cout << "Simulation time: " << time_loop.stop() - writeTime - setupTime << endl << endl;
            }

            //! Write data to files (for movies)
            if (time_steps % output_cycle == 0 && output_cycle > 0)
            {
                startTime = time_loop.stop();
                cout << "Writing data to files at the " << time_steps << "th time step" << endl;
                string output_data = "./movie/" +  to_string(time_steps) + ".txt";
                ofstream data;
                data.open(output_data); 
                data.precision(16);
                for(int ii = 0; ii < N; ii++)
                {
                    data << u[ii] << "\n";
                }
                data.close();
                writeTime = writeTime + time_loop.stop() - startTime;
            }
        }
    }

    
    startTime = time_loop.stop();

    //? Create directory to write simulation results/parameters
    int sys_value = system(("mkdir -p ./" + integrator + "/cores_" + to_string(num_threads)).c_str());
    string directory = "./" + integrator + "/cores_" + to_string(num_threads);
    
    string results = directory + "/Parameters.txt";
    ofstream params;
    params.open(results);
    params << "Grid points: " << N << endl;
    params << "Step size: " << dt << endl;
    params << "Tolerance (for implicit methods): " << tol << endl;
    params << "Simulation time: " << time << endl;
    params << "Total number of time steps: " << time_steps << endl;
    params << "Number of OpenMP threads: " << num_threads << endl;
    params << endl;
    params << "Total iterations (for implicit methods): " << iters_total << endl;
    params << setprecision(16) << "Runtime (s): " << time_loop.total() << endl;
    params.close();

    //? Create file to write final simulation data
    
    string final_data = directory + "/dt_cfl_" + to_string(n_cfl) + "_data.txt";
    ofstream data;
    data.open(final_data); 
    data.precision(16);
    for(int ii = 0; ii < N; ii++)
    {
        data << u[ii] << "\n";
    }
    data.close();

    writeTime = writeTime + time_loop.stop() - startTime;

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time            : " << time << endl;
    cout << "Total number of time steps : " << time_steps << endl;
    cout << "Total number of iterations : " << iters_total << endl;
    cout << "Total setup time (s)       : " << setupTime << endl;
    cout << "Total calculation time (s) : " << time_loop.total()-setupTime-writeTime << endl;
    cout << "Total writing time (s)     : " << writeTime << endl;
    cout << "Total time (s)             : " << time_loop.total() << endl;
    cout << "Number of OpenMP threads   : " << num_threads << endl;
    cout << "==================================================" << endl << endl;

    cout << "Done!" << endl;

    return 0;
}
