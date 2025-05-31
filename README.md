# Parareal
Lightning is slightly faster (sometimes a lot) than final for RK2 on GPU. In other cases it fails to consistently outperform on CPU/GPU.


1. Install Parareal
``` shell
git clone https://github.com/NewryBenson/PararealAdjust.git
```
2. Create build directory
``` shell
cd PararealAdjust && mkdir build && cd build
```
3. Compile the code
``` shell
bash ../compile.sh
```
4. Run the code
``` shell
./Parareal 8 0.5 1e-10 110 "RK4" 10
```

### Info on input arguments:
Argument 1: The executable (program) to be run (./Parareal) <br />
Argument 2: Grid points along X & Y (e.g., 8 --> 2^8 x 2^8) <br />
Argument 3: Time step size in terms of n_cfl x dt_cfl (e.g., 0.5 --> 0.5 * dt_cfl; dt_cfl is computed in main.cpp) <br />
Argument 4: Tolerance, for iterative methods (e.g., 1e-10) <br />
Argument 5: Number of time steps (e.g., 16) <br />
Argument 6: Desired integrator (e.g., "RK4")  <br />
Argument 7: Write data to files every 10 time steps for movies (set to -1 to not write data) <br />
Argument 8: 0 or 1 for CPU and GPU respectively
