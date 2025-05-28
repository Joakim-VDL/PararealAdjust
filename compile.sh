#!/bin/bash

### Compile using nvcc
nvcc -Xcompiler -fopenmp -arch=sm_80 -x cu ../file3.cpp -O3 -o Parareal
