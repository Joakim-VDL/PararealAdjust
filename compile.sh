#!/bin/bash

### Compile using nvcc
nvcc -Xcompiler -fopenmp -arch=sm_80 -x cu ../main.cpp -O3 -o Parareal
