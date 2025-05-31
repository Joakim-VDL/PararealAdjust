#pragma once

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>

// For restrict keyword portability
#if defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

//? ----------------------------------------------------------
//?
//? Description:
//?     A pleothera of functions are defined here that
//?     are used throughout the code.
//?
//? ----------------------------------------------------------

//! ======================================================================================== !//

//! Return double !//

double l1norm_Cpp(const double* RESTRICT vector, size_t N)
{
    double norm = 0.0;
#pragma omp parallel for reduction(+:norm)
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        norm += std::abs(vector[ii]);
    }
    return norm;
}

double l2norm_Cpp(const double* RESTRICT vector, size_t N)
{
    double norm = 0.0;
#pragma omp parallel for reduction(+:norm)
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        norm += vector[ii] * vector[ii];
    }
    return std::sqrt(norm);
}

double factorial(int number)
{
    double fact = 1.0;
    int abs_number = std::abs(number);
    for(int ii = 1; ii <= abs_number; ii++)
    {
        fact *= ii;
    }
    return fact;
}

//! ======================================================================================== !//

//! Return double* !//

//? ones(y) = (y[0:N] =) 1.0
void ones_Cpp(double* RESTRICT x, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        x[ii] = 1.0;
    }
}

//? eigen_ones(y): y[:] = 0, y[0] = 1
void eigen_ones_Cpp(double* RESTRICT x, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        x[ii] = 0.0;
    }
    if (N > 0)
        x[0] = 1.0;
}

//? y = x
void copy_Cpp(const double* RESTRICT x, double* RESTRICT y, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        y[ii] = x[ii];
    }
}

//? y = ax
void axpby_Cpp(double a, const double* RESTRICT x, double* RESTRICT y, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        y[ii] = a * x[ii];
    }
}

//? z = ax + by
void axpby_Cpp(double a, const double* RESTRICT x,
               double b, const double* RESTRICT y,
               double* RESTRICT z, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        z[ii] = a * x[ii] + b * y[ii];
    }
}

//? w = ax + by + cz
void axpby_Cpp(double a, const double* RESTRICT x,
               double b, const double* RESTRICT y,
               double c, const double* RESTRICT z,
               double* RESTRICT w, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        w[ii] = a * x[ii] + b * y[ii] + c * z[ii];
    }
}

//? v = ax + by + cz + dw
void axpby_Cpp(double a, const double* RESTRICT x,
               double b, const double* RESTRICT y,
               double c, const double* RESTRICT z,
               double d, const double* RESTRICT w,
               double* RESTRICT v, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        v[ii] = a * x[ii] + b * y[ii] + c * z[ii] + d * w[ii];
    }
}

//? u = ax + by + cz + dw + ev
void axpby_Cpp(double a, const double* RESTRICT x,
               double b, const double* RESTRICT y,
               double c, const double* RESTRICT z,
               double d, const double* RESTRICT w,
               double e, const double* RESTRICT v,
               double* RESTRICT u, size_t N)
{
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(N); ii++)
    {
        u[ii] = a * x[ii] + b * y[ii] + c * z[ii] + d * w[ii] + e * v[ii];
    }
}

//! ======================================================================================== !//
