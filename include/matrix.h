//
// Created by rahaman on 2/17/17.
//

#ifndef FTCS_3D_MATRIX_H
#define FTCS_3D_MATRIX_H

#include "stdio.h"

// Allocates a continuous n*n*n matrix
double *** matrix_3d_alloc(int n);

// Fills 'density', an n*n*n matrix, with a 3D gaussian
void gauss_3d_init(double ***density, int n, double dx, double mean, double var);

// Prints all elements of an n*n*n matrix to a binary file
void matrix_3d_print(const char *fname, double ***M, int n);

// Prints a 2D slice of the n*n*n matrix to a binary file
void matrix_3d_print_slice(const char *fname, double ***M, int n);

// Frees a 3D matrix, as allocated by matrix_3d_alloc
void matrix_3d_free(double ***M);

void matrix_3d_test(int n);

void gauss_3d_test(int n);

#endif //FTCS_3D_MATRIX_H
