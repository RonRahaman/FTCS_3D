//
// Created by rahaman on 2/17/17.
//

#ifndef FTCS_3D_MATRIX_H
#define FTCS_3D_MATRIX_H

#include "stdio.h"
// Allocates a nx by ny matrix with continuously-allocated columns
double ** matrix_2d_alloc(int nx, int ny);

// Allocates a continuous n*n*n matrix
double *** matrix_3d_alloc(int n);

// Initializes a Gaussian distribution in a 2D domain
void matrix_2d_gauss_init(double **density, int nx, int ny, double dx, double dy, double mean, double var);

// Frees a matrix as allocated by matrix_2d_alloc().
void matrix_2d_free(double **M);

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
