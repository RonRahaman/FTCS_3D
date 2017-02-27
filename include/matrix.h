//
// Created by rahaman on 2/17/17.
//

#ifndef FTCS_3D_MATRIX_H
#define FTCS_3D_MATRIX_H

#include "stdio.h"

double *** matrix_3d_alloc(int n);

void gauss_3d_init(double ***density, int n, double dx, double mean, double var);

void matrix_3d_print(FILE *fp, double ***M, int n, int dim);

void matrix_3d_free(double ***M);

void matrix_3d_test(int n);

void gauss_3d_test(int n);

#endif //FTCS_3D_MATRIX_H
