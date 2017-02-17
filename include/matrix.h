//
// Created by rahaman on 2/17/17.
//

#ifndef FTCS_3D_MATRIX_H
#define FTCS_3D_MATRIX_H


// Allocates a square n x n matrix of doubles with continuous rows.
double ** matrix_2d_alloc(int n);

// Frees a matrix as allocated by dmatrix().
void matrix_2d_free(double **M);

double *** matrix_3d_alloc(int n);

void matrix_3d_free(double ***M);

void matrix_3d_test(int n);

#endif //FTCS_3D_MATRIX_H
