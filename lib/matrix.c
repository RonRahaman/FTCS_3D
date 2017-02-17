//
// Created by rahaman on 2/17/17.
//
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

double *** matrix_3d_alloc(int n) {
  double *   k_dim = (double *)   malloc(n * n * n * sizeof(double));
  double **  j_dim = (double **)  malloc(n * n * sizeof(double *));
  double *** i_dim = (double ***) malloc(n * sizeof(double **));

  for (int i = 0; i < n; i++) {
    i_dim[i] = &j_dim[i*n];
    for (int j = 0; j < n; j++) {
      j_dim[i*n+j] = &k_dim[i*n*n+j*n];
    }
  }

  return i_dim;
}

void matrix_3d_free(double ***M) {
  free(M[0][0]);
  free(M[0]);
  free(M);
}

void matrix_3d_test(int n) {
  double ***M = matrix_3d_alloc(n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        M[i][j][k] = 100 * (i + 1) + 10 * (j + 1) + (k + 1);


  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        printf("%3.0f ", M[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");


  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        printf("%x  ", (unsigned int) &M[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");

  matrix_3d_free(M);
}
