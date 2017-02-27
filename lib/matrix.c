//
// Created by rahaman on 2/17/17.
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

void gauss_3d_init(double ***density, int n, double dx, double mean, double var) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++) {
        double x = i * dx;
        double y = j * dx;
        double z = k * dx;
        density[i][j][k] = 1. / pow(2 * M_PI * var, 1.5) *
                           exp(-1. * (pow(x - mean, 2) + pow(y - mean, 2) + pow(z - mean, 2)) / (2. * var));
      }
}

void matrix_3d_print(const char *fname, double ***M, int n) {
  FILE *fp = fopen(fname, "wb+");
  fwrite((const void *) &M[0][0][0], sizeof(double), (size_t ) n*n*n, fp);
  fclose(fp);
}

void matrix_3d_print_slice(const char *fname, double ***M, int n) {
  FILE *fp = fopen(fname, "wb+");
  for (int j = 0; j < n; j++)
    fwrite((const void *) &M[n/2][j][0], sizeof(double), (size_t ) n, fp);
  fclose(fp);
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

void gauss_3d_test(int n) {
  const double x_max = 2.0;
  const double var = 0.2;
  const double mean = 1.0;
  double dx = x_max / n;

  double ***M = matrix_3d_alloc(n);
  gauss_3d_init(M, n, dx, mean, var);
  matrix_3d_print_slice("/home/rahaman/scratch/gauss_out.bin", M, n);
}
