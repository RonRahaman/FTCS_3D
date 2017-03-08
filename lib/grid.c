//
// Created by Ron Rahaman on 3/6/17.
//

#include "grid.h"
#include "mpi.h"

void  set_2d_cart_bounds(double **domain, Grid_info grid[], int neighbors[][2], double bound_val) {

  // Set boundary conditions...
  // ... on the -x edge
  if (neighbors[0][0] == MPI_PROC_NULL) {
    int i = 0;
    for (int j = 0; j <= grid[1].n+1; j++)
      domain[i][j] = bound_val;
  }
  // ... on the +x edge
  if (neighbors[0][1] == MPI_PROC_NULL) {
    int i = grid[0].n + 1;
    for (int j = 0; j <= grid[1].n+1; j++)
      domain[i][j] = bound_val;
  }
  // ... on the -y edge
  if (neighbors[1][0] == MPI_PROC_NULL) {
    int j = 0;
    for (int i = 0; i <= grid[0].n+1; i++)
      domain[i][j] = bound_val;
  }
  // ... on the +y edge
  if (neighbors[1][1] == MPI_PROC_NULL) {
    int j = grid[1].n + 1;
    for (int i = 0; i <= grid[0].n+1; i++)
      domain[i][j] = bound_val;
  }

}