//
// Created by Ron Rahaman on 3/6/17.
//

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "matrix.h"
#include "cart.h"

int main (int argc, char *argv[]) {
  const int ndim = 2;
  const int global_grid_n = 12;
  const double bound_val = 0.0;

  int world_rank, world_size;      // The rank, size in MPI_COMM_WORLD

  MPI_Comm cart_comm;       // The cartesian communicator
  int cart_rank, cart_size; // Size, rank in the cartesian communicator
  int cart_period[ndim];    // Not using periodic boundary conditions in Cartesian communicator
  int cart_dim[ndim];       // The dimensions of the Cartesian communicator
  int cart_coord[ndim];     // This proc's coordinates in the Cartesian communicator
  int cart_nbr[ndim][2];    // This proc's neighbors along each dimension

  Grid_info global_grid[ndim], grid[ndim];
  MPI_Datatype edge[ndim];

  double ** M;

  const char outfile[] = "cart_demo.out";

  MPI_Init(&argc, &argv);

  // Get info about COMM_WORLD
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Let MPI figure out the best dimensions for the Cartesian comm
  for (int i = 0; i < ndim; i++)
    cart_dim[i] = 0;
  MPI_Dims_create(world_size, ndim, cart_dim);

  // Creates the communicator
  int reorder = 1;                          // Means MPI is allowed to reorder ranks in Cart comm
  for (int i = 0; i < ndim; i++)
    cart_period[i] = 0;                     // Means we're not using periodic bounds
  MPI_Cart_create(MPI_COMM_WORLD, ndim, cart_dim, cart_period, reorder, &cart_comm);

  // Get basic info about the new communicator
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Comm_size(cart_comm, &cart_size);

  // Get the location and neighbors of this proc in the Cartesian communicator
  MPI_Cart_coords(cart_comm, cart_rank, ndim, cart_coord);
  for (int i = 0; i < ndim; i++)
    MPI_Cart_shift(cart_comm, i, 1, &cart_nbr[i][0], &cart_nbr[i][1]);

  if (world_rank == 0) {

    // Check consistency w/ cart comm
    printf("The Cartesian communicator is %d x %d procs\n", cart_dim[0], cart_dim[1]);
    printf("The spatial domain is %d x %d gripoints\n", global_grid_n, global_grid_n);
    if (global_grid_n % cart_dim[0] != 0 || global_grid_n % cart_dim[1] != 0) {
      fprintf(stderr, "ERROR: The number of gridpoints are not evenly divisible by the number of processes\n");
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    }
  }

  // Set info about the global grid
  for (int i = 0; i < ndim; i++)
    global_grid[i].n = global_grid_n;

  // Set info about the local (this proc's) grid
  for (int i = 0; i < ndim; i++)
    grid[i].n = global_grid[i].n / cart_dim[i];

  // Set up datatypes for rows and columns.
  MPI_Type_contiguous(grid[1].n, MPI_DOUBLE, &edge[0]);
  MPI_Type_commit(&edge[0]);

  MPI_Type_vector(grid[0].n, 1, grid[1].n + 2, MPI_DOUBLE, &edge[1]);
  MPI_Type_commit(&edge[1]);

  M = matrix_2d_alloc(grid[0].n + 2, grid[1].n + 2);

  // ==============================================================================================
  // Set initial conditions
  // ==============================================================================================
  for (int i = 0; i <= grid[0].n+1; i++)
    for (int j = 0; j <= grid[1].n+1; j++)
      M[i][j] = -1;
  for (int i = 1; i <= grid[0].n; i++)
    for (int j = 1; j <= grid[1].n; j++)
      M[i][j] = cart_rank+1;

  // ==============================================================================================
  // Set boundary conditions
  // ==============================================================================================

  // ... on the -x edge
  if (cart_nbr[0][0] == MPI_PROC_NULL) {
    int i = 0;
    for (int j = 0; j <= grid[1].n+1; j++)
      M[i][j] = bound_val;
  }
  // ... on the +x edge
  if (cart_nbr[0][1] == MPI_PROC_NULL) {
    int i = grid[0].n + 1;
    for (int j = 0; j <= grid[1].n+1; j++)
      M[i][j] = bound_val;
  }
  // ... on the -y edge
  if (cart_nbr[1][0] == MPI_PROC_NULL) {
    int j = 0;
    for (int i = 0; i <= grid[0].n+1; i++)
      M[i][j] = bound_val;
  }
  // ... on the +y edge
  if (cart_nbr[1][1] == MPI_PROC_NULL) {
    int j = grid[1].n + 1;
    for (int i = 0; i <= grid[0].n+1; i++)
      M[i][j] = bound_val;
  }

  // ==============================================================================================
  // Print subdomains before neighbor exchange
  // ==============================================================================================

  for (int k = 0; k < cart_size; k++) {
    if (cart_rank == k) {
      if (k == 0) {
        printf("\n========================\n");
        printf("Before neighbor exchange");
        printf("\n========================\n\n");
        fflush(stdout);
      }
      printf("Cart rank is %d\n", cart_rank);
      printf("Cart coords are [%d, %d]\n", cart_coord[0], cart_coord[1]);
      printf("Cart nbrs are [[%d, %d], [%d, %d]\n", cart_nbr[0][0], cart_nbr[0][1], cart_nbr[1][0], cart_nbr[1][1]);
      for (int i = 0; i <= grid[0].n+1; i++) {
        for (int j = 0; j <= grid[1].n+1; j++) {
          printf("%2.0f ", M[i][j]);
        }
        printf("\n");
      }
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ==============================================================================================
  // Neighbor exchanges
  // ==============================================================================================

  // ... in the +x direction
  {
    double *sendbuf = &M[1][1];
    double *recvbuf = &M[grid[0].n + 1][1];
    MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][0], 99, recvbuf, 1, edge[0], cart_nbr[0][1], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the -x direction
  {
    double *sendbuf = &M[grid[0].n][1];
    double *recvbuf = &M[0][1];
    MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][1], 99, recvbuf, 1, edge[0], cart_nbr[0][0], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the -y direction
  {
    double *sendbuf = &M[1][1];
    double *recvbuf = &M[1][grid[1].n+1];
    MPI_Sendrecv(sendbuf, 1, edge[1], cart_nbr[1][0], 99, recvbuf, 1, edge[1], cart_nbr[1][1], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the +y direction
  {
    double *sendbuf = &M[1][grid[1].n];
    double *recvbuf = &M[1][0];
    MPI_Sendrecv(sendbuf, 1, edge[1], cart_nbr[1][1], 99, recvbuf, 1, edge[1], cart_nbr[1][0], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }

  // ==============================================================================================
  // Print subdomain after neighbor exchange
  // ==============================================================================================

  for (int k = 0; k < cart_size; k++) {
    if (cart_rank == k) {
      if (k == 0) {
        printf("\n========================\n");
        printf("After neighbor exchange");
        printf("\n========================\n\n");
        fflush(stdout);
      }
      printf("Cart rank is %d\n", cart_rank);
      printf("Cart coords are [%d, %d]\n", cart_coord[0], cart_coord[1]);
      printf("Cart nbrs are [[%d, %d], [%d, %d]\n", cart_nbr[0][0], cart_nbr[0][1], cart_nbr[1][0], cart_nbr[1][1]);
      for (int i = 0; i <= grid[0].n+1; i++) {
        for (int j = 0; j <= grid[1].n+1; j++) {
          printf("%2.0f ", M[i][j]);
        }
        printf("\n");
      }
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ==============================================================================================
  // Output everyone's subdomains to file
  // ==============================================================================================

  MPI_File fh;
  MPI_Datatype filetype, memtype;
  int global_array_sizes[ndim];
  int memarray_sizes[ndim];

  int subarray_sizes[ndim];
  int subarray_global_starts[ndim];
  int subarray_mem_starts[ndim];

  for (int i = 0; i < ndim; i++) {
    global_array_sizes[i] = global_grid[i].n;
    memarray_sizes[i] = grid[i].n + 2;

    subarray_sizes[i] = grid[i].n;
    subarray_global_starts[i] = cart_coord[i] * grid[i].n;
    subarray_mem_starts[i] = 1;
  }

  MPI_Type_create_subarray(ndim, global_array_sizes, subarray_sizes, subarray_global_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);

  MPI_Type_create_subarray(ndim, memarray_sizes, subarray_sizes, subarray_mem_starts, MPI_ORDER_C, MPI_DOUBLE, &memtype);
  MPI_Type_commit(&memtype);

  MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, &M[0][0], 1, memtype, MPI_STATUS_IGNORE);


  MPI_Finalize();
}
