// ************************************************************************************************
// Demonstrates common tasks with an MPI cartesian communicator, including:
//   * Using MPI_Cart_create to create the communicator
//   * Setting up MPI_Datatypes for ghost cells
//   * Using MPI_Sendrecv for ghost-cell exchanges
//   * Usin MPI parallel I/O to output results
//
// R. Rahaman, 2017
// ************************************************************************************************

#include <stdio.h>
#include "mpi.h"
#include "matrix.h"

int main (int argc, char *argv[]) {
  const int ndim = 2;            // Number of spatial dimensions
  const double bound_val = 0.0;  // Boundary values

  const int ngrid = 12;          // Number of gridpoints in each dimension of global domain
  int global_ngrid[ndim];        // Number of gridpoints in each dimension of global domain
  int local_ngrid[ndim];         // Number of gridpoints in each dimension of this proc's subdomain

  int world_rank, world_size;    // The rank, size in MPI_COMM_WORLD

  MPI_Comm cart_comm;            // The cartesian communicator
  int cart_rank, cart_size;      // Size, rank in the cartesian communicator
  int cart_dim[ndim];            // The dimensions of the Cartesian communicator
  int cart_coord[ndim];          // This proc's coordinates in the Cartesian communicator
  int cart_nbr[ndim][2];         // This proc's neighbors along each dimension

  MPI_Datatype edge[ndim];       // Datatypes for memory along the edges of the subdomain

  double ** M;                   // The subdomain itself

  // ==============================================================================================
  // Setup the communicators (including Cartesian communicator)
  // ==============================================================================================

  MPI_Init(&argc, &argv);

  // Get info about MPI_COMM_WORLD
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Let MPI figure out the best dimensions for the Cartesian comm
  for (int i = 0; i < ndim; i++)
    cart_dim[i] = 0;
  MPI_Dims_create(world_size, ndim, cart_dim);

  // Creates the Cartesian communicator
  {
    int reorder = 1;          // Means MPI is allowed to reorder ranks in Cart comm
    int cart_period[ndim];    // Means we aren't using periodic boundary conditions in Cartesian communicator
    for (int i = 0; i < ndim; i++)
      cart_period[i] = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, cart_dim, cart_period, reorder, &cart_comm);
  }

  // Get info about the new Cart comm
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Comm_size(cart_comm, &cart_size);
  MPI_Cart_coords(cart_comm, cart_rank, ndim, cart_coord);
  for (int i = 0; i < ndim; i++)
    MPI_Cart_shift(cart_comm, i, 1, &cart_nbr[i][0], &cart_nbr[i][1]);

  // Ensure that the number of gridpoints are evenly-divisible by the Cart comm
  if (world_rank == 0) {
    printf("The Cartesian communicator is %d x %d procs\n", cart_dim[0], cart_dim[1]);
    printf("The spatial domain is %d x %d gripoints\n", ngrid, ngrid);
    if (ngrid % cart_dim[0] != 0 || ngrid % cart_dim[1] != 0) {
      fprintf(stderr, "ERROR: The number of gridpoints are not evenly divisible by the number of processes\n");
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    }
  }

  // ==============================================================================================
  // Setup the spatial grids
  // ==============================================================================================

  // Info about the global grid
  for (int i = 0; i < ndim; i++)
    global_ngrid[i] = ngrid;

  // Info about the local grid
  for (int i = 0; i < ndim; i++)
    local_ngrid[i] = global_ngrid[i] / cart_dim[i];

  // Malloc the local grid
  M = matrix_2d_alloc(local_ngrid[0] + 2, local_ngrid[1] + 2);

  // ==============================================================================================
  // Initialize datatypes
  // ==============================================================================================

  MPI_Type_contiguous(local_ngrid[1], MPI_DOUBLE, &edge[0]);
  MPI_Type_commit(&edge[0]);

  MPI_Type_vector(local_ngrid[0], 1, local_ngrid[1] + 2, MPI_DOUBLE, &edge[1]);
  MPI_Type_commit(&edge[1]);

  // ==============================================================================================
  // Set initial conditions
  // ==============================================================================================

  for (int i = 0; i <= local_ngrid[0]+1; i++)
    for (int j = 0; j <= local_ngrid[1]+1; j++)
      M[i][j] = -1;
  for (int i = 1; i <= local_ngrid[0]; i++)
    for (int j = 1; j <= local_ngrid[1]; j++)
      M[i][j] = cart_rank+1;

  // ==============================================================================================
  // Set boundary conditions
  // ==============================================================================================

  // ... on the -x edge
  if (cart_nbr[0][0] == MPI_PROC_NULL) {
    int i = 0;
    for (int j = 0; j <= local_ngrid[1]+1; j++)
      M[i][j] = bound_val;
  }
  // ... on the +x edge
  if (cart_nbr[0][1] == MPI_PROC_NULL) {
    int i = local_ngrid[0] + 1;
    for (int j = 0; j <= local_ngrid[1]+1; j++)
      M[i][j] = bound_val;
  }
  // ... on the -y edge
  if (cart_nbr[1][0] == MPI_PROC_NULL) {
    int j = 0;
    for (int i = 0; i <= local_ngrid[0]+1; i++)
      M[i][j] = bound_val;
  }
  // ... on the +y edge
  if (cart_nbr[1][1] == MPI_PROC_NULL) {
    int j = local_ngrid[1] + 1;
    for (int i = 0; i <= local_ngrid[0]+1; i++)
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
      for (int i = 0; i <= local_ngrid[0]+1; i++) {
        for (int j = 0; j <= local_ngrid[1]+1; j++) {
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
    double *recvbuf = &M[local_ngrid[0] + 1][1];
    MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][0], 99, recvbuf, 1, edge[0], cart_nbr[0][1], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the -x direction
  {
    double *sendbuf = &M[local_ngrid[0]][1];
    double *recvbuf = &M[0][1];
    MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][1], 99, recvbuf, 1, edge[0], cart_nbr[0][0], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the -y direction
  {
    double *sendbuf = &M[1][1];
    double *recvbuf = &M[1][local_ngrid[1]+1];
    MPI_Sendrecv(sendbuf, 1, edge[1], cart_nbr[1][0], 99, recvbuf, 1, edge[1], cart_nbr[1][1], 99,
                 cart_comm, MPI_STATUS_IGNORE);
  }
  // ... in the +y direction
  {
    double *sendbuf = &M[1][local_ngrid[1]];
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
      for (int i = 0; i <= local_ngrid[0]+1; i++) {
        for (int j = 0; j <= local_ngrid[1]+1; j++) {
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
  // This is based on examples from "Using Advanced MPI", Chapter 7.

  const char outfile[] = "cart_demo.out";     // Filename
  MPI_File fh;                                // Filehandle
  MPI_Datatype io_filemap, io_memmap;         // Datatypes for mapping subdomain to file, memory layout
  int mem_ngrid[ndim];                        // Number of gridpoints in memory layout (incl. ghost cells)
  int local_filestarts[ndim];                 // Starting coordinates of subdomain in file
  int local_memstarts[ndim];                  // Starting coordinates of subdomain in memory layout

  for (int i = 0; i < ndim; i++) {
    mem_ngrid[i] = local_ngrid[i] + 2;                     // We have one ghost cell on each side
    local_filestarts[i] = cart_coord[i] * local_ngrid[i];  // Subdomain begins at this point in the file
    local_memstarts[i] = 1;                                // Subdomain begins at this point in memory
  }

  // Maps subdomain to file view
  MPI_Type_create_subarray(ndim, global_ngrid, local_ngrid, local_filestarts, MPI_ORDER_C, MPI_DOUBLE, &io_filemap);
  MPI_Type_commit(&io_filemap);

  // Maps subdomain to the memory layout
  MPI_Type_create_subarray(ndim, mem_ngrid, local_ngrid, local_memstarts, MPI_ORDER_C, MPI_DOUBLE, &io_memmap);
  MPI_Type_commit(&io_memmap);

  MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh, 0, MPI_DOUBLE, io_filemap, "native", MPI_INFO_NULL);

  // After all the setup, collective I/O is done with a single function call.
  MPI_File_write_all(fh, &M[0][0], 1, io_memmap, MPI_STATUS_IGNORE);

  MPI_File_close(&fh);

  // ==============================================================================================
  // Cleanup
  // ==============================================================================================

  matrix_2d_free(M);
  MPI_Comm_free(&cart_comm);
  for (int i =0; i < ndim; i++)
    MPI_Type_free(&edge[i]);
  MPI_Type_free(&io_filemap);
  MPI_Type_free(&io_memmap);

  MPI_Finalize();
}
