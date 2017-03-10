//
// Created by Ron Rahaman on 2/20/17.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "matrix.h"


int main(int argc, char *argv[]) {

  const double time_delta = 5e-6; // Interval between timesteps
  const double alpha = 0.001;     // The diffusivity constant
  const double bound_val = 0.0;   // Boundary values
  const double mean = 0.5;        // Parameters for the Gaussian initial condition
  const double var = 0.1;

  // Here, we're assuming symmetric spatial grid, but this can be easily generalized
  const int ngrid = 1000;        // Number of gridpoints in each dimension of global domain
  const double grid_min = 0.0;    // Value of minimum gridpoint in each dimension
  const double grid_max = 1.0;    // Value of maximum gridpoint in each dimension
  const double grid_delta =       // Spacing of gridpoints in each dimension
      (grid_max - grid_min) / ngrid;

  const int ndim = 2;             // Number of spatial dimensions

  int n_timesteps;                // The number of timesteps
  int n_io_steps;                 // After this many steps, output to file

  // Info about the global grid in each dimension.
  int global_ngrid[ndim];         // Number of gridpoints in each dimension of global domain
  double global_grid_min[ndim];   // Minimum gridpoint in each dimension of global domain
  double global_grid_max[ndim];   // Maximum gridpoint in each dimesnsion of global domain
  double global_grid_delta[ndim]; // Grid spacing in each dimension

  // Info about this proc's local grid in each dimension
  int local_ngrid[ndim];
  double local_grid_min[ndim];
  double local_grid_max[ndim];
  double local_grid_delta[ndim];

  int world_rank, world_size;     // The rank, size in MPI_COMM_WORLD

  MPI_Comm cart_comm;             // The cartesian communicator
  int cart_rank, cart_size;       // Size, rank in the cartesian communicator
  int cart_dim[ndim];             // The dimensions of the Cartesian communicator
  int cart_coord[ndim];           // This proc's coordinates in the Cartesian communicator
  int cart_nbr[ndim][2];          // This proc's neighbors along each dimension

  MPI_Datatype edge[ndim];        // Datatypes for memory along the edges of the subdomain

  double **Tnew, **Told;  // The spatial domains for the current/ previous timesteps

  /*
  MPI_Datatype stride;
  int nbrleft, nbrright, nbrtop, nbrbottom;
  int sx, ex, sy, ey;
  int dims[2];
  int cart_period[2];
  double diff2d, diffnorm, dwork;
  double t1, t2;
  */

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
  // Timestepping info
  // ==============================================================================================

  // Double-check stability condition
  if (world_rank == 0) {
    if (alpha * time_delta / pow(grid_delta, ndim) >= 1 / pow(2, ndim)) {
      fprintf(stderr, "ERROR: The Courant stability condition was not met. ");
      fprintf(stderr, "(User provided ndim=%d, alpha=%g, del_t=%g, del_x=%g)\n", ndim, alpha, time_delta, grid_delta);
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_BASE);
    }
  }

  //
  if (world_rank == 0) {
    // n_timesteps = (argc > 1) ? atoi(argv[2]) : 100;
    // n_io_steps  = (argc > 2) ? atoi(argv[3]) : 10;
    n_timesteps = (argc > 1) ? atoi(argv[2]) : 1000;
    n_io_steps  = (argc > 2) ? atoi(argv[3]) : 100;
    printf("The number of timesteps is %d\n", n_timesteps);
    printf("Output will be written every %d\n", n_io_steps);

  }
  MPI_Bcast(&n_timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_io_steps,  1, MPI_INT, 0, MPI_COMM_WORLD);

  // ==============================================================================================
  // Setup spatial grids
  // ==============================================================================================

  // Set info about the global local_grid.  Somewhat redundant, since we're assuming a symmetric grid, but this
  // setup will make it very easy to generalize to grid that's not symmetric.
  for (int i = 0; i < ndim; i++) {
    global_ngrid[i]      = ngrid;
    global_grid_delta[i] = grid_delta;
    global_grid_min[i]   = grid_min;
    global_grid_max[i]   = grid_max;
   }

  for (int i = 0; i < ndim; i++) {
    local_ngrid[i]      = global_ngrid[i] / cart_dim[i];
    local_grid_delta[i] = global_grid_delta[i];
    local_grid_min[i]   = cart_coord[i] * local_ngrid[i] * local_grid_delta[i];
    local_grid_max[i]   = local_grid_min[i] + local_ngrid[i] * local_grid_delta[i];
  }

  // Malloc the local grids
  Told = matrix_2d_alloc(local_ngrid[0] + 2, local_ngrid[1] + 2);
  Tnew = matrix_2d_alloc(local_ngrid[0] + 2, local_ngrid[1] + 2);

  // ==============================================================================================
  // Initialize datatypes for ghost cell exchange
  // ==============================================================================================

  MPI_Type_contiguous(local_ngrid[1], MPI_DOUBLE, &edge[0]);
  MPI_Type_commit(&edge[0]);

  MPI_Type_vector(local_ngrid[0], 1, local_ngrid[1] + 2, MPI_DOUBLE, &edge[1]);
  MPI_Type_commit(&edge[1]);

  // ==============================================================================================
  // Set initial conditions
  // ==============================================================================================

  for (int i = 1; i <= local_ngrid[0]; i++) {
    for (int j = 1; j <= local_ngrid[1]; j++) {
      double x = i * local_grid_delta[0] + local_grid_min[0];
      double y = j * local_grid_delta[1] + local_grid_min[1];
      Told[i][j] = 1. / (2 * M_PI * var) *
                   exp(-1. * (pow(x - mean, 2) + pow(y - mean, 2)) / (2. * var));
    }
  }

  // ==============================================================================================
  // Set boundary conditions
  // ==============================================================================================

  // ... on the -x edge
  if (cart_nbr[0][0] == MPI_PROC_NULL) {
    int i = 0;
    for (int j = 0; j <= local_ngrid[1]+1; j++) {
      Told[i][j] = bound_val;
      Tnew[i][j] = bound_val;
    }
  }
  // ... on the +x edge
  if (cart_nbr[0][1] == MPI_PROC_NULL) {
    int i = local_ngrid[0] + 1;
    for (int j = 0; j <= local_ngrid[1]+1; j++) {
      Told[i][j] = bound_val;
      Tnew[i][j] = bound_val;
    }
  }
  // ... on the -y edge
  if (cart_nbr[1][0] == MPI_PROC_NULL) {
    int j = 0;
    for (int i = 0; i <= local_ngrid[0]+1; i++) {
      Told[i][j] = bound_val;
      Tnew[i][j] = bound_val;
    }
  }
  // ... on the +y edge
  if (cart_nbr[1][1] == MPI_PROC_NULL) {
    int j = local_ngrid[1] + 1;
    for (int i = 0; i <= local_ngrid[0]+1; i++) {
      Told[i][j] = bound_val;
      Tnew[i][j] = bound_val;
    }
  }

  // ==============================================================================================
  // Setup for file I/O
  // ==============================================================================================
  // This is based on examples from "Using Advanced MPI", Chapter 7.

  char outfile[256];                          // Name of the output files.
  MPI_File fh;                                // Filehandle to output file
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


  // ==============================================================================================
  // The timestep loop
  // ==============================================================================================

  for (int k = 0; k < n_timesteps; k++) {

    // --------------------------------------------------------------------------------------------
    // Write to file
    // --------------------------------------------------------------------------------------------

    if (n_io_steps > 0 && k % n_io_steps == 0) {
      sprintf(outfile, "jacobi_%d.out", k);
      MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
      MPI_File_set_view(fh, 0, MPI_DOUBLE, io_filemap, "native", MPI_INFO_NULL);
      MPI_File_write_all(fh, &Told[0][0], 1, io_memmap, MPI_STATUS_IGNORE);
      MPI_File_close(&fh);
    }

    // --------------------------------------------------------------------------------------------
    // Neighbor Exchanges
    // --------------------------------------------------------------------------------------------

    // ... in the +x direction
    {
      double *sendbuf = &Told[1][1];
      double *recvbuf = &Told[local_ngrid[0] + 1][1];
      MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][0], 99, recvbuf, 1, edge[0], cart_nbr[0][1], 99,
                   cart_comm, MPI_STATUS_IGNORE);
    }
    // ... in the -x direction
    {
      double *sendbuf = &Told[local_ngrid[0]][1];
      double *recvbuf = &Told[0][1];
      MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][1], 99, recvbuf, 1, edge[0], cart_nbr[0][0], 99,
                   cart_comm, MPI_STATUS_IGNORE);
    }
    // ... in the -y direction
    {
      double *sendbuf = &Told[1][1];
      double *recvbuf = &Told[1][local_ngrid[1]+1];
      MPI_Sendrecv(sendbuf, 1, edge[1], cart_nbr[1][0], 99, recvbuf, 1, edge[1], cart_nbr[1][1], 99,
                   cart_comm, MPI_STATUS_IGNORE);
    }
    // ... in the +y direction
    {
      double *sendbuf = &Told[1][local_ngrid[1]];
      double *recvbuf = &Told[1][0];
      MPI_Sendrecv(sendbuf, 1, edge[1], cart_nbr[1][1], 99, recvbuf, 1, edge[1], cart_nbr[1][0], 99,
                   cart_comm, MPI_STATUS_IGNORE);
    }

    // --------------------------------------------------------------------------------------------
    // Solve for the next timestep
    // --------------------------------------------------------------------------------------------

    double C = alpha * time_delta / pow(grid_delta, 2);
    for (int i = 1; i <= local_ngrid[0]; i++)
      for (int j = 1; j <= local_ngrid[1]; j++)
        Tnew[i][j] = Told[i][j] +  C * (Told[i-1][j] + Told[i+1][j] + Told[i][j-1] + Told[i][j+1] - 4 * Told[i][j]);

    // --------------------------------------------------------------------------------------------
    // Pointer swap
    // --------------------------------------------------------------------------------------------

    double ** temp = Tnew;
    Tnew = Told;
    Told = temp;

  }

  // ==============================================================================================
  // Cleanup
  // ==============================================================================================

  matrix_2d_free(Told);
  matrix_2d_free(Tnew);
  MPI_Comm_free(&cart_comm);
  for (int i =0; i < ndim; i++)
    MPI_Type_free(&edge[i]);
  MPI_Type_free(&io_filemap);
  MPI_Type_free(&io_memmap);

  MPI_Finalize();
}

