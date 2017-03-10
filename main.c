//
// Created by Ron Rahaman on 2/20/17.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "grid.h"
#include "mpi.h"
#include "matrix.h"


int main(int argc, char *argv[]) {

  const int ndim = 2;             // This is a 2D problem

  const double alpha = 0.001;
  const double mean = 1.0;
  const double var = 0.2;

  const double global_grid_min = 0.0;
  const double global_grid_max = 2.0;
  int global_grid_n;
  double global_grid_del;

  const double time_del = 0.0005;
  int time_n;

  int world_rank, world_size;      // The rank, size in MPI_COMM_WORLD

  MPI_Comm cart_comm;              // The cartesian communicator
  int cart_rank, cart_size;        // Size, rank in the cartesian communicator
  int cart_period[ndim];   // Not using periodic boundary conditions in Cartesian communicator
  int cart_dim[ndim];       // The dimensions of the Cartesian communicator
  int cart_coord[ndim];               // This proc's coordinates in the Cartesian communicator
  int cart_nbr[ndim][2];

  MPI_Datatype edge[ndim];

  Grid_info grid[ndim];    // Grid info for this subdomian
  Grid_info global_grid[ndim];   // Grid info for this subdomain

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

  // Get the size of the problem, number of timesteps
  if (world_rank == 0) {

    global_grid_n = (argc > 0) ? atoi(argv[1]) : 10;
    global_grid_del = (global_grid_max - global_grid_min) / global_grid_n;

    // Check consistency w/ cart comm
    printf("The Cartesian communicator is %d x %d procs\n", cart_dim[0], cart_dim[1]);
    printf("The spatial domain is %d x %d gripoints\n", global_grid_n, global_grid_n);
    if (global_grid_n % cart_dim[0] != 0 || global_grid_n % cart_dim[1] != 0) {
      fprintf(stderr, "ERROR: The number of gridpoints are not evenly divisible by the number of processes\n");
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS);
    }

    // Check stability condition
    time_n = (argc > 1) ? atoi(argv[2]) : 100;
    printf("The number of timesteps is %d", time_n);
    if (alpha * time_del / pow(global_grid_del, ndim) >= 1 / pow(2, ndim)) {
      fprintf(stderr, "ERROR: The Courant stability condition was not met. ");
      fprintf(stderr, "(User provided ndim=%d, alpha=%g, del_t=%g, del_x=%g)\n", ndim, alpha, time_del,
              global_grid_del);
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_BASE);
    }

  }

  // If everything's okay, broadcast the global domain size and set info about my subdomain
  MPI_Bcast(&global_grid_n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_grid_del, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&time_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Set info about the global grid
  for (int i = 0; i < ndim; i++) {
    global_grid[i].n = global_grid_n;
    global_grid[i].min = global_grid_min;
    global_grid[i].max = global_grid_max;
    global_grid[i].del = global_grid_del;
  }

  // Set info about the local (this proc's) grid
  for (int i = 0; i < ndim; i++) {
    grid[i].del = global_grid[i].del;
    grid[i].n = global_grid[i].n / cart_dim[i];
    grid[i].min = cart_coord[i] * grid[i].n * grid[i].del;
    grid[i].max = grid[i].min + grid[i].n * grid[i].del;
  }

  // Set up datatypes for rows and columns.
  MPI_Type_contiguous(grid[1].n, MPI_DOUBLE, &edge[0]);
  MPI_Type_commit(&edge[0]);

  MPI_Type_vector(grid[0].n, 1, grid[1].n + 2, MPI_DOUBLE, &edge[1]);
  MPI_Type_commit(&edge[1]);

  // Malloc the spatial domains.  NOTE THAT WE NEED TO INCLUDE GHOST CELLS
  Told = matrix_2d_alloc(grid[0].n + 2, grid[1].n + 2);
  Tnew = matrix_2d_alloc(grid[0].n + 2, grid[1].n + 2);

  // Initial distribution
  for (int i = 1; i <= grid[0].n; i++) {
    for (int j = 1; j <= grid[1].n; j++) {
      double x = i * grid[0].del;
      double y = j * grid[1].del;
      Told[i][j] = 1. / (2 * M_PI * var) *
                   exp(-1. * (pow(x - mean, 2) + pow(y - mean, 2)) / (2. * var));
    }
  }

  for (int time_step = 0; time_step < time_n; time_step++) {

    double t = time_step * time_del;

    // Set boundary conditions...
    // ..if there is no neighbor in the -x direction
    if (cart_nbr[0][0] == MPI_PROC_NULL) {
      int i = 0;
      for (int j = 0; j <= grid[1].n + 1; j++)
        Told[i][j] = 0.0;
    }
    // ...if there is no neighbor in the +x direction
    if (cart_nbr[0][1] == MPI_PROC_NULL) {
      int i = grid[0].n + 1;
      for (int j = 0; j <= grid[1].n + 1; j++)
        Told[i][j] = 0.0;
    }
    // ...if there is no neighbor in the -y direction
    if (cart_nbr[1][0] == MPI_PROC_NULL) {
      int j = 0;
      for (int i = 0; i <= grid[0].n + 1; i++)
        Told[i][j] = 0.0;
    }
    // ...if there is no neighbor in the +y direction
    if (cart_nbr[1][1] == MPI_PROC_NULL) {
      int j = grid[1].n + 1;
      for (int i = 0; i <= grid[0].n + 1; i++)
        Told[i][j] = 0.0;
    }

    // Exchange ghost cells

    { // Send from this proc in -x direction
      double *sendbuf = &Told[1][1];
      double *recvbuf = &Told[grid[0].n + 1][1];
      MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][0], MPI_ANY_TAG, recvbuf, 1, edge[0], cart_rank, MPI_ANY_TAG,
                   cart_comm, MPI_STATUS_IGNORE);
    }
    { // Exchange in +x direction
      double *sendbuf = &Told[grid[0].n][1];
      double *recvbuf = &Told[0][1];
      MPI_Sendrecv(sendbuf, 1, edge[0], cart_nbr[0][1], MPI_ANY_TAG, recvbuf, 1, edge[0], cart_rank, MPI_ANY_TAG,
                   cart_comm, MPI_STATUS_IGNORE);
    }


    {}
    // Exchange in the +x direction
    //if (cart_nbr[0][1] != MPI_PROC_NULL] && cart_nbgr)

    // Do the timestepping
    for (int i = 1; i <= grid[0].n; i++) {
      for (int j = 1; j <= grid[1].n; j++) {


      }
    }


  }


}

