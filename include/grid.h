//
// Created by Ron Rahaman on 3/6/17.
//

#ifndef FTCS_3D_GRID_H
#define FTCS_3D_GRID_H

typedef struct _grid_info{
  int n;
  double min;
  double max;
  double del;
} Grid_info;

void  set_2d_cart_bounds(double ** domain, Grid_info grid[], int neighbors[][2], double bound_val);

#endif //FTCS_3D_GRID_H
