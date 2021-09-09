import numpy as np

GRID_TYPES = ["radial_grid", "cartesian_1d"]

class Radial_Grid:
    def __init__(self, r_gridsize , rmin ,rmax) -> None:
        self.grid_type = GRID_TYPES[0]
        self.rmin = rmin
        self.rmax = rmax
        self.r_gridsize = r_gridsize
        (self.gridvec, self.grid_dr) = np.linspace(self.rmin, self.rmax, self.r_gridsize \
                ,endpoint=False,retstep=True)
        

class  Cartesian_1d:
    def __init__(self, x_gridsize , xmin ,xmax) -> None:
        self.grid_type = GRID_TYPES[1]
        self.xmin = xmin
        self.xmax = xmax
        self.x_gridsize = x_gridsize
        (self.gridvec, self.grid_dr) = np.linspace(self.xmin, self.xmax, self.x_gridsize \
                ,endpoint=False,retstep=True)

       