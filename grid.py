import numpy as np

GRID_TYPES = ["radial_grid", "cartesian_1d"]

class Radial_Grid:
    def __init__(self, r_gridsize , rmin ,rmax) -> None:
        self.grid_type = GRID_TYPES[0]
        self.rmin = rmin
        self.rmax = rmax
        self.r_gridsize = r_gridsize
        (self.gridvec, self.grid_dr) = np.linspace(self.rmax, self.rmin, self.r_gridsize \
                ,endpoint=False,retstep=True)
        self.gridvec = self.gridvec[::-1]
        
