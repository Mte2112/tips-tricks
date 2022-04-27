# Commonly used packages
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # for colorbar on contourf plots
from matplotlib.colors import Normalize   # used to create the nonlinear cmaps in precip indices
import cartopy as cart
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from numpy import meshgrid, deg2rad, gradient, cos
from xarray import DataArray
from datetime import datetime as dt
import matplotlib as mpl
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import datetime
import warnings
import os
import datetime