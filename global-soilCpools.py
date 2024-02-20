
# #########################################################################################################
# 
# ## Python script for global model comparison of soil carbon pools
# 
# ### Emergent temperature sensitivity of soil organic carbon controlled by mineral associations
# ### Published in Nature Geoscience, 2024
# 
# ### Data sources: 
# - CMIP6 ESMs: https://esgf-node.llnl.gov/search/cmip6/
# - Biogeochemical Testbed: https://doi.org/10.5065/d6nc600w
# - Data Product: https://doi.org/10.5281/zenodo.6539765
# - All covariates are freely available in the references detailed in the manuscript, and are also available from the corresponding author upon request.
# 
# ### Contact: Katerina Georgiou (georgiou1@llnl.gov)
# 
# #########################################################################################################

# ## Set-up packages and functions

# importing packages
#
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
%matplotlib inline
import csv
import numpy.ma as ma
#
from pylab import *
from math import sqrt
import random
#
import warnings
warnings.filterwarnings('ignore')
#
from netCDF4 import Dataset
#
from mpl_toolkits import basemap
import matplotlib as ml
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import ImageGrid
#
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
#
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
#
import pyproj   
from pyproj import Proj
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from shapely.geometry import Polygon
from functools import partial
from pyproj import Geod


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    cmapIn = plt.get_cmap(cmapIn)
    #
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))
    #
    return new_cmap


def truncate_bicolormap(cmapIn='PuOr', minval=0.0, mid1=0.47, mid2=0.53, maxval=1.0, n=128):
    top = plt.get_cmap(cmapIn, n)
    bottom = plt.get_cmap(cmapIn, 128)
    #
    newcolors = np.vstack((top(np.linspace(minval, mid1, n)),
                           bottom(np.linspace(mid2, maxval, n))))
    #
    new_cmap = ListedColormap(newcolors)
    return new_cmap


def global_area(lats_cmip6, lons_cmip6):
    dlatout = 180/float(lats_cmip6.shape[0]) # size of lat grid
    dlonout = 360/float(lons_cmip6.shape[0]) # size of lon grid

    latsize = int(180/dlatout) # as integer
    lonsize = int(360/dlonout) # as integer

    area = np.zeros((latsize,lonsize,))
    outlats = np.arange(90-dlatout/2, -90, -dlatout)
    outlons = np.arange(-180+dlonout/2, 180, dlonout)

    for lato in np.arange(0,(latsize-1),1):
        ymax = outlats[lato] + dlatout/2
        ymin = outlats[lato] - dlatout/2
        xmax = outlons[0] + dlonout/2
        xmin = outlons[0] - dlonout/2

        geom = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])

        geod = Geod(ellps="WGS84")
        geom_area, geom_perimeter = geod.geometry_area_perimeter(geom)

        area[lato,:] = -1*geom_area # area in m^2 (area with counter-clockwise traversal as positive)

    area_ud = np.flipud(area) # in m2
    return area_ud


# ## Model output files

os.chdir("/")

# ### CMIP6 files

# files for historical and ssp585 output for all CMIP6 models (except E3SM)
#
cmip6_data_in = [
    ['ACCESS-ESM1-5',
     'cSoil_Emon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12, # 10yrs
     'cSoilSlow_Lmon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12, # 10yrs
     'npp_Lmon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12, # 10yrs
     'pr_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12, # 30yrs
     'tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12, # 30yrs
     './ssp585/cSoil_Emon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12, # 10yrs
     './ssp585/cSoilSlow_Lmon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12, # 10yrs
     './ssp585/npp_Lmon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12, # 10yrs
     './ssp585/pr_Amon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12, # 30yrs
     './ssp585/tas_Amon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12, # 30yrs
     1850, 1850
    ],
    ['BCC-CSM2-MR',
     'cSoil_Emon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/cSoilSlow_Lmon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/npp_Lmon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     1850, 2015
    ],
    ['CESM2',
     'cSoil_Emon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_CESM2_historical_r4i1p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc', (2090-2065)*12, (2101-2065)*12,
     './ssp585/cSoilSlow_Lmon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc', (2090-2065)*12, (2101-2065)*12,
     './ssp585/npp_Lmon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc', (2090-2065)*12, (2101-2065)*12,
     './ssp585/pr_Amon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc', (2070-2065)*12, (2101-2065)*12,
     './ssp585/tas_Amon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc', (2070-2065)*12, (2101-2065)*12,
     1, 1,
     './ssp585/cSoil_Emon_CESM2_ssp585_r4i1p1f1_gn_201501-206412.nc',
     './ssp585/cSoilSlow_Lmon_CESM2_ssp585_r4i1p1f1_gn_201501-206412.nc'
    ], 
    ['CNRM-ESM2-1',
     'cSoil_Emon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/cSoilSlow_Lmon_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/npp_Lmon_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_CNRM-ESM2-1_ssp585_r1i1p1f2_gr_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     1850, 1850
    ],
    ['IPSL-CM6A-LR',
     'cSoil_Emon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/cSoilSlow_Lmon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/npp_Lmon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     1850, 2015
    ],
    ['MIROC-ES2L',
     'cSoil_Emon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_MIROC-ES2L_ssp585_r1i1p1f2_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/cSoilSlow_Lmon_MIROC-ES2L_ssp585_r1i1p1f2_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/npp_Lmon_MIROC-ES2L_ssp585_r1i1p1f2_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_MIROC-ES2L_ssp585_r1i1p1f2_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_MIROC-ES2L_ssp585_r1i1p1f2_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     1850, 1850
    ],
    ['MRI-ESM2-0',
     'cSoil_Emon_MRI-ESM2-0_historical_r1i2p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'cSoilSlow_Lmon_MRI-ESM2-0_historical_r1i2p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'npp_Lmon_MRI-ESM2-0_historical_r1i2p1f1_gn_185001-201412.nc', (2005-1850)*12, (2015-1850)*12,
     'pr_Amon_MRI-ESM2-0_historical_r1i2p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     'tas_Amon_MRI-ESM2-0_historical_r1i2p1f1_gn_185001-201412.nc', (1985-1850)*12, (2015-1850)*12,
     './ssp585/cSoil_Emon_MRI-ESM2-0_ssp585_r1i2p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/cSoilSlow_Lmon_MRI-ESM2-0_ssp585_r1i2p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/npp_Lmon_MRI-ESM2-0_ssp585_r1i2p1f1_gn_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_MRI-ESM2-0_ssp585_r1i2p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_MRI-ESM2-0_ssp585_r1i2p1f1_gn_201501-210012.nc', (2070-2015)*12, (2101-2015)*12,
     1850, 1850
    ],
    ['NorESM2',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_201001-201412.nc', 0, (2015-2010)*12,
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_201001-201412.nc', 0, (2015-2010)*12,
     'npp_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_201001-201412.nc', 0, (2015-2010)*12, 
     'pr_Amon_NorESM2-MM_historical_r1i1p1f1_gn_201001-201412.nc', 0, (2015-2010)*12, 
     'tas_Amon_NorESM2-MM_historical_r1i1p1f1_gn_201001-201412.nc', 0, (2015-2010)*12, 
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_209101-210012.nc', (2095-2091)*12, (2101-2091)*12,
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_209101-210012.nc', (2095-2091)*12, (2101-2091)*12,
     './ssp585/npp_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_209101-210012.nc', 0, (2101-2091)*12,
     './ssp585/pr_Amon_NorESM2-MM_ssp585_r1i1p1f1_gn_209101-210012.nc', 0, (2101-2091)*12,
     './ssp585/tas_Amon_NorESM2-MM_ssp585_r1i1p1f1_gn_209101-210012.nc', 0, (2101-2091)*12,
     1, 1,
     #
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_200001-200912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_199001-199912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_198001-198912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_197001-197912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_196001-196912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_195001-195912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_194001-194912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_193001-193912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_192001-192912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_191001-191912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_190001-190912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_189001-189912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_188001-188912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_187001-187912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_186001-186912.nc',
     'cSoil_Emon_NorESM2-MM_historical_r1i1p1f1_gn_185001-185912.nc',
     #
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_200001-200912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_199001-199912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_198001-198912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_197001-197912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_196001-196912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_195001-195912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_194001-194912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_193001-193912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_192001-192912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_191001-191912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_190001-190912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_189001-189912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_188001-188912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_187001-187912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_186001-186912.nc',
     'cSoilSlow_Lmon_NorESM2-MM_historical_r1i1p1f1_gn_185001-185912.nc',
     #
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_208101-209012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_207101-208012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_206101-207012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_205101-206012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_204101-205012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_203101-204012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_202101-203012.nc',
     './ssp585/cSoil_Emon_NorESM2-MM_ssp585_r1i1p1f1_gn_201502-202012.nc',
     #
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_208101-209012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_207101-208012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_206101-207012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_205101-206012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_204101-205012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_203101-204012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_202101-203012.nc',
     './ssp585/cSoilSlow_Lmon_NorESM2-MM_ssp585_r1i1p1f1_gn_201502-202012.nc'
    ]
]


# ### E3SM files

# files for historical and ssp585 output for CMIP6 E3SM
#
cmip6_e3sm_in = [
    ['E3SM-1-1-ECA', 
     'E3SMv1-ECA.historical.cSoil.185001_201412.nc', (2010-1850)*12, (2015-1850)*12,
     'gpp_Lmon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_201001-201412.nc', 0, (2015-2010)*12, 
     'ra_Lmon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_201001-201412.nc', 0, (2015-2010)*12,
     'pr_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_201001-201412.nc', 0, (2015-2010)*12,
     'tas_Amon_E3SM-1-1-ECA_historical_r1i1p1f1_gr_201001-201412.nc', 0, (2015-2010)*12,
     './ssp585/E3SMv1-ECA.historical.cSoil.201501_210012.nc', (2095-2015)*12, (2101-2015)*12,
     './ssp585/gpp_Lmon_E3SM-1-1-ECA_ssp585-bgc_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/ra_Lmon_E3SM-1-1-ECA_ssp585-bgc_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/pr_Amon_E3SM-1-1-ECA_ssp585-bgc_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     './ssp585/tas_Amon_E3SM-1-1-ECA_ssp585-bgc_r1i1p1f1_gr_201501-210012.nc', (2090-2015)*12, (2101-2015)*12,
     1850, 1850
    ]
]


# ### Testbed files

# files for historical and ssp585 output for testbed
#
data_in_testbed = [
    ['CASA-CNP',
     './testbed/ann_casaclm_rcp85.nc', 
     (2005-1900), (2015-1900), # annual averages
     (2090-1900), (2100-1900),
     './testbed/ann_casaclm_rcp85.nc',
     (1985-1900), (2015-1900), 
     (2070-1900), (2100-1900),
     './testbed/RAIN.nc', 
     (1985-1850)*12, (2015-1850)*12 # monthly
    ],
    ['MIMICS',
     './testbed/ann_mimics_rcp85.nc', 
     (2005-1900), (2015-1900), # annual averages
     (2090-1900), (2100-1900),
     './testbed/ann_casaclm_rcp85.nc', 
     (1985-1900), (2015-1900), 
     (2070-1900), (2100-1900),
     './testbed/RAIN.nc', 
     (1985-1850)*12, (2015-1850)*12 # monthly
    ],
    ['CORPSE',
     './testbed/ann_corpse_rcp85.nc', 
     (2005-1900), (2015-1900), # annual averages
     (2090-1900), (2100-1900),
     './testbed/ann_casaclm_rcp85.nc', 
     (1985-1900), (2015-1900), 
     (2070-1900), (2100-1900),
     './testbed/RAIN.nc', 
     (1985-1850)*12, (2015-1850)*12 # monthly
    ]
]


# ## Loading model output

###################### initializing aggregated model output storage
#
name_cmip6_allmodels = []
#
lats_cmip6_allmodels = []
lons_cmip6_allmodels = []
#
cSoil_cmip6_allmodels = []
cSoilSlow_cmip6_allmodels = []
npp_cmip6_allmodels = []
pr_cmip6_allmodels = []
tas_cmip6_allmodels = []
#
cSoil_cmip6_rcp85_allmodels = []
cSoilSlow_cmip6_rcp85_allmodels = []
npp_cmip6_rcp85_allmodels = []
pr_cmip6_rcp85_allmodels = []
tas_cmip6_rcp85_allmodels = []
#
cSoil_cmip6_change_allmodels = []
cSoilSlow_cmip6_change_allmodels = []
ratio_cmip6_allmodels = []
ratio_cmip6_rcp85_allmodels = []
npp_cmip6_change_allmodels = []
pr_cmip6_change_allmodels = []
tas_cmip6_change_allmodels = []
#
cSoil_allmodels = []
cSoilSlow_allmodels = []
time_allmodels = []


# ### Loading CMIP6

# loading historical and ssp585 output for all CMIP6 models (except E3SM)
#
for i in range(0,8):
    #
    model_i=i
    #
    print( 'Model '+cmip6_data_in[model_i][0])
    name_cmip6 = cmip6_data_in[model_i][0]
    #
    ###################### historical output
    #
    print( ' opening file '+cmip6_data_in[model_i][1])
    cSoilfile_cmip6 = Dataset(cmip6_data_in[model_i][1])
    cSoil_cmip6 = cSoilfile_cmip6.variables['cSoil'][cmip6_data_in[model_i][2]:cmip6_data_in[model_i][3],:,:].mean(axis=0)
    cSoil_cmip6_all = cSoilfile_cmip6.variables['cSoil'][:]
    time_cmip6 = cSoilfile_cmip6.variables['time'][:]/365 + cmip6_data_in[model_i][31]
    lats_cmip6 = cSoilfile_cmip6.variables['lat'][:]
    lons_cmip6 = cSoilfile_cmip6.variables['lon'][:]
    cSoilfile_cmip6.close()
    #
    if name_cmip6 == 'NorESM2':
        jmax = 16
        for j in range(0,jmax):
            print( ' opening file '+cmip6_data_in[model_i][33+j])
            cSoilfile_cmip6 = Dataset(cmip6_data_in[model_i][33+j])
            cSoil_cmip6_temp = cSoilfile_cmip6.variables['cSoil'][:]
            time_cmip6_temp = cSoilfile_cmip6.variables['time'][:]/365 + cmip6_data_in[model_i][31]
            #
            cSoil_cmip6_all = ma.concatenate([cSoil_cmip6_temp, cSoil_cmip6_all])
            time_cmip6 = np.concatenate([time_cmip6_temp, time_cmip6])
            cSoilfile_cmip6.close()
            del time_cmip6_temp, cSoil_cmip6_temp
    #
    print( ' opening file '+cmip6_data_in[model_i][4])
    cSoilSlowfile_cmip6 = Dataset(cmip6_data_in[model_i][4])
    cSoilSlow_cmip6 = cSoilSlowfile_cmip6.variables['cSoilSlow'][cmip6_data_in[model_i][5]:cmip6_data_in[model_i][6],:,:].mean(axis=0)
    cSoilSlow_cmip6_all = cSoilSlowfile_cmip6.variables['cSoilSlow'][:]
    cSoilSlowfile_cmip6.close()
    #
    if name_cmip6 == 'NorESM2':
        kmax = 16
        for k in range(0,kmax):
            print( ' opening file '+cmip6_data_in[model_i][33+jmax+k])
            cSoilSlowfile_cmip6 = Dataset(cmip6_data_in[model_i][33+jmax+k])
            cSoilSlow_cmip6_temp = cSoilSlowfile_cmip6.variables['cSoilSlow'][:]
            cSoilSlow_cmip6_all = ma.concatenate([cSoilSlow_cmip6_temp, cSoilSlow_cmip6_all])
            cSoilSlowfile_cmip6.close()
            del cSoilSlow_cmip6_temp
    #
    print( ' opening file '+cmip6_data_in[model_i][7])
    nppfile_cmip6 = Dataset(cmip6_data_in[model_i][7])
    npp_cmip6 = nppfile_cmip6.variables['npp'][cmip6_data_in[model_i][8]:cmip6_data_in[model_i][9],:,:].mean(axis=0) * 60. * 60. * 24. * 365. * 1000 #s to yr and kg to g
    nppfile_cmip6.close()
    #
    print( ' opening file '+cmip6_data_in[model_i][10])
    prfile_cmip6 = Dataset(cmip6_data_in[model_i][10])
    pr_cmip6 = prfile_cmip6.variables['pr'][cmip6_data_in[model_i][11]:cmip6_data_in[model_i][12],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
    prfile_cmip6.close()
    #
    print( ' opening file '+cmip6_data_in[model_i][13])
    tasfile_cmip6 = Dataset(cmip6_data_in[model_i][13])
    tas_cmip6 = tasfile_cmip6.variables['tas'][cmip6_data_in[model_i][14]:cmip6_data_in[model_i][15],:,:].mean(axis=0) - 273.15 # K to celcius
    tasfile_cmip6.close()
    #
    ###################### ssp585 output
    #
    print( ' opening file '+cmip6_data_in[model_i][16])
    cSoilfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][16])
    cSoil_cmip6_rcp85 = cSoilfile_cmip6_rcp85.variables['cSoil'][cmip6_data_in[model_i][17]:cmip6_data_in[model_i][18],:,:].mean(axis=0)
    cSoil_cmip6_rcp85_all = cSoilfile_cmip6_rcp85.variables['cSoil'][:]
    time_cmip6_rcp85 = cSoilfile_cmip6_rcp85.variables['time'][:]/365 + cmip6_data_in[model_i][32]
    cSoilfile_cmip6_rcp85.close()
    #
    if name_cmip6 == 'CESM2':
        print( ' opening file '+cmip6_data_in[model_i][33])
        cSoilfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][33])
        cSoil_cmip6_rcp85_2015to2064 = cSoilfile_cmip6_rcp85.variables['cSoil'][:]
        time_cmip6_rcp85_2015to2064 = cSoilfile_cmip6_rcp85.variables['time'][:]/365 + cmip6_data_in[model_i][32]
        #
        cSoil_cmip6_rcp85_all = ma.concatenate([cSoil_cmip6_rcp85_2015to2064, cSoil_cmip6_rcp85_all])
        time_cmip6_rcp85 = np.concatenate([time_cmip6_rcp85_2015to2064, time_cmip6_rcp85])
        cSoilfile_cmip6_rcp85.close()
        del time_cmip6_rcp85_2015to2064, cSoil_cmip6_rcp85_2015to2064
    #
    if name_cmip6 == 'NorESM2':
        jmax_rcp = 8
        for j in range(0,jmax_rcp):
            print( ' opening file '+cmip6_data_in[model_i][33+jmax+kmax+j])
            cSoilfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][33+jmax+kmax+j])
            cSoil_cmip6_rcp85_temp = cSoilfile_cmip6_rcp85.variables['cSoil'][:]
            time_cmip6_rcp85_temp = cSoilfile_cmip6_rcp85.variables['time'][:]/365 + cmip6_data_in[model_i][32]

            #
            cSoil_cmip6_rcp85_all = ma.concatenate([cSoil_cmip6_rcp85_temp, cSoil_cmip6_rcp85_all])
            time_cmip6_rcp85 = np.concatenate([time_cmip6_rcp85_temp, time_cmip6_rcp85])
            cSoilfile_cmip6_rcp85.close()
            del time_cmip6_rcp85_temp, cSoil_cmip6_rcp85_temp
    #
    print( ' opening file '+cmip6_data_in[model_i][19])
    cSoilSlowfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][19])
    cSoilSlow_cmip6_rcp85 = cSoilSlowfile_cmip6_rcp85.variables['cSoilSlow'][cmip6_data_in[model_i][20]:cmip6_data_in[model_i][21],:,:].mean(axis=0)
    cSoilSlow_cmip6_rcp85_all = cSoilSlowfile_cmip6_rcp85.variables['cSoilSlow'][:]
    cSoilSlowfile_cmip6_rcp85.close()
    #
    if name_cmip6 == 'CESM2':
        print( ' opening file '+cmip6_data_in[model_i][34])
        cSoilSlowfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][34])
        cSoilSlow_cmip6_rcp85_2015to2064 = cSoilSlowfile_cmip6_rcp85.variables['cSoilSlow'][:]
        cSoilSlow_cmip6_rcp85_all = ma.concatenate([cSoilSlow_cmip6_rcp85_2015to2064, cSoilSlow_cmip6_rcp85_all])
        cSoilSlowfile_cmip6_rcp85.close()
        del cSoilSlow_cmip6_rcp85_2015to2064
    #
    if name_cmip6 == 'NorESM2':
        kmax_rcp = 8
        for k in range(0,kmax_rcp):
            print( ' opening file '+cmip6_data_in[model_i][33+jmax+kmax+jmax_rcp+k])
            cSoilSlowfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][33+jmax+kmax+jmax_rcp+k])
            cSoilSlow_cmip6_rcp85_temp = cSoilSlowfile_cmip6_rcp85.variables['cSoilSlow'][:]
            cSoilSlow_cmip6_rcp85_all = ma.concatenate([cSoilSlow_cmip6_rcp85_temp, cSoilSlow_cmip6_rcp85_all])
            cSoilSlowfile_cmip6_rcp85.close()
            del cSoilSlow_cmip6_rcp85_temp
    #
    print( ' opening file '+cmip6_data_in[model_i][22])
    nppfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][22])
    npp_cmip6_rcp85 = nppfile_cmip6_rcp85.variables['npp'][cmip6_data_in[model_i][23]:cmip6_data_in[model_i][24],:,:].mean(axis=0) * 60. * 60. * 24. * 365. * 1000 #s to yr and kg to g
    nppfile_cmip6_rcp85.close()
    #
    print( ' opening file '+cmip6_data_in[model_i][25])
    prfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][25])
    pr_cmip6_rcp85 = prfile_cmip6_rcp85.variables['pr'][cmip6_data_in[model_i][26]:cmip6_data_in[model_i][27],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
    prfile_cmip6_rcp85.close()
    #
    print( ' opening file '+cmip6_data_in[model_i][28])
    tasfile_cmip6_rcp85 = Dataset(cmip6_data_in[model_i][28])
    tas_cmip6_rcp85 = tasfile_cmip6_rcp85.variables['tas'][cmip6_data_in[model_i][29]:cmip6_data_in[model_i][30],:,:].mean(axis=0) - 273.15 # K to celcius
    tasfile_cmip6_rcp85.close()
    #
    ###################### masking
    #
    common_mask = np.logical_or(pr_cmip6 < 1e1,npp_cmip6 < 1e1)
    #
    cSoil_cmip6 = np.ma.masked_array(cSoil_cmip6, mask=common_mask)
    cSoilSlow_cmip6 = np.ma.masked_array(cSoilSlow_cmip6, mask=common_mask)
    npp_cmip6 = np.ma.masked_array(npp_cmip6, mask=common_mask)
    pr_cmip6 = np.ma.masked_array(pr_cmip6, mask=common_mask)
    tas_cmip6 = np.ma.masked_array(tas_cmip6, mask=common_mask)
    #
    cSoil_cmip6_rcp85 = np.ma.masked_array(cSoil_cmip6_rcp85, mask=common_mask)
    cSoilSlow_cmip6_rcp85 = np.ma.masked_array(cSoilSlow_cmip6_rcp85, mask=common_mask)
    npp_cmip6_rcp85 = np.ma.masked_array(npp_cmip6_rcp85, mask=common_mask)
    pr_cmip6_rcp85 = np.ma.masked_array(pr_cmip6_rcp85, mask=common_mask)
    tas_cmip6_rcp85 = np.ma.masked_array(tas_cmip6_rcp85, mask=common_mask)
    #
    cSoil_cmip6_change = cSoil_cmip6_rcp85 - cSoil_cmip6 
    cSoilSlow_cmip6_change = cSoilSlow_cmip6_rcp85 - cSoilSlow_cmip6
    npp_cmip6_change = npp_cmip6_rcp85 - npp_cmip6
    pr_cmip6_change = pr_cmip6_rcp85 - pr_cmip6
    tas_cmip6_change = tas_cmip6_rcp85 - tas_cmip6
    #
    ratio_cmip6 = cSoilSlow_cmip6/cSoil_cmip6 
    ratio_cmip6_rcp85 = cSoilSlow_cmip6_rcp85/cSoil_cmip6_rcp85
    #
    cSoilSlow_all = ma.concatenate([cSoilSlow_cmip6_all, cSoilSlow_cmip6_rcp85_all])
    cSoil_all = ma.concatenate([cSoil_cmip6_all, cSoil_cmip6_rcp85_all])
    time_all = np.concatenate([time_cmip6, time_cmip6_rcp85])
    #
    print(' resolution = ', round(180/cSoil_all.shape[1],2),'x', round(360/cSoil_all.shape[2],2), 'for years =', cSoil_all.shape[0]/12,'\n')
    #
    del time_cmip6, time_cmip6_rcp85, cSoil_cmip6_all, cSoil_cmip6_rcp85_all, cSoilSlow_cmip6_all, cSoilSlow_cmip6_rcp85_all
    #
    ###################### appending
    #
    name_cmip6_allmodels.append(name_cmip6)
    #
    lats_cmip6_allmodels.append(lats_cmip6)
    lons_cmip6_allmodels.append(lons_cmip6)
    #
    cSoil_cmip6_allmodels.append(cSoil_cmip6)
    cSoilSlow_cmip6_allmodels.append(cSoilSlow_cmip6)
    npp_cmip6_allmodels.append(npp_cmip6)
    pr_cmip6_allmodels.append(pr_cmip6)
    tas_cmip6_allmodels.append(tas_cmip6)
    #
    cSoil_cmip6_rcp85_allmodels.append(cSoil_cmip6_rcp85)
    cSoilSlow_cmip6_rcp85_allmodels.append(cSoilSlow_cmip6_rcp85)
    npp_cmip6_rcp85_allmodels.append(npp_cmip6_rcp85)
    pr_cmip6_rcp85_allmodels.append(pr_cmip6_rcp85)
    tas_cmip6_rcp85_allmodels.append(tas_cmip6_rcp85)
    #
    ratio_cmip6_allmodels.append(ratio_cmip6)
    ratio_cmip6_rcp85_allmodels.append(ratio_cmip6_rcp85)
    #
    cSoil_cmip6_change_allmodels.append(cSoil_cmip6_change)
    cSoilSlow_cmip6_change_allmodels.append(cSoilSlow_cmip6_change)
    npp_cmip6_change_allmodels.append(npp_cmip6_change)
    pr_cmip6_change_allmodels.append(pr_cmip6_change)
    tas_cmip6_change_allmodels.append(tas_cmip6_change)
    #
    cSoil_allmodels.append(cSoil_all)
    cSoilSlow_allmodels.append(cSoilSlow_all)
    time_allmodels.append(time_all)
    #
    del name_cmip6, lats_cmip6, lons_cmip6,cSoil_cmip6, cSoilSlow_cmip6, npp_cmip6, pr_cmip6, tas_cmip6
    del cSoil_cmip6_rcp85, cSoilSlow_cmip6_rcp85, npp_cmip6_rcp85, pr_cmip6_rcp85, tas_cmip6_rcp85
    del cSoil_cmip6_change, cSoilSlow_cmip6_change, npp_cmip6_change, pr_cmip6_change, tas_cmip6_change
    del cSoil_all, cSoilSlow_all, time_all, ratio_cmip6, ratio_cmip6_rcp85


# ### Loading E3SM

# loading historical and ssp585 output for CMIP6 E3SM
#
model_i = 0
#
print( 'Model '+cmip6_e3sm_in[model_i][0])
name_cmip6 = cmip6_e3sm_in[model_i][0]
#
###################### historical output
#
print( ' opening file '+cmip6_e3sm_in[model_i][1])
cSoilfile_cmip6 = Dataset(cmip6_e3sm_in[model_i][1])
cSoil_cmip6 = (cSoilfile_cmip6.variables['SOIL1C'][cmip6_e3sm_in[model_i][2]:cmip6_e3sm_in[model_i][3],:,:].mean(axis=0) +
               cSoilfile_cmip6.variables['SOIL2C'][cmip6_e3sm_in[model_i][2]:cmip6_e3sm_in[model_i][3],:,:].mean(axis=0) +
               cSoilfile_cmip6.variables['SOIL3C'][cmip6_e3sm_in[model_i][2]:cmip6_e3sm_in[model_i][3],:,:].mean(axis=0))/1000
cSoil_cmip6_all = (cSoilfile_cmip6.variables['SOIL1C'][:] + 
                   cSoilfile_cmip6.variables['SOIL2C'][:] + 
                   cSoilfile_cmip6.variables['SOIL3C'][:])/1000
time_cmip6 = cSoilfile_cmip6.variables['time'][:]/365 + 1850
lats_cmip6 = cSoilfile_cmip6.variables['lat'][:]
lons_cmip6 = cSoilfile_cmip6.variables['lon'][:]
#
cSoilSlow_cmip6 = cSoilfile_cmip6.variables['SOIL3C'][cmip6_e3sm_in[model_i][2]:cmip6_e3sm_in[model_i][3],:,:].mean(axis=0)/1000
cSoilSlow_cmip6_all = cSoilfile_cmip6.variables['SOIL3C'][:]/1000
cSoilfile_cmip6.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][4])
gppfile_cmip6 = Dataset(cmip6_e3sm_in[model_i][4])
gpp_cmip6 = gppfile_cmip6.variables['gpp'][cmip6_e3sm_in[model_i][5]:cmip6_e3sm_in[model_i][6],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr 
gppfile_cmip6.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][7])
rafile_cmip6 = Dataset(cmip6_e3sm_in[model_i][7])
ra_cmip6 = rafile_cmip6.variables['ra'][cmip6_e3sm_in[model_i][8]:cmip6_e3sm_in[model_i][9],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
rafile_cmip6.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][10])
prfile_cmip6 = Dataset(cmip6_e3sm_in[model_i][10])
pr_cmip6 = prfile_cmip6.variables['pr'][cmip6_e3sm_in[model_i][11]:cmip6_e3sm_in[model_i][12],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
prfile_cmip6.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][13])
tasfile_cmip6 = Dataset(cmip6_e3sm_in[model_i][13])
tas_cmip6 = tasfile_cmip6.variables['tas'][cmip6_e3sm_in[model_i][14]:cmip6_e3sm_in[model_i][15],:,:].mean(axis=0) - 273.15 # K to celcius
lats_forcing_cmip6 = tasfile_cmip6.variables['lat'][:]
lons_forcing_cmip6 = tasfile_cmip6.variables['lon'][:]
tasfile_cmip6.close()
#
###################### ssp585 output
#
print( ' opening file '+cmip6_e3sm_in[model_i][16])
cSoilfile_cmip6_rcp85 = Dataset(cmip6_e3sm_in[model_i][16])
cSoil_cmip6_rcp85 = (cSoilfile_cmip6_rcp85.variables['SOIL1C'][cmip6_e3sm_in[model_i][17]:cmip6_e3sm_in[model_i][18],:,:].mean(axis=0) +
                     cSoilfile_cmip6_rcp85.variables['SOIL2C'][cmip6_e3sm_in[model_i][17]:cmip6_e3sm_in[model_i][18],:,:].mean(axis=0) +
                     cSoilfile_cmip6_rcp85.variables['SOIL3C'][cmip6_e3sm_in[model_i][17]:cmip6_e3sm_in[model_i][18],:,:].mean(axis=0))/1000
cSoil_cmip6_rcp85_all = (cSoilfile_cmip6_rcp85.variables['SOIL1C'][:] +
                         cSoilfile_cmip6_rcp85.variables['SOIL2C'][:] +
                         cSoilfile_cmip6_rcp85.variables['SOIL3C'][:])/1000
time_cmip6_rcp85 = cSoilfile_cmip6_rcp85.variables['time'][:]/365 + 1850
#
cSoilSlow_cmip6_rcp85 = cSoilfile_cmip6_rcp85.variables['SOIL3C'][cmip6_e3sm_in[model_i][17]:cmip6_e3sm_in[model_i][18],:,:].mean(axis=0)/1000
cSoilSlow_cmip6_rcp85_all = cSoilfile_cmip6_rcp85.variables['SOIL3C'][:]/1000
cSoilfile_cmip6_rcp85.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][19])
gppfile_cmip6_rcp85 = Dataset(cmip6_e3sm_in[model_i][19])
gpp_cmip6_rcp85 = gppfile_cmip6_rcp85.variables['gpp'][cmip6_e3sm_in[model_i][20]:cmip6_e3sm_in[model_i][21],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr 
gppfile_cmip6_rcp85.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][22])
rafile_cmip6_rcp85 = Dataset(cmip6_e3sm_in[model_i][22])
ra_cmip6_rcp85 = rafile_cmip6_rcp85.variables['ra'][cmip6_e3sm_in[model_i][23]:cmip6_e3sm_in[model_i][24],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
rafile_cmip6_rcp85.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][25])
prfile_cmip6_rcp85 = Dataset(cmip6_e3sm_in[model_i][25])
pr_cmip6_rcp85 = prfile_cmip6_rcp85.variables['pr'][cmip6_e3sm_in[model_i][26]:cmip6_e3sm_in[model_i][27],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
prfile_cmip6_rcp85.close()
#
print( ' opening file '+cmip6_e3sm_in[model_i][28])
tasfile_cmip6_rcp85 = Dataset(cmip6_e3sm_in[model_i][28])
tas_cmip6_rcp85 = tasfile_cmip6_rcp85.variables['tas'][cmip6_e3sm_in[model_i][29]:cmip6_e3sm_in[model_i][30],:,:].mean(axis=0) - 273.15 # K to celcius
tasfile_cmip6_rcp85.close()
#
###################### variable sums
#
npp_cmip6 = gpp_cmip6 - ra_cmip6
npp_cmip6_rcp85 = gpp_cmip6_rcp85 - ra_cmip6_rcp85
#
del gpp_cmip6, ra_cmip6, gpp_cmip6_rcp85, ra_cmip6_rcp85
#
###################### regrid forcing to 1.9 x 2.5
#
nlons, nlats = np.meshgrid(lons_cmip6, lats_cmip6)
#
tas_cmip6 = basemap.interp(tas_cmip6, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1) # interpolation order = 1 for bilinear interpolation
tas_cmip6_rcp85 = basemap.interp(tas_cmip6_rcp85, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1)
#
npp_cmip6 = basemap.interp(npp_cmip6, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1)
npp_cmip6_rcp85 = basemap.interp(npp_cmip6_rcp85, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1)
#
pr_cmip6 = basemap.interp(pr_cmip6, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1)
pr_cmip6_rcp85 = basemap.interp(pr_cmip6_rcp85, lons_forcing_cmip6, lats_forcing_cmip6, nlons, nlats, order=1)
#
###################### masking
#
common_mask = np.logical_or(pr_cmip6 < 1e1,npp_cmip6 < 1e1)
#
cSoil_cmip6 = np.ma.masked_array(cSoil_cmip6, mask=common_mask)
cSoilSlow_cmip6 = np.ma.masked_array(cSoilSlow_cmip6, mask=common_mask)
npp_cmip6 = np.ma.masked_array(npp_cmip6, mask=common_mask)
pr_cmip6 = np.ma.masked_array(pr_cmip6, mask=common_mask)
tas_cmip6 = np.ma.masked_array(tas_cmip6, mask=common_mask)
#
cSoil_cmip6_rcp85 = np.ma.masked_array(cSoil_cmip6_rcp85, mask=common_mask)
cSoilSlow_cmip6_rcp85 = np.ma.masked_array(cSoilSlow_cmip6_rcp85, mask=common_mask)
npp_cmip6_rcp85 = np.ma.masked_array(npp_cmip6_rcp85, mask=common_mask)
pr_cmip6_rcp85 = np.ma.masked_array(pr_cmip6_rcp85, mask=common_mask)
tas_cmip6_rcp85 = np.ma.masked_array(tas_cmip6_rcp85, mask=common_mask)
#
cSoil_cmip6_change = cSoil_cmip6_rcp85 - cSoil_cmip6 
cSoilSlow_cmip6_change = cSoilSlow_cmip6_rcp85 - cSoilSlow_cmip6
npp_cmip6_change = npp_cmip6_rcp85 - npp_cmip6
pr_cmip6_change = pr_cmip6_rcp85 - pr_cmip6
tas_cmip6_change = tas_cmip6_rcp85 - tas_cmip6
#
ratio_cmip6 = cSoilSlow_cmip6/cSoil_cmip6 
ratio_cmip6_rcp85 = cSoilSlow_cmip6_rcp85/cSoil_cmip6_rcp85
#
cSoilSlow_all = ma.concatenate([cSoilSlow_cmip6_all, cSoilSlow_cmip6_rcp85_all])
cSoil_all = ma.concatenate([cSoil_cmip6_all, cSoil_cmip6_rcp85_all])
time_all = np.concatenate([time_cmip6, time_cmip6_rcp85])
#
print(' resolution = ', round(180/cSoil_all.shape[1],2),'x', round(360/cSoil_all.shape[2],2), 'for years =', cSoil_all.shape[0]/12,'\n')
#
del time_cmip6, time_cmip6_rcp85, cSoil_cmip6_all, cSoil_cmip6_rcp85_all, cSoilSlow_cmip6_all, cSoilSlow_cmip6_rcp85_all
#
###################### appending
#
name_cmip6_allmodels.append(name_cmip6)
#
lats_cmip6_allmodels.append(lats_cmip6)
lons_cmip6_allmodels.append(lons_cmip6)
#
cSoil_cmip6_allmodels.append(cSoil_cmip6)
cSoilSlow_cmip6_allmodels.append(cSoilSlow_cmip6)
npp_cmip6_allmodels.append(npp_cmip6)
pr_cmip6_allmodels.append(pr_cmip6)
tas_cmip6_allmodels.append(tas_cmip6)
#
cSoil_cmip6_rcp85_allmodels.append(cSoil_cmip6_rcp85)
cSoilSlow_cmip6_rcp85_allmodels.append(cSoilSlow_cmip6_rcp85)
npp_cmip6_rcp85_allmodels.append(npp_cmip6_rcp85)
pr_cmip6_rcp85_allmodels.append(pr_cmip6_rcp85)
tas_cmip6_rcp85_allmodels.append(tas_cmip6_rcp85)
#
cSoil_cmip6_change_allmodels.append(cSoil_cmip6_change)
cSoilSlow_cmip6_change_allmodels.append(cSoilSlow_cmip6_change)
npp_cmip6_change_allmodels.append(npp_cmip6_change)
pr_cmip6_change_allmodels.append(pr_cmip6_change)
tas_cmip6_change_allmodels.append(tas_cmip6_change)
#
ratio_cmip6_allmodels.append(ratio_cmip6)
ratio_cmip6_rcp85_allmodels.append(ratio_cmip6_rcp85)
#
cSoil_allmodels.append(cSoil_all)
cSoilSlow_allmodels.append(cSoilSlow_all)
time_allmodels.append(time_all)
#
del name_cmip6, lats_cmip6, lons_cmip6,cSoil_cmip6, cSoilSlow_cmip6, npp_cmip6, pr_cmip6, tas_cmip6
del cSoil_cmip6_rcp85, cSoilSlow_cmip6_rcp85, npp_cmip6_rcp85, pr_cmip6_rcp85, tas_cmip6_rcp85
del cSoil_cmip6_change, cSoilSlow_cmip6_change, npp_cmip6_change, pr_cmip6_change, tas_cmip6_change
del cSoil_all, cSoilSlow_all, time_all, ratio_cmip6, ratio_cmip6_rcp85


# ### Loading testbed

# loading historical and rcp85 output for testbed
#
for i in range(0,3):
    #
    model_i = i
    #
    print( 'Model '+data_in_testbed[model_i][0])
    name_cmip6 = data_in_testbed[model_i][0]
    #
    ###################### loading soil variables
    #
    print( ' opening file '+data_in_testbed[model_i][1])
    cSoilfile_testbed = Dataset(data_in_testbed[model_i][1])
    #
    time_cmip6 = cSoilfile_testbed.variables['time'][:]
    lats_cmip6 = cSoilfile_testbed.variables['lat'][:]
    lons_cmip6 = cSoilfile_testbed.variables['lon'][:]
    #
    if name_cmip6 == 'CASA-CNP':
        cSoil_cmip6_all = (cSoilfile_testbed.variables['csoilmic'][:] + 
                           cSoilfile_testbed.variables['csoilslow'][:] + 
                           cSoilfile_testbed.variables['csoilpass'][:])*(1/1000) # from g to kg
        cSoilSlow_cmip6_all = (cSoilfile_testbed.variables['csoilpass'][:])*(1/1000) # from g to kg
    #
    if name_cmip6 == 'MIMICS':
        cSoil_cmip6_all = (cSoilfile_testbed.variables['cMICr'][:] + 
                           cSoilfile_testbed.variables['cMICk'][:] + 
                           cSoilfile_testbed.variables['cSOMa'][:] + 
                           cSoilfile_testbed.variables['cSOMc'][:] + 
                           cSoilfile_testbed.variables['cSOMp'][:])*(1/1000) # from g to kg 
        cSoilSlow_cmip6_all = (cSoilfile_testbed.variables['cSOMp'][:])*(1/1000) # from g to kg
    #
    if name_cmip6 == 'CORPSE':
        cSoil_cmip6_all = (cSoilfile_testbed.variables['SoilProtected_C1'][:] + 
                               cSoilfile_testbed.variables['SoilProtected_C2'][:] + 
                               cSoilfile_testbed.variables['SoilProtected_C3'][:] + 
                               cSoilfile_testbed.variables['Soil_C1'][:] + 
                               cSoilfile_testbed.variables['Soil_C2'][:] + 
                               cSoilfile_testbed.variables['Soil_C3'][:] + 
                               cSoilfile_testbed.variables['Soil_LiveMicrobeC'][:])*(1/1000) # from g to kg
        cSoilSlow_cmip6_all = (cSoilfile_testbed.variables['SoilProtected_C1'][:] + 
                               cSoilfile_testbed.variables['SoilProtected_C2'][:] + 
                               cSoilfile_testbed.variables['SoilProtected_C3'][:])*(1/1000) # from g to kg
    # 
    cSoilfile_testbed.close()
    #
    cSoil_cmip6 = cSoil_cmip6_all[data_in_testbed[model_i][2]:data_in_testbed[model_i][3],:,:].mean(axis=0)
    cSoilSlow_cmip6 = cSoilSlow_cmip6_all[data_in_testbed[model_i][2]:data_in_testbed[model_i][3],:,:].mean(axis=0)
    #
    cSoil_cmip6_rcp85 = cSoil_cmip6_all[data_in_testbed[model_i][4]:data_in_testbed[model_i][5],:,:].mean(axis=0)
    cSoilSlow_cmip6_rcp85 = cSoilSlow_cmip6_all[data_in_testbed[model_i][4]:data_in_testbed[model_i][5],:,:].mean(axis=0)
    #
    ###################### loading climate variables
    #
    print( ' opening file '+data_in_testbed[model_i][6])
    climatefile_testbed = Dataset(data_in_testbed[model_i][6])
    #
    npp_cmip6 = climatefile_testbed.variables['cnpp'][data_in_testbed[model_i][7]:data_in_testbed[model_i][8],:,:].mean(axis=0) * 365. # per day to year
    tas_cmip6 = climatefile_testbed.variables['tairC'][data_in_testbed[model_i][7]:data_in_testbed[model_i][8],:,:].mean(axis=0) # C

    npp_cmip6_rcp85 = climatefile_testbed.variables['cnpp'][data_in_testbed[model_i][9]:data_in_testbed[model_i][10],:,:].mean(axis=0) * 365. # per day to year
    tas_cmip6_rcp85 = climatefile_testbed.variables['tairC'][data_in_testbed[model_i][9]:data_in_testbed[model_i][10],:,:].mean(axis=0) # C
    #
    climatefile_testbed.close()
    #
    print( ' opening file '+data_in_testbed[model_i][11])
    prfile_testbed = Dataset(data_in_testbed[model_i][11])
    pr_cmip6 = prfile_testbed.variables['RAIN'][data_in_testbed[model_i][12]:data_in_testbed[model_i][13],:,:].mean(axis=0) * 60. * 60. * 24. * 365. #s to yr
    pr_cmip6_rcp85 = pr_cmip6
    prfile_testbed.close()
    #
    ###################### masking
    #
    common_mask = np.logical_or(pr_cmip6 < 1e1,npp_cmip6 < 1e1)
    #
    cSoil_cmip6 = np.ma.masked_array(cSoil_cmip6, mask=common_mask)
    cSoilSlow_cmip6 = np.ma.masked_array(cSoilSlow_cmip6, mask=common_mask)
    npp_cmip6 = np.ma.masked_array(npp_cmip6, mask=common_mask)
    pr_cmip6 = np.ma.masked_array(pr_cmip6, mask=common_mask)
    tas_cmip6 = np.ma.masked_array(tas_cmip6, mask=common_mask)
    #
    cSoil_cmip6_rcp85 = np.ma.masked_array(cSoil_cmip6_rcp85, mask=common_mask)
    cSoilSlow_cmip6_rcp85 = np.ma.masked_array(cSoilSlow_cmip6_rcp85, mask=common_mask)
    npp_cmip6_rcp85 = np.ma.masked_array(npp_cmip6_rcp85, mask=common_mask)
    pr_cmip6_rcp85 = np.ma.masked_array(pr_cmip6_rcp85, mask=common_mask)
    tas_cmip6_rcp85 = np.ma.masked_array(tas_cmip6_rcp85, mask=common_mask)
    #
    cSoil_cmip6_change = cSoil_cmip6_rcp85 - cSoil_cmip6 
    cSoilSlow_cmip6_change = cSoilSlow_cmip6_rcp85 - cSoilSlow_cmip6
    npp_cmip6_change = npp_cmip6_rcp85 - npp_cmip6
    pr_cmip6_change = pr_cmip6_rcp85 - pr_cmip6
    tas_cmip6_change = tas_cmip6_rcp85 - tas_cmip6
    #
    ratio_cmip6 = cSoilSlow_cmip6/cSoil_cmip6 
    ratio_cmip6_rcp85 = cSoilSlow_cmip6_rcp85/cSoil_cmip6_rcp85
    #
    cSoilSlow_all = cSoilSlow_cmip6_all
    cSoil_all = cSoil_cmip6_all
    time_all = time_cmip6
    #
    print(' resolution = ', round(180/cSoil_all.shape[1],2),'x', round(360/cSoil_all.shape[2],2), 'for years =', cSoil_all.shape[0],'\n')
    #
    del time_cmip6, cSoil_cmip6_all, cSoilSlow_cmip6_all
    #
    ###################### appending
    #
    name_cmip6_allmodels.append(name_cmip6)
    #
    lats_cmip6_allmodels.append(lats_cmip6)
    lons_cmip6_allmodels.append(lons_cmip6)
    #
    cSoil_cmip6_allmodels.append(cSoil_cmip6)
    cSoilSlow_cmip6_allmodels.append(cSoilSlow_cmip6)
    npp_cmip6_allmodels.append(npp_cmip6)
    pr_cmip6_allmodels.append(pr_cmip6)
    tas_cmip6_allmodels.append(tas_cmip6)
    #
    cSoil_cmip6_rcp85_allmodels.append(cSoil_cmip6_rcp85)
    cSoilSlow_cmip6_rcp85_allmodels.append(cSoilSlow_cmip6_rcp85)
    npp_cmip6_rcp85_allmodels.append(npp_cmip6_rcp85)
    pr_cmip6_rcp85_allmodels.append(pr_cmip6_rcp85)
    tas_cmip6_rcp85_allmodels.append(tas_cmip6_rcp85)
    #
    cSoil_cmip6_change_allmodels.append(cSoil_cmip6_change)
    cSoilSlow_cmip6_change_allmodels.append(cSoilSlow_cmip6_change)
    npp_cmip6_change_allmodels.append(npp_cmip6_change)
    pr_cmip6_change_allmodels.append(pr_cmip6_change)
    tas_cmip6_change_allmodels.append(tas_cmip6_change)
    #
    ratio_cmip6_allmodels.append(ratio_cmip6)
    ratio_cmip6_rcp85_allmodels.append(ratio_cmip6_rcp85)
    #
    cSoil_allmodels.append(cSoil_all)
    cSoilSlow_allmodels.append(cSoilSlow_all)
    time_allmodels.append(time_all)
    #
    del name_cmip6, lats_cmip6, lons_cmip6,cSoil_cmip6, cSoilSlow_cmip6, npp_cmip6, pr_cmip6, tas_cmip6
    del cSoil_cmip6_rcp85, cSoilSlow_cmip6_rcp85, npp_cmip6_rcp85, pr_cmip6_rcp85, tas_cmip6_rcp85
    del cSoil_cmip6_change, cSoilSlow_cmip6_change, npp_cmip6_change, pr_cmip6_change, tas_cmip6_change
    del cSoil_all, cSoilSlow_all, time_all, ratio_cmip6, ratio_cmip6_rcp85


# ## Loading observational datasets

# ### Loading soil carbon and climate data

# loading observations
#
name_obs = 'Data-Product'
print(name_obs)
#
cru_datafilename = './observations/cru_ts_3.1_climo_1961-1990.nc'
#
gpcc_datafilename = './observations/precip.mon.total.v6.nc'
#
modisnpp_datafilename = './observations/MOD17A3_Science_NPP_mean_00_14_regridhalfdegree.nc'
landcover_datafilename = './observations/MCD12C1.A2012001.051.2013178154403.nc'
#
maoc_soc_datafilename = './observations/global_MOC_CI_1m.nc'
#
#
### load temperature data
#
print( ' opening file '+cru_datafilename)
cru_datafile = Dataset(cru_datafilename)
tas_obs = cru_datafile.variables['tmp'][:].mean(axis=0) # in C with shape = (12,360,720) take annual mean
lats_obs = cru_datafile.variables['lat'][:]
lons_obs = cru_datafile.variables['lon'][:]
cru_datafile.close()
#
###  load precip data
#
print( ' opening file '+gpcc_datafilename)
gpcc_datafile = Dataset(gpcc_datafilename)
#gpcc_time = gpcc_datafile.variables['time'][:] # days since 1800, so starting from gpcc_time[960]/365+1800 = 1981
start_date_index = 960                                                        # to gpcc_time[1319]/365+1800 = 2011
pr_obs = gpcc_datafile.variables['precip'][start_date_index:,::-1,:].mean(axis=0) * 12 # (mm/yr) monthly avg with shape (1320, 360, 720) take annual mean
gpcc_datafile.close()
# 
gpcc_data_temp = pr_obs.copy()
pr_obs[:,0:360] = gpcc_data_temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
pr_obs[:,360:] = gpcc_data_temp[:,0:360]
#
###  load npp data
#
print( ' opening file '+modisnpp_datafilename)
modisnpp_datafile = Dataset(modisnpp_datafilename) 
npp_obs = modisnpp_datafile.variables['npp'][:] # with shape = (360,720) should be in kg/m2/yr
modisnpp_datafile.close()
#
###  load landcover data
#
print( ' opening file '+landcover_datafilename)
landcover_datafile = Dataset(landcover_datafilename) 
landlats = landcover_datafile.variables['latitude'][:]
landlons = landcover_datafile.variables['longitude'][:]
landmap = landcover_datafile.variables['Majority_Land_Cover_Type_1'][:] # in shape (3600,7200)
landcover_datafile.close()
#
landlats_fix = flip(landlats,0)
landlons_fix = landlons
landmap_fix = np.flipud(landmap)
#
nlons, nlats = np.meshgrid(lons_obs, lats_obs)
#
landmap_halfdegree = basemap.interp(landmap_fix, landlons_fix, landlats_fix, nlons, nlats, order=0) # interpolation order = 0 for nearest-neighbor interpolation (order = 1 for bilinear interpolation)
#
landcover = landmap_halfdegree*1
old = landmap_halfdegree*1
#
# including tundra, desert, peatland (now 1 to 10 categories – exclude 8 to 10 for tundra, desert, peatland)
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests (1 to 5, 8) boreal
          & (nlats > 50)] = 1 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests (1 to 5, 8) temperate
          & (nlats < 50) & (nlats > 23)] = 2 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests (1 to 5, 8) temperate
          & (nlats > -50) & (nlats < -23)] = 2 
landcover[((old == 1) | (old == 2) | (old == 3) | (old == 4) | (old == 5) | (old == 8)) #forests (1 to 5, 8) tropical
          & (nlats < 23) & (nlats > -23)] = 3 
landcover[(old == 10) & (nlats < 60)] = 4 #grasslands
landcover[((old == 6) | (old == 7)) & (nlats < 60)] = 5 #shrublands
landcover[(old == 9)] = 6 #savannas
landcover[((old == 12) | (old == 13) | (old == 14))] = 7 #cropland
landcover[(old == 11)] = 10 #wetland/peatland
landcover[(old == 15)] = 0 #snow/ice
landcover[(old == 16)] = 9 #desert
landcover[((old == 0) | (old == 17))] = 0 #water and unclassified
landcover[((old == 6) | (old == 7)) & (nlats > 60)] = 8 #tundra shrubland
landcover[(old == 10) & (nlats > 60)] = 8 #tundra shrubland
#
del landlats_fix, landlons_fix, landlats, landlons, landmap, landmap_fix, landmap_halfdegree, old
#
### load MAOC data and bounds
#
print( ' opening file '+maoc_soc_datafilename)
maoc_soc_data = Dataset(maoc_soc_datafilename)
cSoilSlow_obs = maoc_soc_data.variables['moc'][:]
cSoilSlow_5th = maoc_soc_data.variables['moc_low'][:]
cSoilSlow_95th = maoc_soc_data.variables['moc_high'][:]
cSoil_obs = maoc_soc_data.variables['soc'][:]
maoc_soc_data.close()
#
###################### masking
#
common_mask = np.logical_or(pr_obs < 2e1,npp_obs < 2e1)
#
cSoil_obs = np.ma.masked_array(cSoil_obs, mask=common_mask)
cSoilSlow_obs = np.ma.masked_array(cSoilSlow_obs, mask=common_mask)
npp_obs = np.ma.masked_array(npp_obs, mask=common_mask)
pr_obs = np.ma.masked_array(pr_obs, mask=common_mask)
tas_obs = np.ma.masked_array(tas_obs, mask=common_mask)
landcover = np.ma.masked_array(landcover, mask=common_mask)
#
print(' resolution = ', round(180/cSoil_obs.shape[0],2),'x', round(360/cSoil_obs.shape[1],2),'\n')
#
###################### appending
#
name_cmip6_allmodels.append(name_obs)
#
lats_cmip6_allmodels.append(lats_obs)
lons_cmip6_allmodels.append(lons_obs)
#
cSoil_cmip6_allmodels.append(cSoil_obs)
cSoilSlow_cmip6_allmodels.append(cSoilSlow_obs)
npp_cmip6_allmodels.append(npp_obs)
pr_cmip6_allmodels.append(pr_obs)
tas_cmip6_allmodels.append(tas_obs)
#
cSoil_cmip6_rcp85_allmodels.append(cSoil_obs)
cSoilSlow_cmip6_rcp85_allmodels.append(cSoilSlow_obs)
npp_cmip6_rcp85_allmodels.append(npp_obs)
pr_cmip6_rcp85_allmodels.append(pr_obs)
tas_cmip6_rcp85_allmodels.append(tas_obs)
#
cSoil_cmip6_change_allmodels.append(cSoil_obs*0)
cSoilSlow_cmip6_change_allmodels.append(cSoilSlow_obs*0)
npp_cmip6_change_allmodels.append(npp_obs*0)
pr_cmip6_change_allmodels.append(pr_obs*0)
tas_cmip6_change_allmodels.append(tas_obs*0)
#
ratio_cmip6_allmodels.append(cSoilSlow_obs/cSoil_obs)
ratio_cmip6_rcp85_allmodels.append(cSoilSlow_obs/cSoil_obs*0)
#
cSoil_allmodels.append(cSoil_obs)
cSoilSlow_allmodels.append(cSoilSlow_obs)
time_allmodels.append('2010')
#

# ### Loading auxiliary datasets

## load the organic soil maps
histel_map = './observations/NCSCD_Circumarctic_histel_pct_05deg.nc'
histosol_map = './observations/NCSCD_Circumarctic_histosol_pct_05deg.nc'
#
print(' opening file: '+histel_map)
histel_file = Dataset(histel_map)
histel_pct = histel_file.variables['NCSCD_Circumarctic_histel_pct_05deg.tif'][:]
histel_lats = histel_file.variables['lat'][:]
histel_lons = histel_file.variables['lon'][:]
histel_file.close()
#
print(' opening file: '+histosol_map)
histosol_file = Dataset(histosol_map)
histosol_pct = histosol_file.variables['NCSCD_Circumarctic_histosol_pct_05deg.tif'][:]
histosol_lats = histosol_file.variables['lat'][:]
histosol_lons = histosol_file.variables['lon'][:]
histosol_file.close()
#
max_hist_frac = 50.
#
organics = histel_pct.astype('float') + histosol_pct.astype('float')
#
organics_lat_offset = int((histosol_lats.min() - lats_obs.min() )*2) # number of missing 0.5 degree gridcells -- since same resolution, this is equal to: len(lats_common) - len(histosol_lats)
IM_common = len(lons_obs)
JM_common = len(lats_obs)
organics_commonmap = np.zeros([JM_common, IM_common], dtype=bool)
organics_commonmap[organics_lat_offset:organics_lat_offset+len(histosol_lats),:] = organics[::-1,:] > max_hist_frac
organics_commonmap = np.ma.masked_array(organics_commonmap)
#
### load the PET data
modispet_datafilename = './observations/MOD16A3_Science_PET_mean_00_13_regridhalfdegree.nc'
#
min_pmpet = -1000
#
print( ' opening file '+modispet_datafilename)
modispet_datafile = Dataset(modispet_datafilename)
modispet_data = modispet_datafile.variables['pet'][:]
modispet_lats = modispet_datafile.variables['lat'][:]
modispet_lons = modispet_datafile.variables['lon'][:]
modispet_datafile.close()


## load the soil texture dataset and regridding
clayfilename_t = "./observations/T_CLAY.nc4"
siltfilename_t = "./observations/T_SILT.nc4"
clayfilename_s = "./observations/S_CLAY.nc4"
siltfilename_s = "./observations/S_SILT.nc4"
#
nlons, nlats = np.meshgrid(lons_obs, lats_obs)
#
### load clay + silt data
print(' opening file: '+clayfilename_t)
clayfile_t = Dataset(clayfilename_t, format='NETCDF4')
claylats = clayfile_t.variables['lat'][:]
claylons = clayfile_t.variables['lon'][:]
claymap_in_t = clayfile_t.variables['T_CLAY'][:] # in shape (3600,7200)
clayfile_t.close()
#
claymap_halfdegree_t = basemap.interp(claymap_in_t, claylons, claylats, nlons, nlats, order=1)
#
print(' opening file: '+siltfilename_t)
siltfile_t = Dataset(siltfilename_t, format='NETCDF4')
siltlats = siltfile_t.variables['lat'][:]
siltlons = siltfile_t.variables['lon'][:]
siltmap_in_t = siltfile_t.variables['T_SILT'][:] # in shape (3600,7200)
siltfile_t.close()
#
siltmap_halfdegree_t = basemap.interp(siltmap_in_t, siltlons, siltlats, nlons, nlats, order=1) 
#
print(' opening file: '+clayfilename_s)
clayfile_s = Dataset(clayfilename_s, format='NETCDF4')
claymap_in_s = clayfile_s.variables['S_CLAY'][:] # in shape (3600,7200)
clayfile_s.close()
#
claymap_halfdegree_s = basemap.interp(claymap_in_s, claylons, claylats, nlons, nlats,  order=1) 
#
print(' opening file: '+siltfilename_s)
siltfile_s = Dataset(siltfilename_s, format='NETCDF4')
siltmap_in_s = siltfile_s.variables['S_SILT'][:] # in shape (3600,7200)
siltfile_s.close()
#
siltmap_halfdegree_s = basemap.interp(siltmap_in_s, siltlons, siltlats, nlons, nlats, order=1) 
#
tex_obs = (claymap_halfdegree_t+claymap_halfdegree_s)/2 + (siltmap_halfdegree_t+siltmap_halfdegree_s)/2
#
clay_obs = (claymap_halfdegree_t+claymap_halfdegree_s)/2
silt_obs = (siltmap_halfdegree_t+siltmap_halfdegree_s)/2
#
del claymap_in_t, claylons, claylats, siltmap_in_t, siltlons, siltlats, claymap_in_s, siltmap_in_s
del claymap_halfdegree_t, claymap_halfdegree_s, siltmap_halfdegree_t, siltmap_halfdegree_s
#
tex_obs = np.ma.masked_array(tex_obs, mask=common_mask)
clay_obs = np.ma.masked_array(clay_obs, mask=common_mask)
silt_obs = np.ma.masked_array(silt_obs, mask=common_mask)
#
print(' resolution = ', round(180/tex_obs.shape[0],2),'x', round(360/tex_obs.shape[1],2),'\n')


# ## Checking loaded data

print(name_cmip6_allmodels)

## checking original resolution 
for i in range(0,len(name_cmip6_allmodels)):
    print(cSoil_cmip6_allmodels[i].shape, 'resolution = ', 
          round(180/cSoil_cmip6_allmodels[i].shape[0],2),'x', 
          round(360/cSoil_cmip6_allmodels[i].shape[1],2), 'for '+name_cmip6_allmodels[i])

## fine resolution 
cSoil_cmip6_allmodels_regrid=[]
cSoilSlow_cmip6_allmodels_regrid=[]
cSoil_cmip6_change_allmodels_regrid=[]
cSoilSlow_cmip6_change_allmodels_regrid=[]
ratio_cmip6_allmodels_regrid=[]
#
pr_cmip6_allmodels_regrid=[]
npp_cmip6_allmodels_regrid=[]
tas_cmip6_allmodels_regrid=[]
pr_change_cmip6_allmodels_regrid=[]
npp_change_cmip6_allmodels_regrid=[]
tas_change_cmip6_allmodels_regrid=[]
#
# lat, lon for fine resolution
nlons, nlats = np.meshgrid(lons_obs+180, lats_obs)
#
for i in range(0,len(name_cmip6_allmodels)-1):
    #
    cSoil_regrid = basemap.interp(cSoil_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    cSoilSlow_regrid = basemap.interp(cSoilSlow_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True)
    #
    cSoil_change_regrid = basemap.interp(cSoil_cmip6_change_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    cSoilSlow_change_regrid = basemap.interp(cSoilSlow_cmip6_change_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    #
    ratio_regrid = basemap.interp(ratio_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    #
    pr_regrid = basemap.interp(pr_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    npp_regrid = basemap.interp(npp_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    tas_regrid = basemap.interp(tas_cmip6_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    #
    pr_change_regrid = basemap.interp(pr_cmip6_change_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    npp_change_regrid = basemap.interp(npp_cmip6_change_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True) 
    tas_change_regrid = basemap.interp(tas_cmip6_change_allmodels[i], lons_cmip6_allmodels[i], lats_cmip6_allmodels[i], nlons, nlats, order=0, masked=True)
    #
    # match to to lons_obs, lats_obs
    temp = cSoil_regrid.copy()
    cSoil_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    cSoil_regrid[:,360:] = temp[:,0:360]
    temp = cSoilSlow_regrid.copy()
    cSoilSlow_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    cSoilSlow_regrid[:,360:] = temp[:,0:360]
    #
    temp = cSoil_change_regrid.copy()
    cSoil_change_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    cSoil_change_regrid[:,360:] = temp[:,0:360]
    temp = cSoilSlow_change_regrid.copy()
    cSoilSlow_change_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    cSoilSlow_change_regrid[:,360:] = temp[:,0:360]
    #
    temp = ratio_regrid.copy()
    ratio_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    ratio_regrid[:,360:] = temp[:,0:360]
    #
    temp = pr_regrid.copy()
    pr_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    pr_regrid[:,360:] = temp[:,0:360]
    temp = npp_regrid.copy()
    npp_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    npp_regrid[:,360:] = temp[:,0:360]
    temp = tas_regrid.copy()
    tas_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    tas_regrid[:,360:] = temp[:,0:360]
    #
    temp = pr_change_regrid.copy()
    pr_change_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    pr_change_regrid[:,360:] = temp[:,0:360]
    temp = npp_change_regrid.copy()
    npp_change_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    npp_change_regrid[:,360:] = temp[:,0:360]
    temp = tas_change_regrid.copy()
    tas_change_regrid[:,0:360] = temp[:,360:] # match to (lons_common, lats_common) by moving longitudinally 
    tas_change_regrid[:,360:] = temp[:,0:360]
    #
    cSoil_cmip6_allmodels_regrid.append(cSoil_regrid)
    cSoilSlow_cmip6_allmodels_regrid.append(cSoilSlow_regrid)
    cSoil_cmip6_change_allmodels_regrid.append(cSoil_change_regrid)
    cSoilSlow_cmip6_change_allmodels_regrid.append(cSoilSlow_change_regrid)
    ratio_cmip6_allmodels_regrid.append(ratio_regrid)
    pr_cmip6_allmodels_regrid.append(pr_regrid)
    npp_cmip6_allmodels_regrid.append(npp_regrid)
    tas_cmip6_allmodels_regrid.append(tas_regrid)
    pr_change_cmip6_allmodels_regrid.append(pr_change_regrid)
    npp_change_cmip6_allmodels_regrid.append(npp_change_regrid)
    tas_change_cmip6_allmodels_regrid.append(tas_change_regrid)
    #
    print(cSoil_regrid.shape, 'resolution = ', 
          round(180/cSoil_regrid.shape[0],2),'x', 
          round(360/cSoil_regrid.shape[1],2), 'for '+name_cmip6_allmodels[i])

# calculating means
cSoilmean = np.mean(cSoil_cmip6_allmodels_regrid, axis=0)
cSoilStd = np.std(cSoil_cmip6_allmodels_regrid, axis=0, ddof=1)
#
cSoilSlowmean = np.mean(cSoilSlow_cmip6_allmodels_regrid, axis=0)
cSoilSlowStd = np.std(cSoilSlow_cmip6_allmodels_regrid, axis=0, ddof=1)
#
ratiomean = np.mean(ratio_cmip6_allmodels_regrid, axis=0)
ratioStd = np.std(ratio_cmip6_allmodels_regrid, axis=0, ddof=1)
#
pr_mean = np.mean(pr_cmip6_allmodels_regrid, axis=0)
pr_Std = np.std(pr_cmip6_allmodels_regrid, axis=0, ddof=1)
#
npp_mean = np.mean(npp_cmip6_allmodels_regrid, axis=0)
npp_Std = np.std(npp_cmip6_allmodels_regrid, axis=0, ddof=1)
#
tas_mean = np.mean(tas_cmip6_allmodels_regrid, axis=0)
tas_Std = np.std(tas_cmip6_allmodels_regrid, axis=0, ddof=1)
#
cSoil_changemean = np.mean(cSoil_cmip6_change_allmodels_regrid, axis=0)
cSoil_changeStd = np.std(cSoil_cmip6_change_allmodels_regrid, axis=0, ddof=1)
#
cSoilSlow_changemean = np.mean(cSoilSlow_cmip6_change_allmodels_regrid, axis=0)
cSoilSlow_changeStd = np.std(cSoilSlow_cmip6_change_allmodels_regrid, axis=0, ddof=1)
#
pr_changemean = np.mean(pr_change_cmip6_allmodels_regrid, axis=0)
pr_changeStd = np.std(pr_change_cmip6_allmodels_regrid, axis=0, ddof=1)
#
npp_changemean = np.mean(npp_change_cmip6_allmodels_regrid, axis=0)
npp_changeStd = np.std(npp_change_cmip6_allmodels_regrid, axis=0, ddof=1)
#
tas_changemean = np.mean(tas_change_cmip6_allmodels_regrid, axis=0)
tas_changeStd = np.std(tas_change_cmip6_allmodels_regrid, axis=0, ddof=1)
#
common_mask = np.logical_or(pr_obs < 1e2, npp_obs < 1e1)
common_mask = np.logical_or(common_mask[:], pr_mean < 1e2)
common_mask = np.logical_or(common_mask[:], npp_mean < 1e1)
#
cSoilmean = np.ma.masked_array(cSoilmean, mask=common_mask)
cSoilStd = np.ma.masked_array(cSoilStd, mask=common_mask)
cSoilSlowmean = np.ma.masked_array(cSoilSlowmean, mask=common_mask)
cSoilSlowStd = np.ma.masked_array(cSoilSlowStd, mask=common_mask)
ratiomean = np.ma.masked_array(ratiomean, mask=common_mask)
ratioStd = np.ma.masked_array(ratioStd, mask=common_mask)
pr_mean = np.ma.masked_array(pr_mean, mask=common_mask)
pr_Std = np.ma.masked_array(pr_Std, mask=common_mask)
npp_mean = np.ma.masked_array(npp_mean, mask=common_mask)
npp_Std = np.ma.masked_array(npp_Std, mask=common_mask)
tas_mean = np.ma.masked_array(tas_mean, mask=common_mask)
tas_Std = np.ma.masked_array(tas_Std, mask=common_mask)
#
cSoil_changemean = np.ma.masked_array(cSoil_changemean, mask=common_mask)
cSoil_changeStd = np.ma.masked_array(cSoil_changeStd, mask=common_mask)
cSoilSlow_changemean = np.ma.masked_array(cSoilSlow_changemean, mask=common_mask)
cSoilSlow_changeStd = np.ma.masked_array(cSoilSlow_changeStd, mask=common_mask)
pr_changemean = np.ma.masked_array(pr_changemean, mask=common_mask)
pr_changeStd = np.ma.masked_array(pr_changeStd, mask=common_mask)
npp_changemean = np.ma.masked_array(npp_changemean, mask=common_mask)
npp_changeStd = np.ma.masked_array(npp_changeStd, mask=common_mask)
tas_changemean = np.ma.masked_array(tas_changemean, mask=common_mask)
tas_changeStd = np.ma.masked_array(tas_changeStd, mask=common_mask)


# ## Plotting proportion protected

print(name_cmip6_allmodels)

# ### mapping globally

mlons, mlats = np.meshgrid(lons_obs, lats_obs)
#
common_mask = np.logical_or(pr_obs < 1e2, npp_obs < 1e1)
common_mask = np.logical_or(common_mask[:], pr_mean < 1e2)
common_mask = np.logical_or(common_mask[:], npp_mean < 1e1)
common_mask = np.logical_or(common_mask[:], landcover > 7.5) # exclude tundra, desert, peatland
common_mask = np.logical_or(common_mask[:], organics_commonmap[:]) # histosols
common_mask = np.logical_or(common_mask[:], (pr_obs/modispet_data)[:] < 0.05) # aridity index
#
cSoil_obs_mask = np.ma.masked_array(cSoil_obs, mask=common_mask)
cSoilSlow_obs_mask = np.ma.masked_array(cSoilSlow_obs, mask=common_mask)
cSoilSlow_5th_mask = np.ma.masked_array(cSoilSlow_5th, mask=common_mask)
cSoilSlow_95th_mask = np.ma.masked_array(cSoilSlow_95th, mask=common_mask)
cSoilmean_mask = np.ma.masked_array(cSoilmean, mask=common_mask)
cSoilSlowmean_mask = np.ma.masked_array(cSoilSlowmean, mask=common_mask)


### mapping data product and multi-model mean
fig = plt.figure(figsize=(17,9))
plt.subplot(221)
# mapping data product
#
cmap = truncate_bicolormap(cmapIn='BrBG_r', minval=0.1, mid1=0.47, mid2=0.53, maxval=0.9) 
cmap.set_bad('white')
bounds = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
m = Basemap(projection='robin',lon_0=0,resolution='c') # 'robin', 'moll', 'kav7', 'mbtfpq'
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(mlons, mlats, 100*cSoilSlow_obs_mask/cSoil_obs_mask, cmap=cmap, norm=norm, latlon=True)
m.drawparallels(np.arange(-90.,91.,180.), linewidth=1, labels=[False,False,False,False], dashes=[1,0])
m.drawmeridians(np.arange(-180.,181.,360.), linewidth=1, labels=[False,False,False,False], dashes=[1,0])
m.drawparallels(np.arange(-90.,91.,30.), linewidth=0.7, labels=[True,False,False,False], dashes=[2,2], fontsize=12)
m.drawmeridians(np.arange(-180.,181.,60.), linewidth=0.7, labels=[False,False,False,True], dashes=[2,2], fontsize=12)
#
plt.subplot(223)
# mapping model mean
#
m = Basemap(projection='robin',lon_0=0,resolution='c')    
m.drawmapboundary(color='k', fill_color='none')
m.drawcoastlines(color='k', linewidth=0.4)
im1 = m.pcolormesh(mlons, mlats, 100*cSoilSlowmean_mask/cSoilmean_mask, norm=norm, cmap=cmap, latlon=True)
cb = m.colorbar(im1, size="4%", pad="12%", location='bottom')
cb.ax.tick_params(labelsize=13)
cb.set_label('Proportion protected C (%)', fontsize=15)
m.drawparallels(np.arange(-90.,91.,180.), linewidth=1, labels=[False,False,False,False], dashes=[1,0])
m.drawmeridians(np.arange(-180.,181.,360.), linewidth=1, labels=[False,False,False,False], dashes=[1,0])
m.drawparallels(np.arange(-90.,91.,30.), linewidth=0.7, labels=[True,False,False,False], dashes=[2,2], fontsize=12)
m.drawmeridians(np.arange(-180.,181.,60.), linewidth=0.7, labels=[False,False,False,True], dashes=[2,2], fontsize=12)
#
plt.subplots_adjust(wspace=0.3, hspace=0.05)
plt.show()

# ### mapping model output

# mapping coarse output
totmodels = len(name_cmip6_allmodels)
#
for i in range(0,totmodels-1): 
    #
    model_i = i
    #
    name_cmip6 = name_cmip6_allmodels[model_i]
    print(name_cmip6)
    #
    lats_cmip6 = lats_cmip6_allmodels[model_i]
    lons_cmip6 = lons_cmip6_allmodels[model_i]
    #
    mlons, mlats = np.meshgrid(lons_cmip6, lats_cmip6)
    #
    cSoil_cmip6 = cSoil_cmip6_allmodels[model_i]
    cSoilSlow_cmip6 = cSoilSlow_cmip6_allmodels[model_i]
    #
    # mapping model output
    fig = plt.figure(num=None, figsize=(6, 4))
    m = Basemap(projection='gall',llcrnrlat=-65,urcrnrlat=90,llcrnrlon=-170,urcrnrlon=190,resolution='c')
    m.drawmapboundary(color='k', fill_color='none')
    m.drawcoastlines(color='k', linewidth=0.4)
    bounds = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = truncate_bicolormap(cmapIn='BrBG_r', minval=0.1, mid1=0.47, mid2=0.53, maxval=0.9) 
    cmap.set_bad('white')
    im1 = m.pcolormesh(mlons, mlats, 100*cSoilSlow_cmip6/cSoil_cmip6, cmap=cmap, norm=norm, latlon=True)
    #cb = m.colorbar(im1, size="2%", pad="4%")
    #cb.ax.tick_params(labelsize=13)
    #cb.set_label('Proportion protected C (%)', fontsize=15)
    plt.show()
    #fig.savefig("./ratio_%s.png" % name_cmip6, bbox_inches='tight', dpi=600)

# ### plotting global means

model_order = [12, 0, 1, 2, 3, 8, 4, 5, 6, 7, 9, 10, 11] # reordering for figure
#
cSoilSlow_total=[]
cSoil_total=[]
ratio_total_mean=[]
names_total=[]
ratio_rcp85_total_mean=[]
cSoil_change_total=[]
cSoilSlow_change_total=[]
#
common_mask = np.logical_or(pr_obs < 1e2, npp_obs < 1e1)
common_mask = np.logical_or(common_mask[:], pr_mean < 1e2)
common_mask = np.logical_or(common_mask[:], npp_mean < 1e1)
common_mask = np.logical_or(common_mask[:], landcover > 7.5) # exclude tundra, desert, peatland
common_mask = np.logical_or(common_mask[:], organics_commonmap[:]) # histosols
common_mask = np.logical_or(common_mask[:], (pr_obs/modispet_data)[:] < 0.05) # aridity index
#
#
area_ud = global_area(lats_obs, lons_obs)
#
i = 0
model_i = model_order[i]
names_total.append(name_cmip6_allmodels[model_i])
print(name_cmip6_allmodels[model_i])
#
cSoil_cmip6 = cSoil_cmip6_allmodels[model_i]
cSoilSlow_cmip6 = cSoilSlow_cmip6_allmodels[model_i]
ratio_cmip6 = cSoilSlow_cmip6/cSoil_cmip6
#
cSoilSlow_cmip6_mask = np.ma.MaskedArray(cSoilSlow_cmip6, mask=common_mask)
cSoil_cmip6_mask = np.ma.MaskedArray(cSoil_cmip6, mask=common_mask)
#
cSoilSlow_total.append(round(np.sum(cSoilSlow_cmip6_mask * area_ud)/1e12,2))
cSoil_total.append(round(np.sum(cSoil_cmip6_mask * area_ud)/1e12,2))
#
ratio_cmip6_mask = np.ma.MaskedArray(ratio_cmip6, mask=common_mask)
ratio_total_mean.append(round(np.ma.average(ratio_cmip6_mask, weights=area_ud),2))
#
cSoilSlow_5th_mask = np.ma.MaskedArray(cSoilSlow_5th, mask=common_mask)
cSoilSlow_95th_mask = np.ma.MaskedArray(cSoilSlow_95th, mask=common_mask)
cSoil_obs_mask = np.ma.MaskedArray(cSoil_cmip6, mask=common_mask)
#
cSoilSlow_change_total.append(round(np.sum((np.ma.MaskedArray(cSoilSlow_cmip6_change_allmodels[model_i], mask=common_mask)) * area_ud)/1e12,2))
cSoil_change_total.append(round(np.sum((np.ma.MaskedArray(cSoil_cmip6_change_allmodels[model_i], mask=common_mask)) * area_ud)/1e12,2))
#
ratio_rcp85_cmip6 = cSoilSlow_cmip6_rcp85_allmodels[model_i]/cSoil_cmip6_rcp85_allmodels[model_i]
ratio_rcp85_total_mean.append(round(np.ma.average(np.ma.MaskedArray(ratio_rcp85_cmip6, mask=common_mask), weights=area_ud),2))
#
totmodels = len(model_order)
#
for i in range(1,totmodels):
    model_i = model_order[i]
    #
    names_total.append(name_cmip6_allmodels[model_i])
    print(name_cmip6_allmodels[model_i])
    #
    cSoil_cmip6 = cSoil_cmip6_allmodels_regrid[model_i]
    cSoilSlow_cmip6 = cSoilSlow_cmip6_allmodels_regrid[model_i]
    ratio_cmip6 = cSoilSlow_cmip6/cSoil_cmip6
    #
    cSoilSlow_cmip6_mask = np.ma.MaskedArray(cSoilSlow_cmip6, mask=common_mask)
    cSoil_cmip6_mask = np.ma.MaskedArray(cSoil_cmip6, mask=common_mask)
    #
    cSoilSlow_total.append(round(np.sum(cSoilSlow_cmip6_mask * area_ud)/1e12,2))
    cSoil_total.append(round(np.sum(cSoil_cmip6_mask * area_ud)/1e12,2))
    #
    ratio_cmip6_mask = np.ma.MaskedArray(ratio_cmip6, mask=common_mask)
    ratio_total_mean.append(round(np.ma.average(ratio_cmip6_mask, weights=area_ud),2))
    #
    cSoil_change_cmip6 = cSoil_cmip6_change_allmodels_regrid[model_i]
    cSoilSlow_change_cmip6 = cSoilSlow_cmip6_change_allmodels_regrid[model_i]
    ratio_rcp85_cmip6 = (cSoilSlow_cmip6+cSoilSlow_change_cmip6)/(cSoil_cmip6+cSoil_change_cmip6)
    #
    cSoilSlow_change_total.append(round(np.sum((np.ma.MaskedArray(cSoilSlow_cmip6_change_allmodels_regrid[model_i], mask=common_mask)) * area_ud)/1e12,2))
    cSoil_change_total.append(round(np.sum((np.ma.MaskedArray(cSoil_cmip6_change_allmodels_regrid[model_i], mask=common_mask)) * area_ud)/1e12,2))
    #
    ratio_rcp85_total_mean.append(round(np.ma.average(np.ma.MaskedArray(ratio_rcp85_cmip6, mask=common_mask), weights=area_ud),2))

# plotting global proportion protected
N = totmodels
ind = np.arange(N)
#
fig, ax = plt.subplots(nrows=1)
plt.axis([-0.5, N-0.5, 0, 100])
#
plt.plot(ind[0], (np.asarray(ratio_total_mean)*100)[0], linestyle='None', marker='o', c='k', markersize=10, label='Mineral-\nassociated', markeredgecolor='k')
#
area_ud = global_area(lats_obs, lons_obs)
ratio_obs_low = round(np.ma.average(cSoilSlow_5th_mask/cSoil_obs_mask, weights=area_ud),2)*100
ratio_obs_high = round(np.ma.average(cSoilSlow_95th_mask/cSoil_obs_mask, weights=area_ud),2)*100
#
ratio_obs_mean = ratio_total_mean[0]*100
#
lower_error =  ratio_obs_mean-ratio_obs_low
upper_error =  ratio_obs_high-ratio_obs_mean
plt.errorbar(ind[0], ratio_obs_mean, yerr=[[lower_error], [upper_error]], fmt='o', linestyle='None', marker='o', ecolor='k', c='k', markersize=9, capsize=5)
#
plt.plot(ind[1:11], (np.asarray(ratio_total_mean)*100)[1:11], linestyle='None', marker='o', c='grey', markersize=10, label='Passive', markeredgecolor='k')
plt.plot(ind[11:13], (np.asarray(ratio_total_mean)*100)[11:13], linestyle='None', marker='o', c='lightgrey', markersize=10, label='Physicochemically \nprotected', markeredgecolor='k')
#
plt.vlines([0.5], [0], [100], linestyles='solid', colors='k',linewidth=1 )
#
ax.set_xticks(np.arange(N))
ax.tick_params(labelsize=14)
ax.set_xticklabels(names_total, rotation=90, ha='center', fontsize=13)
ax.set_ylabel('Proportion of protected C (%)', fontsize=18)
ax.legend(fontsize=13, bbox_to_anchor=(0.98,  0.68), loc='upper left', frameon=False)
plt.show()

# ### plotting Hovmöller diagrams 

totmodels = len(name_cmip6_allmodels)
#
for i in range(0,totmodels-1):
    #
    model_i = i
    #
    name_cmip6 = name_cmip6_allmodels[model_i]
    lat_cSoilSlow_all = np.nanmean((cSoilSlow_allmodels[model_i]), axis=2)
    lat_cSoil_all = np.nanmean((cSoil_allmodels[model_i]), axis=2)
    #
    latsmesh, timemesh = np.meshgrid(lats_cmip6_allmodels[model_i], time_allmodels[model_i]) #+1850
    #
    # make these smaller to increase the resolution
    dx, dy = 0.1, 0.1
    #
    # generate 2 2d grids for the x & y bounds
    y = latsmesh
    x = timemesh
    #
    cmap = plt.get_cmap('PiYG')
    #
    fig, (ax0) = plt.subplots(nrows=1,figsize=(6,2))
    #
    if ((name_cmip6 == 'CASA-CNP') or (name_cmip6 == 'MIMICS') or (name_cmip6 == 'CORPSE')):
        z = lat_cSoilSlow_all - lat_cSoilSlow_all[0:1]
    else:
        z = lat_cSoilSlow_all - lat_cSoilSlow_all[((1900-1850)*12):((1900-1850)*12+1)] # change since 1900
    #
    z = z[:-1, :-1]
    im = ax0.pcolormesh(x, y, z, cmap=cmap, vmin=-0.5, vmax=0.5)
    cb = fig.colorbar(im, ax=ax0)
    ax0.set_title(name_cmip6)
    ax0.set_ylabel('Latitude')
    cb.set_label('Change in protected C \n(kg C m$^{-2}$)')
    ax0.axis([1900,2100,-50, 75])
    #
    fig.tight_layout()
    plt.show()
    #fig.savefig("./hovmoller_%s.png" % name_cmip6, bbox_inches='tight', dpi=600)

# ## Saving output csv files

# saving model output
for i in range(0,len(name_cmip6_allmodels)-1):
    model_i = i
    #
    name_cmip6 = name_cmip6_allmodels[model_i]
    print(name_cmip6)
    #
    nlons, nlats = np.meshgrid(lons_obs, lats_obs)
    #
    lats_cmip6 = (nlats).flatten()
    lons_cmip6 = (nlons).flatten()
    #
    area_ud = global_area(lats_obs, lons_obs)
    area_cmip6 = (area_ud).flatten()
    #
    cSoil_cmip6 = (cSoil_cmip6_allmodels_regrid[model_i]).flatten()
    cSoilSlow_cmip6 = (cSoilSlow_cmip6_allmodels_regrid[model_i]).flatten()
    npp_cmip6 = (npp_cmip6_allmodels_regrid[model_i]).flatten()
    pr_cmip6 = (pr_cmip6_allmodels_regrid[model_i]).flatten()
    tas_cmip6 = (tas_cmip6_allmodels_regrid[model_i]).flatten()
    #
    cSoil_cmip6_change = (cSoil_cmip6_change_allmodels_regrid[model_i]).flatten()
    cSoilSlow_cmip6_change = (cSoilSlow_cmip6_change_allmodels_regrid[model_i]).flatten()
    npp_cmip6_change = (npp_change_cmip6_allmodels_regrid[model_i]).flatten()
    pr_cmip6_change = (pr_change_cmip6_allmodels_regrid[model_i]).flatten()
    tas_cmip6_change = (tas_change_cmip6_allmodels_regrid[model_i]).flatten()
    #
    biome = (landcover).flatten()
    clay = (clay_obs).flatten()
    silt = (silt_obs).flatten()
    #
    combined = [lats_cmip6, lons_cmip6, cSoil_cmip6, cSoilSlow_cmip6, npp_cmip6, pr_cmip6, tas_cmip6,
               cSoil_cmip6_change, cSoilSlow_cmip6_change, npp_cmip6_change, pr_cmip6_change, tas_cmip6_change,
               biome, clay, silt, area_cmip6]
    combined = np.transpose(np.asarray(combined))
    #
    df = pd.DataFrame(combined)
    #df.to_csv('./ESM_output_%s.csv' % name_cmip6, index=False, 
    #          header=['lats','lons','cSoil','cSoilSlow','npp','pr','tas',
    #                  'change_cSoil','change_cSoilSlow','change_npp','change_pr','change_tas',
    #                  'landcover','clay','silt','grid_area'])


# saving data product only
model_i = 12
#
name_cmip6 = name_cmip6_allmodels[model_i]
print(name_cmip6)
#
lats = lats_cmip6_allmodels[model_i]
lon = lons_cmip6_allmodels[model_i]
#
mlons, mlats = np.meshgrid(lons_obs, lats_obs)
lats_cmip6 = (mlats).flatten()
lons_cmip6 = (mlons).flatten()
#
area_ud = global_area(lats_obs, lons_obs)
area_cmip6 = (area_ud).flatten()
#
cSoil_cmip6 = (cSoil_cmip6_allmodels[model_i]).flatten()
cSoilSlow_cmip6 = (cSoilSlow_cmip6_allmodels[model_i]).flatten()
npp_cmip6 = (npp_cmip6_allmodels[model_i]).flatten()
pr_cmip6 = (pr_cmip6_allmodels[model_i]).flatten()
tas_cmip6 = (tas_cmip6_allmodels[model_i]).flatten()
#
cSoil_cmip6_change = (cSoil_cmip6_change_allmodels[model_i]).flatten()
cSoilSlow_cmip6_change = (cSoilSlow_cmip6_change_allmodels[model_i]).flatten()
npp_cmip6_change = (npp_cmip6_change_allmodels[model_i]).flatten()
pr_cmip6_change = (pr_cmip6_change_allmodels[model_i]).flatten()
tas_cmip6_change = (tas_cmip6_change_allmodels[model_i]).flatten()
#
biome = (landcover).flatten()
clay = (clay_obs).flatten()
silt = (silt_obs).flatten()
#
combined = [lats_cmip6, lons_cmip6, cSoil_cmip6, cSoilSlow_cmip6, npp_cmip6, pr_cmip6, tas_cmip6,
           cSoil_cmip6_change, cSoilSlow_cmip6_change, npp_cmip6_change, pr_cmip6_change, tas_cmip6_change,
           biome, clay, silt, area_cmip6]
combined = np.transpose(np.asarray(combined))
#
df = pd.DataFrame(combined)
#df.to_csv('./ESM_output_%s.csv' % name_cmip6, index=False, 
#          header=['lats','lons','cSoil','cSoilSlow','npp','pr','tas',
#                  'change_cSoil','change_cSoilSlow','change_npp','change_pr','change_tas','landcover','clay','silt','grid_area'])

