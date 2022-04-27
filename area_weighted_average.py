#!/usr/bin/env python
# coding: utf-8

# # Computing area weighted average for global/zonal means (& monthly anomalies)

# #### Functions for area weighting

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from numpy import meshgrid, deg2rad, gradient, cos
import warnings


# In[ ]:


# import file(s)
ds = xr.load_dataset('/path/to/dataset_file.nc')
df = pd.read_csv('/path/to/file.txt')


# In[2]:


# Calculation for grid cell area
def area_grid(lat, lon):
    
    """
    Calculate the area of each grid cell
    (lat, lon)--> (lat, lon), grid-cell area (m**2)
    """

    x, y = meshgrid(lon, lat)
    R = radius(y)
    rlon = deg2rad(gradient(x, axis=1))
    rlat = deg2rad(gradient(y, axis=0))
    rx = R*rlon*cos(deg2rad(y))
    ry = R*rlat
  
    area = ry*rx

    data_array = DataArray(
        area, dims=["lat", "lon"], 
        coords={"lat": lat, "lon": lon})
    return data_array


# In[3]:


# Note: this script **does not** make the assumption that Earth is a perfect sphere. It is more accurately defined as a spheroid below
def radius(lat):
    
    '''
    calculate radius of Earth using lat (degrees)
    lat--> radius(m)
    '''
    
    # Since Earth isn't a perfect sphere, must define the spheroid (sphere-like object)
    a = 6378137.0 #equatorial radius (m)
    b = 6356752.3142 #radius at poles (m)
    e2 = 1 - (b**2/a**2)
    
    # convert to geocentric
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # calculate radius
    r = ((a * (1 - e2)**0.5)/(1 - (e2 * np.cos(lat_gc)**2))**0.5)

    return r


# #### Addressing gaps in the data (if any) 
# It is very important to identify if there are gaps in the data before proceeding. If this is a complete dataset, then proceed. If the gaps are somewhat insignificant, perhaps try "nearest neighbor" interpolation or another technique. For larger data gaps, a more sophisticated statistical technique for weighting may be necessary.

# *Simple interpolation example using np.interp*

# In[23]:


# Convert gaps to NaN (if they aren't NaN already) and interpolate these values in the next step. 
# Below is an example approach using a pandas dataframe. If working with a dataset, the approach will not be much different
## Note: gaps may be stored as -9999 or another number. A quick way to check is to plot a time series of the data and visually see the extremes that don't fit
## Another note: if area-weighting a variable other than temp, replace 'temp' with variable name
df['temp'] = df['temp'].replace(9999, np.nan)


# In[25]:


# Interpolate NaN temp values
temp_nan = df['temp']
mask = np.isnan(temp_nan)
temp_nan[mask] = np.interp(np.flatnonzero(mask), 
                       np.flatnonzero(~mask), 
                       temp_nan[~mask])

df['temp'] = temp_nan


# #### Compute weights

# In[162]:


# If working with a dataframe, convert the df to an xarray dataset first. Or manipulate the code below to be compatible with the dataframe.

# Example of df to ds conversion 
# Step 1: set index of dataframe
df = df.set_index(['lon', 'lat', 'time'])
# Step 2: convert to xarray dataset
ds = df.to_xarray()


# In[ ]:


# Compute weighting where 'ds' is the name of the dataset. replace ds with your dataset name. 

# Area dataArray
array_area = area_grid(ds['lat'], ds['lon'])
# Total area
total_area = array_area.sum(['lat','lon'])
# Area-weighted temp
temp_weighted = (ds['temp']*array_area)/total_area
# weighting
weight_tot = (array_area/total_area).sum(('lat','lon'))
# global average. If you are only interested in the global mean temp, the calculation is simple. Just sum all weighted values to get the average.
global_mean = temp_weighted.sum(dim=('lat','lon'))


# #### Define zones and compute averages for anomaly calculation

# In[178]:


# define area array for cells and sum for each zone
# this will compute the weighting for each zone. this will be used to get the nominal zonal averages 
# variable interpretation: ex. N24N90 = from 24°N to 90°N, S24N24 = from 24°S to 24°N 

NHem = (array_area.where(array_area['lat'] >= 0)/total_area).sum(('lat','lon'))
SHem = (array_area.where(array_area['lat'] <= 0)/total_area).sum(('lat','lon'))
N24N90 = (array_area.where((array_area['lat'] >= 24) & (array_area['lat'] <= 90))/total_area).sum(('lat','lon'))
S24N24 = (array_area.where((array_area['lat'] >= -24) & (array_area['lat'] <= 24))/total_area).sum(('lat','lon'))
S24S90 = (array_area.where((array_area['lat'] >= -90) & (array_area['lat'] <= -24))/total_area).sum(('lat','lon'))
N64N90 = (array_area.where((array_area['lat'] >= 64) & (array_area['lat'] <= 90))/total_area).sum(('lat','lon'))
N44N64 = (array_area.where((array_area['lat'] >= 44) & (array_area['lat'] <= 64))/total_area).sum(('lat','lon'))
N24N44 = (array_area.where((array_area['lat'] >= 24) & (array_area['lat'] <= 44))/total_area).sum(('lat','lon'))
EQUN24 = (array_area.where((array_area['lat'] >= 0) & (array_area['lat'] <= 24))/total_area).sum(('lat','lon'))
EQUS24 = (array_area.where((array_area['lat'] >= -24) & (array_area['lat'] <= 0))/total_area).sum(('lat','lon'))
S24S44 = (array_area.where((array_area['lat'] >= -44) & (array_area['lat'] <= -24))/total_area).sum(('lat','lon'))
S44S64 = (array_area.where((array_area['lat'] >= -64) & (array_area['lat'] <= -44))/total_area).sum(('lat','lon'))
S64S90 = (array_area.where((array_area['lat'] >= -90) & (array_area['lat'] <= -64))/total_area).sum(('lat','lon'))


# In[179]:


# Mask by zone. Zones are broken up as follows: {tropics:(0-24), subtropics:(24-44), extratropics=(44-64), polar:(64-90)}
# NOTE: zonal values summed will = global mean. Zonal values as is need to be divided by the respective zonal weighting to get nominal zonal average

n_polar = temp_weighted.where((temp_weighted['lat'] >= 64.0) & (temp_weighted['lat'] <= 90.0), 
                             drop=True).sum(dim=('lat', 'lon'))
n_extratropics = temp_weighted.where((temp_weighted['lat'] >= 44.0) & (temp_weighted['lat'] <= 64.0), 
                                 drop=True).sum(dim=('lat', 'lon'))
n_subtropics = temp_weighted.where((temp_weighted['lat'] >= 24) & (temp_weighted['lat'] <= 44.0), 
                                   drop=True).sum(dim=('lat', 'lon'))
n_tropics = temp_weighted.where((temp_weighted['lat'] >= 0.0) & (temp_weighted['lat'] <= 24), 
                                  drop=True).sum(dim=('lat', 'lon'))
s_tropics = temp_weighted.where((temp_weighted['lat'] >= -24) & (temp_weighted['lat'] <= 0.0), 
                                  drop=True).sum(dim=('lat', 'lon'))
s_subtropics = temp_weighted.where((temp_weighted['lat'] >= -44) & (temp_weighted['lat'] <= -24), 
                                 drop=True).sum(dim=('lat', 'lon'))
s_extratropics = temp_weighted.where((temp_weighted['lat'] >= -64.0) & (temp_weighted['lat'] <= -44.0), 
                                   drop=True).sum(dim=('lat', 'lon'))
s_polar = temp_weighted.where((temp_weighted['lat'] >= -90.0) & (temp_weighted['lat'] <= -64.0), 
                                   drop=True).sum(dim=('lat', 'lon'))


# In[180]:


# To compute monthly anomalies for a given period, calculate the mean for the base period. 
# Note: using a base period of 2007-2016. Change base period if necessary
base = temp_weighted.sel(time=slice("2007-01-01", "2016-12-31"))


# In[181]:


# Calculate base mean temperature values for zone and time, **weighted values, not nominal** 
n_polar_base = base.where((base['lat'] >= 64.0) & (base['lat'] <= 90.0), 
                             drop=True).sum(dim=('lat', 'lon'))
n_extratropics_base = base.where((base['lat'] >= 44.0) & (base['lat'] <= 64.0), 
                                 drop=True).sum(dim=('lat', 'lon'))
n_subtropics_base = base.where((base['lat'] >= 24) & (base['lat'] <= 44.0), 
                                   drop=True).sum(dim=('lat', 'lon'))
n_tropics_base = base.where((base['lat'] >= 0.0) & (base['lat'] <= 24), 
                                  drop=True).sum(dim=('lat', 'lon'))
s_tropics_base = base.where((base['lat'] >= -24) & (base['lat'] <= 0.0), 
                                  drop=True).sum(dim=('lat', 'lon'))
s_subtropics_base = base.where((base['lat'] >= -44) & (base['lat'] <= -24), 
                                 drop=True).sum(dim=('lat', 'lon'))
s_extratropics_base = base.where((base['lat'] >= -64.0) & (base['lat'] <= -44.0), 
                                   drop=True).sum(dim=('lat', 'lon'))
s_polar_base = base.where((base['lat'] >= -90.0) & (base['lat'] <= -64.0), 
                                   drop=True).sum(dim=('lat', 'lon'))


# #### Compute anomalies

# In[182]:


# Compute climatological mean for the base period
# Subtract that value from the monthly mean weighted temp for each month in each zone to get monthly anom
# Note: using base period and not using any kind of running mean. If you want to compute the anomaly using a running mean and need a script, reach out to me @ maxwell.t.elling@nasa.gov

n_polar_anom = n_polar.groupby('time.month') - n_polar_base.groupby('time.month').mean(dim=('time'))
n_extratropics_anom = n_extratropics.groupby('time.month') - n_extratropics_base.groupby('time.month').mean(dim=('time'))
n_subtropics_anom = n_subtropics.groupby('time.month') - n_subtropics_base.groupby('time.month').mean(dim=('time'))
n_tropics_anom = n_tropics.groupby('time.month') - n_tropics_base.groupby('time.month').mean(dim=('time'))
s_tropics_anom = s_tropics.groupby('time.month') - s_tropics_base.groupby('time.month').mean(dim=('time'))
s_subtropics_anom = s_subtropics.groupby('time.month') - s_subtropics_base.groupby('time.month').mean(dim=('time'))
s_extratropics_anom = s_extratropics.groupby('time.month') - s_extratropics_base.groupby('time.month').mean(dim=('time'))
s_polar_anom = s_polar.groupby('time.month') - s_polar_base.groupby('time.month').mean(dim=('time'))


# In[183]:


# Sum the monthly zonal anoms to get the global mean anom. Note: results are in °C

''' Weighting was calculated via dividing weighting area by total global area, 
resulting in temp values that are weighted unproportionately to the zonal area. 
The quick fix is to divide by zonal area.

Zonal and hemispheric means must be divided by the repsective total zonal weighting 
because the zonal anomalies above were weighted using the total area over the whole globe.

When you divide by the total zonal weighting, you are essentially weighting the grid cells 
of each zone against the total area of that zone, which will produce correct values. 

If you do not divide by the respective zonal weight, the sum of grid cell weights in each zone 
will not add up to 100% (rather the weights for all the zones will add to 100%). This will 
cause the temp values to be much smaller (and incorrect).
'''

global_mean_anomaly = n_polar_anom+n_extratropics_anom+n_subtropics_anom+n_tropics_anom+s_tropics_anom+s_subtropics_anom+s_extratropics_anom+s_polar_anom

NHem = (n_polar_anom+n_extratropics_anom+n_subtropics_anom+n_tropics_anom)/(NHem)
SHem = (s_tropics_anom+s_subtropics_anom+s_extratropics_anom+s_polar_anom)/(SHem)
N24N90 = (n_polar_anom+n_extratropics_anom+n_subtropics_anom)/(N24N90)
S24N24 = (n_tropics_anom+s_tropics_anom)/(S24N24)
S24S90 = (s_subtropics_anom+s_extratropics_anom+s_polar_anom)/(S24S90)
N64N90 = (n_polar_anom)/(N64N90)
N44N64 = (n_extratropics_anom)/(N44N64)
N24N44 = (n_subtropics_anom)/(N24N44)
EQUN24 = (n_tropics_anom)/(EQUN24)
EQUS24 = (s_tropics_anom)/(EQUS24)
S24S44 = (s_subtropics_anom)/(S24S44)
S44S64 = (s_extratropics_anom)/(S44S64)
S64S90 = (s_polar_anom)/(S64S90)

