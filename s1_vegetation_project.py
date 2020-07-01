#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:33:35 2020

@author: matthew
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys

sys.path.append("/home/matthew/university_work/01_blind_signal_separation_python/13_ICASAR/ICASAR_GitHub")                  # location of ICASAR functions
from ICASAR_functions import ICASAR

sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")
#from small_plot_functions import matrix_show #, col_to_ma, r2_arrays_to_googleEarth
# from synth_ts import *
# from small_plot_functions import low_resolution_ifgs
# from small_plot_functions import *
# from ifg_plotting_funcs import * 
from insar_tools import files_to_daisy_chain, get_daisy_chain, mask_nans, r3_to_r2, baseline_from_names
from S1VP_auxilliary_functions import r2_arrays_to_googleEarth 
#from S1VP_auxilliary_functions import r2_arrays_to_googleEarth, get_daisy_chain, mask_nans, r3_to_r2, baseline_from_names


def xarray_to_numpy(xarray_file):
    """ Open one of Milan's xarray files and extract ifgs (or coherence)
    """
    
    import numpy as np
    import xarray as xr
    
    dUNW = xr.open_dataset(xarray_file)

    # 0 get the data
    n_ifgs = len(dUNW.data_vars)
    nx = dUNW.dims['lon']
    ny = dUNW.dims['lat']
    
    phUnw_r3 = np.zeros((n_ifgs, ny, nx))                                              # initaite rank 3
    phUnw_r2 = np.zeros((n_ifgs, (ny*nx)))                                              # initaite and simpler rank2, with ifgs as rows
    ifg_names = []
        
    for ifg_n, data_variable in enumerate(dUNW.data_vars):
        phUnw_r3[ifg_n,] = dUNW[data_variable]
        phUnw_r2[ifg_n,] = np.ravel(dUNW[data_variable])
        ifg_names.append(data_variable)
        
    # 1: get the lats and lons
    lons = np.asarray(dUNW.coords['lon'])
    lats = np.asarray(dUNW.coords['lat'])
        
    return phUnw_r3, phUnw_r2, ifg_names, lons, lats

def tcs_plot(tcs, baselines_cs):
    """Pot the time courses to be easier to see (each gets its own row)
    """
    
    n_tcs = tcs.shape[1]                                                             # each new tc is a colum
    
    f, axes = plt.subplots(n_tcs, 1)
    f.canvas.set_window_title('Time courses plot')
    
    for n_tc in range(n_tcs):
        axes[n_tc].scatter(baselines_cs, tcs[:,n_tc])
        axes[n_tc].plot(baselines_cs, tcs[:,n_tc])
        axes[n_tc].axhline(0, c = 'k')


def baselines_cs_with_gaps(ifg_names):
    """Quick and dirty function to get the baslines for a time sereis with gaps
    """
    import numpy as np
    from datetime import datetime
    
    baselines_cs = []
    master_ifg1 = datetime.strptime(ifg_names[0][:8], '%Y%m%d')
    for ifg_name in ifg_names:
        slave_ifgn = datetime.strptime(ifg_name[9:], '%Y%m%d')
        baselines_cs.append((slave_ifgn - master_ifg1).days)
        
    baselines_cs = np.asarray(baselines_cs)
    return baselines_cs


def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.  
    Useful when converting between vectors used by BSS methods results that are to be plotted
    Inputs:
        col | rank 1 array | 
        pixel_mask | array mask (rank 2)
    Outputs:
        source | rank 2 masked array | colun as a masked 2d array
    2017/10/04 | collected from various functions and placed here.  
    """
    import numpy.ma as ma 
    import numpy as np
    
    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()   
    return source

#%% Things to set

ref_region = (10,10)                                                # set a reference region, in x then y, from top left (ie like a matrix)
point_interest = (42, 30)                                           # x then y, from top left?

ICASAR_settings = {"n_comp" : 6,                                    # number of components to recover with ICA (ie the number of PCA sources to keep)
                    "bootstrapping_param" : (200, 0),
                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param" : (30, 12),                       # (perplexity, early_exaggeration)
                    "ica_param" : (1e-2, 150),                     # (tolerance, max iterations)
                    "figures" : 'png+window',
                    "ge_kmz"    :  True}                            # make a google earth .kmz of the ICs


#%% Open the xarray files for unwrapped phase and coherence.  
#data = xr.open_dataset('unw.nc')

phUnw_r3, _, ifg_names, lons, lats = xarray_to_numpy('unw.nc')
coh_r3, _, _, _, _ = xarray_to_numpy('coh.nc')

print('Revesring the order of the latitudes (bit of a fudge) ')
lats = lats[::-1]

mean_coh = np.mean(coh_r3, axis = 0)
r2_arrays_to_googleEarth(mean_coh[np.newaxis,], lons, lats, layer_name_prefix = 'layer', kmz_filename = 'mean_coherence')

#%% Set a reference region

_, ny, nx = phUnw_r3.shape                                                                  # get size of ifg, in pixels
ref_pixel_value = phUnw_r3[:,ref_region[1], ref_region[0]][:, np.newaxis, np.newaxis]
ref_pixel_r3 = np.repeat(np.repeat(ref_pixel_value, ny, 1), nx, 2)                          # duplicate to be the same size as all the ifgs.  
phUnw_r3 -= ref_pixel_r3                                                                    # remove from all ifgs so ref pixel is always 0


#%% Do ICASAR with all ifgs

# phUnw_r3_ma, n_nan = mask_nans(phUnw_r3, figures = True, threshold = 2)                          # mask any nans, and remove ifgs with more than threshold % of pixels as nans
# phUnw_r2 = r3_to_r2(phUnw_r3_ma)                                                                        # convert to rank 2 (ifgs as row vectors), ready for ICASAR

# sources, tcs, residual, Iq, n_clusters, S_all_info, phUnw_mean = ICASAR(phUnw_r2['ifgs'], mask = phUnw_r2['mask'],
#                                                                         lons = lons, lats = lats, **ICASAR_settings,
#                                                                         out_folder = './ICASAR_outputs_all_ifgs/')




#%% Do ICASAR with only the short baseline ifgs

phUnw_r3, ifg_names = get_daisy_chain(phUnw_r3, ifg_names)                                        # get the daisy chain ifgs.  

# Milan - 2020/06/30 addition  PhUnwr3 is 83x110x120, as expected, but there are some pixels that are nan occasioanly in the off interferogram, and have to be removed from the entire time series.  

phUnw_r3_ma, n_nan = mask_nans(phUnw_r3 , threshold = 2)                                          # mask any nans (and drop any ifgs that more than 2% of them are nans)

# now PhUnw_r3_ma is a masked array version of phUnw_r3 - same size (83x110x120), but some pixels in that are masked

phUnw_r2 = r3_to_r2(phUnw_r3_ma )                                                                  # convert to rank 2 (ie row vectors, which are needed for ICA)

# if we want to see the pixels I had to drop (ie the mask):
f, ax =plt.subplots()
ax.imshow(phUnw_r2['mask'])
ax.set_title('nans mask')

# and having a look at the size of things
n_pixels_masked = np.sum(phUnw_r2['mask'])                  # should be 60
# 60 pixels are masked.  (110x120) = 13200
#       |->                 -60    = 13140
# phUnw_r2['ifgs'] shape is     83 x 13140   (so the samenumber of pixels)

# to get back to a rank 2 masked array, use a little fuctnion called col_to_ma.  
test_ifg = col_to_ma(phUnw_r2['ifgs'][0,], phUnw_r2['mask'])
# and continuing as before.  

baselines_cs  = baselines_cs_with_gaps(ifg_names)                                                  # get the cumulative baselines (ie 6 12 18 24 etc if all 6 day)

sources, _,  tcs, residual, Iq, n_clusters, S_all_info, phUnw_mean  = ICASAR(phUnw_r2 ['ifgs'], mask = phUnw_r2 ['mask'],
                                                                                             lons = lons, lats = lats, **ICASAR_settings,
                                                                                             out_folder = './ICASAR_outputs/')

tcs_plot(np.cumsum(tcs , axis = 0), baselines_cs )                                                  # plot the cumulative time coursses

#%% Point of interest

poi_ts = phUnw_r3_ma [:, point_interest[1], point_interest[0]]
poi_ts_cs = np.cumsum(poi_ts)

f, ax = plt.subplots(1,1)
ax.scatter(np.arange(poi_ts_cs.shape[0]), poi_ts_cs)
ax.axhline(0, c = 'k')
plt.suptitle(f'Time series for one point (x:{point_interest[0]} y:{point_interest[1]}) ')
f.canvas.set_window_title('Time series for one point')

#%% Save the ifgs as pngs.  

# from small_plot_functions import matrix_show

# for ifg_n, ifg_ma in enumerate(phUnw_r3_ma):
#     matrix_show(ifg_ma, ifg_names[ifg_n], save_path = "./ifgs/")


