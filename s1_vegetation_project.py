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
#from small_plot_functions import matrix_show, col_to_ma, r2_arrays_to_googleEarth
# from synth_ts import *
# from small_plot_functions import low_resolution_ifgs
# from small_plot_functions import *
# from ifg_plotting_funcs import * 
#from insar_tools_copy import files_to_daisy_chain, get_daisy_chain, mask_nans, r3_to_r2, baseline_from_names
from S1VP_auxilliary_functions import r2_arrays_to_googleEarth, get_daisy_chain, mask_nans, r3_to_r2, baseline_from_names


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

#%% Things to set

ICASAR_settings = {"n_comp" : 6,                                    # number of components to recover with ICA (ie the number of PCA sources to keep)
                    "bootstrapping_param" : (200, 0),
                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param" : (30, 12),                       # (perplexity, early_exaggeration)
                    "ica_param" : (1e-2, 150),                     # (tolerance, max iterations)
                    "figures" : 'png+window',
                    "ge_kmz"    :  True}                            # make a google earth .kmz of the ICs


#%% Open the xarray files
#data = xr.open_dataset('unw.nc')

phUnw_r3, _, ifg_names, lons, lats = xarray_to_numpy('unw.nc')
coh_r3, _, _, _, _ = xarray_to_numpy('coh.nc')

print('Revesring the order of the latitudes.  ')
lats = lats[::-1]

#%% Look at coherence

mean_coh = np.mean(coh_r3, axis = 0)

r2_arrays_to_googleEarth(mean_coh[np.newaxis,], lons, lats, layer_name_prefix = 'layer', kmz_filename = 'mean_coherence')

#%% get the daisy chains   | Possibly this can be updated to solve for daisy chains, rather than just discarding the other ones? 

#_, daisy_chain_ifgs, _ = files_to_daisy_chain(ifg_names, figures = True)          # find the dates of the daisy chain ifgs
phUnw_r3, ifg_names_dc = get_daisy_chain(phUnw_r3, ifg_names)                                           # get the daisy chain ifgs.  
phUnw_r3_ma = mask_nans(phUnw_r3)                                                                       # mask any nans
phUnw_r2 = r3_to_r2(phUnw_r3_ma)                                                                        # conver to rank 2

baselines = baseline_from_names(ifg_names_dc)
baselines_cs = np.cumsum(baselines)

#%% Save the ifgs as pngs.  

# from small_plot_functions import matrix_show

# for ifg_n, ifg_ma in enumerate(phUnw_r3_ma):
#     matrix_show(ifg_ma, ifg_names_dc[ifg_n], save_path = "./ifgs/")


#%% ICASAR

sources, tcs, residual, Iq, n_clusters, S_all_info = ICASAR(phUnw_r2['ifgs'], mask = phUnw_r2['mask'],
                                                            lons = lons, lats = lats, **ICASAR_settings)



#%%

def baselines_cs_with_gaps(ifg_names):
    """
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


baselines_cs2 = baselines_cs_with_gaps(ifg_names_dc)

#%%


def tcs_plot(tcs, baselines_cs):
    """
    """
    
    n_tcs = tcs.shape[1]                                                             # each new tc is a colum
    
    f, axes = plt.subplots(n_tcs, 1)
    f.canvas.set_window_title('Time courses plot')
    
    for n_tc in range(n_tcs):
        axes[n_tc].scatter(baselines_cs, tcs[:,n_tc])
        axes[n_tc].plot(baselines_cs, tcs[:,n_tc])
        axes[n_tc].axhline(0, c = 'k')
        
tcs_plot(np.cumsum(tcs, axis = 0), baselines_cs2)
    
    
    
    
    
    

    
    
    
    
    
#%%
    
    
    

f, ax = plt.subplots()