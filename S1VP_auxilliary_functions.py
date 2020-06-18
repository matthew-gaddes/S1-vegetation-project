# -*- coding: utf-8 -*-
"""
 A selection of functions for helping to work with LiCSAR

Common paths:
path = '/nfs/a1/homes/eemeg/galapagos_from_asf_v2/'                 # path to the SLC files
path = '/nfs/a1/homes/eemeg/data_etna_from_asf/'                 # path to the SLC files

"""

#%%



def r2_to_r3(ifgs_r2, mask):
    """ Given a rank2 of ifgs as row vectors, convert it to a rank3. 
    Inputs:
        ifgs_r2 | rank 2 array | ifgs as row vectors 
        mask | rank 2 array | to convert a row vector ifg into a rank 2 masked array        
    returns:
        phUnw | rank 3 array | n_ifgs x height x width
    History:
        2020/06/10 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
    from small_plot_functions import col_to_ma
    
    n_ifgs = ifgs_r2.shape[0]
    ny, nx = col_to_ma(ifgs_r2[0,], mask).shape                                   # determine the size of an ifg when it is converter from being a row vector
    
    ifgs_r3 = np.zeros((n_ifgs, ny, nx))                                                # initate to store new ifgs
    for ifg_n, ifg_row in enumerate(ifgs_r2):                                           # loop through all ifgs
        ifgs_r3[ifg_n,] = col_to_ma(ifg_row, mask)                                  
    
    mask_r3 = np.repeat(mask[np.newaxis,], n_ifgs, axis = 0)                            # expand the mask from r2 to r3
    ifgs_r3_ma = ma.array(ifgs_r3, mask = mask_r3)                                      # and make a masked array    
    return ifgs_r3_ma



def r3_to_r2(phUnw):
    """ Given a rank3 of ifgs, convert it to rank2 and a mask.  Works with either masked arrays or just arrays.  
    Inputs:
        phUnw | rank 3 array | n_ifgs x height x width
    returns:
        r2_data['ifgs'] | rank 2 array | ifgs as row vectors
        r2_data['mask'] | rank 2 array 
    History:
        2020/06/09 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
    
    if ma.isMaskedArray(phUnw):
        n_pixels = len(ma.compressed(phUnw[0,]))                                            # if it's a masked array, get the number of non-masked pixels
        mask = ma.getmask(phUnw)[0,]                                                        # get the mask, which is assumed to be constant through time
    else:
        n_pixels = len(np.ravel(phUnw[0,]))                                                 # or if a normal numpy array, just get the number of pixels
        mask = np.zeros(phUnw[0,].shape)                                                    # or make a blank mask
 
    r2_ifgs = np.zeros((phUnw.shape[0], n_pixels))                                          # initiate to store ifgs as rows in
    for ifg_n, ifg in enumerate(phUnw):
        if ma.isMaskedArray(phUnw):
            r2_ifgs[ifg_n,] = ma.compressed(ifg)                                            # non masked pixels into row vectors
        else:
            r2_ifgs[ifg_n,] = np.ravel(ifg)                                                 # or all just pixles into row vectors

    r2_data = {'ifgs' : r2_ifgs,                                                            # make into a dictionary.  
               'mask' : mask}          
    return r2_data



#%%






#%%

def invert_to_DC(ifg_acq_numbers, tcs):
    """
    Given a list of which acquisitions the time series of interferograms were made,
    returns the time courses for the simplest daisy chain of interferograms.
    From the method described in Lundgren ea 2001
    """
    import numpy as np
    n_ifgs = len(ifg_acq_numbers)
    n_times = np.max(ifg_acq_numbers)                 # this will be the number of model parameters
    d = tcs
    g = np.zeros((n_ifgs, n_times))                 # initiate as zeros

    for i, ifg_acq_number in enumerate(ifg_acq_numbers):
        g[i, ifg_acq_number[0]:ifg_acq_number[1]] = 1
    m = np.linalg.inv(g.T @ g) @ g.T @ d                       # m (n_sources x 1)

    return m




#%%

def files_to_daisy_chain(ifg_names, figures = True):
    """
    Given a list of interfergram names (masterDate_slaveDate), it:
        - finds all the acquisition dates
        - forms a list of names of the simplest daisy chain of interfegrams we can make
        - lists which number acquistion each interferogram is between (e.g (0, 3))
    Inputs:
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January)
        figures | boolean | For the figure output of the termporal baselines.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    # get acquistion dates (ie when each SAR image was taken)
    dates_acq = []
    for date in ifg_names:
        date1 = date[:8]
        date2 = date[9::]
        if date1 not in dates_acq:
            dates_acq.append(date1)
        if date2 not in dates_acq:
            dates_acq.append(date2)
    dates_acq = sorted(dates_acq)

    # get the dates for the daisy chain of interferograms (ie the simplest span to link them all)
    daisy_chain_ifgs= []
    for i in range(len(dates_acq)-1):
        daisy_chain_ifgs.append(dates_acq[i] + '_' + dates_acq[i+1])

    # get acquestion dates in terms of days since first one
    days_elapsed = np.zeros((len(dates_acq), 1))
    first_acq = datetime.strptime(ifg_names[0][:8], '%Y%m%d')
    for i, date in enumerate(dates_acq):
        date_time = datetime.strptime(date, '%Y%m%d')
        days_elapsed[i,0] = (date_time - first_acq).days

    # find which acquisiton number each ifg spans
    ifg_acq_numbers = []
    for i, ifg in enumerate(ifg_names):
        master = ifg[:8]
        slave = ifg[9:]
        pair = (dates_acq.index(master), dates_acq.index(slave))
        ifg_acq_numbers.append(pair)



    if figures:                                                             # temp baseline plot
        f, ax = plt.subplots(1)
        for i, file in enumerate(ifg_names):
            master = datetime.strptime(file[:8], '%Y%m%d')
            slave = datetime.strptime(file[9:], '%Y%m%d')
            master_xval = (master - first_acq).days
            slave_xval = (slave - first_acq).days
            ax.plot((master_xval, slave_xval), (i,i), '-')
        for i in range(len(dates_acq)-1):
            master = datetime.strptime(dates_acq[i], '%Y%m%d')
            slave = datetime.strptime(dates_acq[i+1], '%Y%m%d')
            master_xval = (master - first_acq).days
            slave_xval = (slave - first_acq).days
            ax.plot((master_xval, slave_xval), (-len(dates_acq)+i,-len(dates_acq)+i), '-', c = 'k')
        ax.set_ylabel('Ifg. #')
        ax.set_xlabel('Days since first acquisition')

    return dates_acq, daisy_chain_ifgs, ifg_acq_numbers

#%% delete all but daisy chain ifgs

def get_daisy_chain(phUnw, ifg_names):
    """Given a an array of LicSAR ifgs and their names (which are slaveData_masterDate),
    return only the daisy chain of ifgs.
    Inputs:
        phUnw | r2 or 3 array | ifgs as (samples x height x width) or as (samples x n_pixels)
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January)
        
    Returns:
        phUnw | r2 or 3 array | ifgs as (samples x height x width) or as (samples x n_pixels), but only daisy chain
        ifg_names | list of strings | of form 20180101_20190102 (for a 1 day ifg in January), but only daisy chain
    
    History:
        2019/??/?? | MEG | Written
        2020/06/02 | MEG | Update to handle phUnw as either rank 2 or rank 3
        
    """
    from insar_tools import files_to_daisy_chain

    _, daisy_chain_ifg_names, _ = files_to_daisy_chain(ifg_names, figures = False)                   # get the names of the daisy chain ifgs

    daisy_chain_ifg_args = []                                                                       # list to store which number ifg each daisy chain ifg is
    for daisy_chain_ifg_name in daisy_chain_ifg_names:                                              # loop through populating hte list
        try:
            daisy_chain_ifg_args.append(ifg_names.index(daisy_chain_ifg_name))
        except:
            print(f'{daisy_chain_ifg_name} could not be found in the interferogram list.  Skipping it.  ')
            pass
    phUnw = phUnw[daisy_chain_ifg_args,]                                                         # select the desired ifgs
    ifg_names = [ifg_names[i] for i in daisy_chain_ifg_args]                                        # select the ifg names

    return phUnw, ifg_names


#%%

def baseline_from_names(names_list):
    """Given a list of ifg names in the form YYYYMMDD_YYYYMMDD, find the temporal baselines in days_elapsed
    Inputs:
        names_list | list | in form YYYYMMDD_YYYYMMDD
    Returns:
        baselines | list of ints | baselines in days
    History:
        2020/02/16 | MEG | Documented
    """

    from datetime import datetime, timedelta

    baselines = []
    for file in names_list:

        master = datetime.strptime(file.split('_')[-2], '%Y%m%d')
        slave = datetime.strptime(file.split('_')[-1][:8], '%Y%m%d')
        baselines.append(-1 *(master - slave).days)
    return baselines


#%%


def r2_arrays_to_googleEarth(images_r3_ma, lons, lats, layer_name_prefix = 'layer', kmz_filename = 'ICs'):
    """ Given one or several arrays in a rank3 array, create a multilayer Google Earth file (.kmz) of them.  
    Inputs:
        images_r3_ma | rank3 masked array |x n_images x ny x nx
        lons | rank 1 array | lons of each pixel in the image.  
        lats | rank 1 array | lats of each pixel in theimage
        layer_name_prefix | string | Can be used to set the name of the layes in the kmz (nb of the form layer_name_prefix_001 etc. )
        kmz_filename | string | Sets the name of the kmz produced
    Returns:
        kmz file
    History:
        2020/06/10 | MEG | Written
    """
    
    import numpy as np
    import numpy.ma as ma
    import os
    import shutil
    import simplekml
    from small_plot_functions import r2_array_to_png

    n_images = images_r3_ma.shape[0]    
    # 0 temporary folder for intermediate pngs
    try:
        os.mkdir('./temp_kml')                                                                       # make a temporay folder to save pngs
    except:
        print("Can't create a folder for temporary kmls.  Trying to delete 'temp_kml' incase it exisits already... ", end = "")
        try:
            shutil.rmtree('./temp_kml')                                                              # try to remove folder
            os.mkdir('./temp_kml')                                                                       # make a temporay folder to save pngs
            print("Done. ")
        except:
          raise Exception("Problem making a temporary directory to store intermediate pngs" )

    # 1: Initiate the kml
    kml = simplekml.Kml()
        
    # 2 Begin to loop through each iamge
    for n_image in np.arange(n_images)[::-1]:                                           # Reverse so that first IC is processed last and appears as visible
        layer_name = f"{layer_name_prefix}_{str(n_image).zfill(3)}"                     # get the name of a layer a sttring
        r2_array_to_png(images_r3_ma[n_image,], layer_name, './temp_kml/')              # save as an intermediate .png
        
        ground = kml.newgroundoverlay(name= layer_name)                                 # add the overlay to the kml file
        ground.icon.href = f"./temp_kml/{layer_name}.png"                               # and the actual image part
    
        ground.gxlatlonquad.coords = [(lons[0], lats[0]), (lons[-1],lats[0]),           # lon, lat of image south west, south east
                                      (lons[-1], lats[-1]), (lons[0],lats[-1])]         # north east, north west  - order is anticlockwise around the square, startign in the lower left
       
    #3: Tidy up at the end
    kml.savekmz(f"{kmz_filename}.kmz", format=False)                                    # Saving as KMZ
    shutil.rmtree('./temp_kml')    

