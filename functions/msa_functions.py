# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import h5py
import os
###############################################################################
def msa2spec_sum_para(file_path, signal_progress = None, signal_sum_spec = None):
    ''' 
    This function reads out the spectrum of a .msa-file and reads out the
    detector parameters given in the .msa-file. It returns an dict containing
    the names of the single-spectrum files, the sum_spec and the measurement
    parameters.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .MSA-file.
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1]
    int number of spectra
    '''
    file_name = os.path.split(file_path)[1]
    if os.path.exists(f'{file_path}/data/data.h5'):
        with h5py.File(f'{file_path}/data/data.h5', 'r+') as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    else:
        empty_file = h5py.File(f'{file_path}/data/data.h5', 'w')
        empty_file.close()
    worth_fit = []
    counts = []
    folder_path = os.path.dirname(file_path)
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    if signal_sum_spec != None:
        signal_sum_spec.emit('None')
    with open(file_path,'rb') as infile:
        content = infile.read().decode('utf-8','ignore').replace('\r','').split('\n')
    for counter, line in enumerate(content):
        if signal_progress != None:
            signal_progress.emit(counter)
        ### read out the channels
        if '#NPOINTS' in line:
            channels = int(line.split()[2])
        ### read out of the parameters
        elif '#OFFSET' in line:
            a0 = float(line.split()[2])
        elif '#XPERCHAN' in line:
            a1 = float(line.replace('    : ','').replace('#XPERCHAN','').replace('\r\n',''))
        ### read out life_time
        elif '#LIVETIME' in line:
            life_time = float(line.replace('#LIVETIME  -s: ',''))
        elif '#REALTIME' in line:
            real_time = float(line.replace('#REALTIME  -s: ',''))
        ### read out start and end of the spectra
        elif '#SPECTRUM' in line:
            start = counter
        elif '#ENDOFDATA' in line:
            end = counter     
        elif '#NXPOS' in line:
            x_pos = int(line.split()[2])
        elif '#NYPOS' in line:
            y_pos = int(line.split()[2])
        elif '#NZPOS' in line:
            z_pos = int(line.split()[2])
        elif '#XSTEP' in line:
            x_step = int(line.split()[2])*1e-3
        elif '#YSTEP' in line:
            y_step = int(line.split()[2])*1e-3
        elif '#ZSTEP' in line:
            z_step = int(line.split()[2])*1e-3
        elif '#XNULL' in line:
            x0 = int(line.split()[2])*1e-3
        elif '#YNULL' in line:
            y0 = int(line.split()[2])*1e-3
        elif '#ZNULL' in line:
            z0 = int(line.split()[2])*1e-3
    gating_time = 600e-9                                                          ### has to be determined
    position_dimension = np.array([x_pos, y_pos, z_pos])
    steps = np.array([x_step, y_step, z_step])
    steps[steps==0] = 1
    origin = [x0, y0, z0]
    positions = build_positions(position_dimension, origin, steps)
    tensor_positions = build_tensor_position(position_dimension)
    max_energy =  a0 + a1*channels
    if signal_sum_spec != None:
        signal_sum_spec.emit('progress spectra')
    spectra_tmp = content[start+1:end] ### reads out the lines containing the spectra
    spectra_tmp = np.asarray([np.array(line.split()).astype(np.float64) for line in spectra_tmp])
    spectra_tmp = np.ndarray.flatten(spectra_tmp) # writes the intensities in one line

    factor = 0.25
    zero_peak_position = 25
    zero_peak_frequency = 100   
    #spectra_tmp = np.divide(spectra_tmp, life_time)
    spectra_tmp = np.reshape(spectra_tmp,(int(np.ceil(len(spectra_tmp)/channels)),channels)) # reshapes intensity to channels*spectra

    for i in range(len(spectra_tmp)):
        life_time = np.sum(spectra_tmp[i][zero_peak_position-int(20*factor):zero_peak_position+int(20*factor)]) / zero_peak_frequency
        if life_time == 0:
            print('error in %s \nlife_time is ZERO\nsetting life_time to 1'%file_path)
            life_time = 1
        print('life_time',life_time)
        spectra_tmp[i] = np.divide(spectra_tmp[i], life_time)

    if signal_sum_spec != None:
        signal_sum_spec.emit('spectra done')
    spectra = {}

    for i in range(len(spectra_tmp)):
        spectra['%d'%i] = spectra_tmp[i]
        counts.append(sum(spectra_tmp[i]))
        if sum(spectra_tmp[i]) > 1000:
            worth_fit.append(True)
        else:
            worth_fit.append(False)
    max_pixel_spec = np.max(np.array(list(spectra.values())), axis = 0)
    sum_spec = np.sum(list(spectra.values()), axis = 0)
    parameters = [a0,a1,0.110, 0.1, life_time, max_energy, gating_time, real_time]
    
    with h5py.File('%s/data/data.h5'%folder_path, 'r+') as tofile:
        tofile.create_dataset(f'{file_name}/spectra', data = np.array(list(spectra.values())),
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/max pixel spec', data = max_pixel_spec,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/sum spec', data = sum_spec,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/counts', data = counts,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/position dimension', data = position_dimension,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/positions', data = positions,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/tensor positions', data = tensor_positions,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/parameters', data = parameters,
                              compression="gzip")
    
    return spectra, sum_spec, parameters


def msa2positions(file_path):
    ''' 
    This function reads out the positions of a .msa-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.
    
    Returns
    -------
    [x_pos, y_pos, z_pos]
    '''
    with open(file_path,'rb') as infile:
        content = infile.read().decode('utf-8','ignore').replace('\r','').split('\n')
    for line in content:
        ### read out of the x,y,z positions to calculate the number of spectra
        if '#NXPOS' in line:
            x_pos = int(line.split()[2])
        if '#NYPOS' in line:
            y_pos = int(line.split()[2])
        if '#NZPOS' in line:
            z_pos = int(line.split()[2])
            break
        ### read out start and end of the spectra   

    return [x_pos, y_pos, z_pos]


def msa2life_time(file_path):
    ''' 
    This function reads out the Life-Time of a .msa-file .
    '''  
    with open(file_path,'rb') as infile:
        content = infile.read().decode('utf-8','ignore').replace('\r','').split('\n')
    for line in content:
        if '#LIVETIME' in line:
            life_time = float(line.replace('#LIVETIME  -s: ',''))
            break
    return life_time


def msa2real_time(file_path):
    ''' 
    This function reads out the Life-Time of a .msa-file .
    '''  
    with open(file_path,'rb') as infile:
        content = infile.read().decode('utf-8','ignore').replace('\r','').split('\n')
    for line in content:
        if '#REALTIME' in line:
            real_time = float(line.replace('#REALTIME  -s: ',''))
            break
    return real_time


def msa2channels(file_path):
    ''' 
    This function reads out the number of channels of a .msa-file .
    '''  
    with open(file_path,'rb') as infile:
        content = infile.read().decode('utf-8','ignore').replace('\r','').split('\n')
    for line in content:
        if '#NPOINTS' in line:
            channels = int(line.replace('#NPOINTS     : ',''))
            break
    return channels


def build_positions(positions, origin, steps):
    '''
    This function returns the measurement position array for x,y,z

    Parameters
    ----------
    positions : list, array
        list or array containing the number of x,y,z positions
    origin : list, array
        list or array containing the x0,y0,z0 position
    steps : list, array
        list or array containing the x,y,z step width 

    Returns
    -------
    array
        array containing the measurement positions xi,yi,zi

    '''
    'returns an array of Form [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]'
    
    ends, pos = [], []
    for i in range(3):
        ends.append(np.around(origin[i]+steps[i]*positions[i],3))
        pos.append(np.arange(origin[i], origin[i]+steps[i]*positions[i], steps[i]))
        if np.around(pos[i][-1], 3) == ends[i]:
            pos[i] = np.delete(pos[i], len(pos[i])-1)
    positions = np.array(np.meshgrid(pos[0], pos[1], pos[2])).T.reshape(-1,3)
    
    return sorted(positions, key = lambda x:x[1])


def build_tensor_position(positions):
    '''
    This function returns the measurement tensor position array for x,y,z

    Parameters
    ----------
    positions : list, array
        list or array containing the number of x,y,z positions


    Returns
    -------
    array
        array containing the measurement positions xi,yi,zi

    '''
    'returns an array of Form [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]'
    
    pos_arr_x = np.arange(positions[0])
    pos_arr_y = np.arange(positions[1])
    pos_arr_z = np.arange(positions[2])
    tensor_positions = np.array(np.meshgrid(pos_arr_x,pos_arr_y,pos_arr_z)).T.reshape(-1,3)
    return sorted(tensor_positions, key = lambda x:x[1])
