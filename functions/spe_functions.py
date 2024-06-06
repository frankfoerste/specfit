# -*- coding: utf-8 -*-
"""
This programm will extract the channels of the given .spe-file and save them
to a .txt-file with the same name.
"""
###############################################################################
import numpy as np
import os
import time as t
from glob import iglob
from glob import glob
import pickle
import psutil ### Module to determine system memory properties
import matplotlib.pyplot as plt
plt.ion()

###############################################################################
        
        
def spe2spec_para(file_path, print_warning = False):
    ''' 
    This function reads out the spectrum of a .spe-file and reads out the
    detector parameters given in the .spe-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .spe-file.
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    ### define default values
    a0 = -0.023312115768056554
    a1 = 0.02020159747941543
    FWHM = 0.09805524231499399
    Fano = 0.1277816784400016
    channels = False
    gating_time = 200e-6
    
    data_start = False
    life_time = False
    spectrum = []
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if not isinstance(channels, bool):
               spectrum.append(int(line))
            elif data_start == True:
                data_start = False
                channels = int(line.split()[1])
            elif life_time == True:
                life_time = int(line)/1000
                real_time = int(line)/1000
            else:
                if '$DATA' in line:
                    data_start = True
                elif '$MEAS_TIM' in line:
                    life_time = True
    parameters = [a0, a1, Fano, FWHM, life_time, a0 + a1 * (channels), gating_time, real_time]
    return np.asarray(spectrum), parameters
      

def many_spe2spec_para(folder_path, signal = None ,worth_fit_threshold = 200,
                       save_sum_spec = True, save_spectra = True, 
                       save_counts = True, save_parameters = True,
                       save_any = True, print_warning = False,
                       save_spec_as_dict = True,
                       return_values = False):
    ''' 
    This function reads out the spectrum of a .spe-file and reads out the
    detector parameters given in the .spe-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .spe-file.
    index : int
        the number of the spectrum in the measurement set
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    worth_fit = []
    counts = []
    parameters = []
    # spectra = {}
    a0 = -0.023312115768056554
    a1 = 0.02020159747941543
    FWHM = 0.09805524231499399
    Fano = 0.1277816784400016
    gating_time = 200e-6
    signal_progress = signal
    folder_size = 0
    machine_memory = psutil.virtual_memory().total * 1E-9
    start = t.time()
    sorted_folder = np.sort(glob('%s/*.spe'%folder_path))
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    if save_spec_as_dict == False:
        positions = spe_tensor_positions(folder_path)
        ### read out the x,y,z axes
        x = np.unique(positions[:,0])
        y = np.unique(positions[:,1])
        z = np.unique(positions[:,2])
        ### calculate the stepsizes in every direction
        try: x_steps = x[1] - x[0]
        except: x_steps = 1
        try: y_steps = y[1] - y[0]
        except: y_steps = 1
        try: z_steps = z[1] - z[0]
        except: z_steps = 1
        ### calculate the tensor positions by dividing positions by steps
        tensor_positions = np.copy(positions)
        ### now subtract to 0
        tensor_positions[:,0] -= x[0]
        tensor_positions[:,1] -= y[0]
        tensor_positions[:,2] -= z[0]
        tensor_positions /= [x_steps, y_steps, z_steps]
        tensor_positions = np.array(tensor_positions, dtype = int)
        
    folder_size = sum(os.path.getsize(f) for f in sorted_folder if os.path.isfile(f)) * 1E-9 
    life_time = False
    if (folder_size / machine_memory) < 0.7:                              ### for machines with big memory
        print('machine memory big enough. creating spectra dict')
        for file_nr, spe_file in enumerate(sorted_folder):
            data_start = False
            life_time = False
            channels = False
            spectrum = []
            with open(spe_file,'r', encoding = 'ISO-8859-1') as infile:
                for line in infile:
                    if not isinstance(channels, bool):
                       spectrum.append(int(line))
                    elif data_start == True:
                        # data_start = False
                        channels = int(line.split()[1])
                    elif life_time == True:
                        life_time = int(line)/1000
                        real_time = int(line)/1000
                    else:
                        if '$DATA' in line:
                            data_start = True
                        elif '$MEAS_TIM' in line:
                            life_time = True

            spectrum = np.asarray(spectrum)
            
            parameters.append([a0, a1, Fano, FWHM, life_time, a0 + a1 * (channels), gating_time, real_time])
            spectrum = np.divide(spectrum, life_time)
            
            sum_spec_tmp = sum(spectrum)
            counts.append(sum_spec_tmp)
            if sum_spec_tmp > worth_fit_threshold:
                worth_fit.append(True)
            else:
                worth_fit.append(False)
            ### now we try to rebuild specfit load in routine to support array
            ### spectra
            if save_spec_as_dict:
                if file_nr == 0:
                    spectra = {}
                spectra['%i'%file_nr] = spectrum
            else:
                if file_nr == 0:
                    spectra = np.empty((len(x), len(y), len(z), 4096))
                x_pos, y_pos, z_pos = tensor_positions[file_nr]
                spectra[x_pos][y_pos][z_pos] = spectrum
            if signal_progress != None:
                signal_progress.emit(file_nr)
        if save_spectra and save_any:
            if save_spec_as_dict:
                pickle.dump(spectra, open('%s/data/spectra.pickle'%(folder_path),'wb'),
                            protocol = pickle.HIGHEST_PROTOCOL)
                max_pixel_spec = np.max(np.array(list(spectra.values())), axis = 0)
                sum_spec = calc_sum_spec(spectra)
            else:
                np.save('%s/data/spectra.npy'%(folder_path), spectra)
                max_pixel_spec = np.max(spectra.reshape((np.prod(spectra.shape[:-1]),4096)), axis = 0)
                sum_spec = np.sum(spectra.reshape((np.prod(spectra.shape[:-1]),4096)), axis = 0)
        
        np.save('%s/data/max_pixel_spec'%(folder_path),max_pixel_spec)
        if save_sum_spec  and save_any: 
            np.save('%s/data/sum_spec'%(folder_path),sum_spec)
    else:                                                                       #for machines with low memory
        print('machine memory to small. get a better one you creep.')
    
    print('spe_loading: parameters_shape:\t', np.array(parameters).shape)
    #np.save('%s/data/fit_bool'%(folder_path),worth_fit)
    if save_counts and save_any:
        np.save('%s/data/counts'%(folder_path),counts)
    if save_parameters and save_any:
        np.save('%s/data/parameters'%(folder_path),parameters)
    print('spe loadingtime - %f'%(t.time()-start))
    if return_values:
        if save_spec_as_dict == False:
            return spectra, parameters, positions, tensor_positions
        else:
            return spectra, parameters


def sum_from_single_files(folder_path, save_sum_spec = True):
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    first_spec = True
    for single_spec_file in iglob('%s/single_spectra/*.npy'%folder_path):
        if first_spec == True:
            sum_spec = np.load(single_spec_file)
            first_spec = False
        else:
            sum_spec += np.load(single_spec_file)
    if save_sum_spec: np.save('%s/data/sum_spec'%folder_path,sum_spec)
    return sum_spec

    
def spe2life_time(file_path):
    ''' 
    This function reads out the life-time in seconds of a .spe-file.
    '''
    life_time = False
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if life_time:
                life_time = int(line)/1000
                break
            elif '$MEAS_TIM' in line:
                life_time = True
    return life_time
    

def spe2real_time(file_path):
    ''' 
    This function reads out the real-time in seconds of a .spe-file.
    '''
    life_time = False
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if life_time:
                life_time = int(line)/1000
                break
            if '$MEAS_TIM' in line:
                life_time = True
    return life_time
    

def norm2sec(spectrum, time):
    '''
    This function normalizes the given spectrum to seconds based on life or real time.
    
    Parameters
    ----------
    spectrum : list
    time : float
    
    Returns
    -------
    list of the normed spectrum
    '''
    spectrum[:] = [i / time for i in spectrum]
    return spectrum


def spe2channels(file_path):
    '''
    This function reads out the channe numbers.
    '''
    channels = False
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if channels == True:
                channels = int(line.split()[1])
                return channels
            elif '$DATA' in line:
                channels = True


def spe_log_content(file_path):
    '''
    This function reads out all the parameters of the scan saved in the .log_file.
    
    Parameters
    ----------
    folder_path: str - path of the folder
    
    Returns
    -------
    [[scan_width],[start],[end],[positions]] = spe_log_content(file_path)
    '''

    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        k = 0
        l = 0
        content = [[],[],[],[]]
        for line in infile:
            if k == 2:
                content[0+l].append(float(line.split()[1]))
                l += 1
                k = 0
            else:
                content[0+l].append(float(line.split()[1]))
                k += 1
        content[3] =[int(content[3][i]) for i in range(3)]
    return content
    
   
def spe_tensor_position(file_path):
    '''
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = spe_tensor_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    '''
    if file_path[-4:] == '.spe':
        with open(file_path, 'r', encoding = 'ISO-8859-1') as infile:
            for line in infile:
                    if 'Ioni' in line:
                        x = float(line.split()[2])
                        y = float(line.split()[3])
                        z = 1
                        return [x,y,z]


def spe_tensor_positions(folder_path, file_type = '.spe'):
    '''
    This function reads out the tensor position of all spe/txt-files in the 
    given folder_path
    
    Parameters
    ----------
    folder_path: str - path of the folder containing the spe or txt files
    
    Returns
    ------- 
    positions = np.array([x0,y0,z0], [x0,y1,z0], ..., [xn,ym,zk])
    '''
    files = glob(folder_path+'*'+file_type)
    positions = []
    for file in files:
        positions.append(spe_tensor_position(file))
    return np.array(positions)
    

def convert_string(string):
    string = string.replace(',','.')
    power = float(string[-2:])
    leading = float(string[:6])
    convert = 10**power*leading/1000
    return convert


def calc_sum_spec(spectrum): 
    '''This function calculates the sum spectrum of all given spectra.'''  
    values = np.asarray(list(spectrum.values()))
    nr_arrays = len(values)
    sum_spec = np.divide(values.sum(axis = 0),nr_arrays)
    return sum_spec

