# -*- coding: utf-8 -*-
"""
This programm will extract the channels of the given .spx-file and save them
to a .txt-file with the same name.
"""
###############################################################################
import numpy as np
import os
import time as t
from glob import iglob
from glob import glob
import codecs
import natsort as ns
import pickle
import psutil ### Module to determine system memory properties
import matplotlib.pyplot as plt
plt.ion()
import h5py
###############################################################################
        
def spx2spec_para(file_path, 
                  save_results=True,
                  print_warning=False):
    ''' 
    This function reads out the spectrum of a .spx-file and reads out the
    detector parameters given in the .spx-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    file_name = os.path.split(file_path)[1]
    ### define default values
    a0 = -0.96
    a1 = 0.01
    FWHM = 0.0792
    Fano = 0.113
    channels = False
    gating_time = 3e-6
    zero_peak_frequency = 1e4
    
    # life_time = False
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '<Channels>' in line:
                spectrum = line.replace('<Channels>','').replace('</Channels>','').replace('\r\n','').replace('    ','').split(',')
                break
            elif '<CalibAbs>' in line:
                a0 = float(line.replace('/','').replace('<CalibAbs>','').replace('\r\n','').replace(',','.'))
            elif '<CalibLin>' in line:
                a1 = float(line.replace('/','').replace('<CalibLin>','').replace('\r\n','').replace(',','.'))
            elif '<ZeroPeakPosition>' in line:
                zero_peak_position = int(line.replace('<ZeroPeakPosition>','').replace('</ZeroPeakPosition>','').replace('\r\n','').replace('    ',''))
            elif '<ZeroPeakFrequency>' in line:
                zero_peak_frequency = int(line.replace('<ZeroPeakFrequency>','').replace('</ZeroPeakFrequency>','').replace('\r\n','').replace('    ',''))
            elif '<SigmaAbs>' in line:
                FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(float(line.replace('/','').replace('<SigmaAbs>','').replace('\r\n','').replace(',','.')))
            elif '<SigmaLin>' in line:
                Fano = float(line.replace('/','').replace('<SigmaLin>','').replace('\r\n','').replace(',','.'))/(3.85e-3) # 3.85eV is die mittlere Energie zur Erzeugung eines Elektron Loch Paares
            elif '<LifeTime>' in line:
                life_time = int(line.replace('<LifeTime>','').replace('</LifeTime>','').replace('\r\n','').replace('    ','')) / 1000.0
            elif '<ZeroPeakFrequency>' in line:
                zero_peak_frequency = int(line.replace('<ZeroPeakFrequency>','').replace('</ZeroPeakFrequency>','').replace('\r\n','').replace('    ',''))
            elif '<RealTime>' in line:
                real_time = int(line.replace('<RealTime>','').replace('</RealTime>','').replace('\r\n','').replace('    ','')) / 1000.0
            elif '<ChannelCount>' in line:
                channels = int(line.replace('<ChannelCount>','').replace('</ChannelCount>','').replace('\r\n','').replace('    ',''))
            elif '<PulsePairResTimeCount>' in line:
                gating_time  = int(line.replace('<PulsePairResTimeCount>','').replace('</PulsePairResTimeCount>','').replace('\r\n','').replace('    ','')) * 1e-6
                if gating_time == 0.0:
                    gating_time = 3e-6
    if channels == False:
        channels = len(spectrum)
    try: spectrum = [int(intensity) for intensity in spectrum]
    except: 
        print('spectrum modified, containing float numbers!')
        spectrum = [float(intensity) for intensity in spectrum]
    ### determine life_time via ROI-methode. Sum over spectrum_tmp[75:116] and
    ### divide by zero peak frequency (10000 per second for M4)
    ### check if detector is set in 10keV, 20keV or 40keV and manipulate the
    ### spectrum in order to represent a 40keV spectrum
    max_energy = a0 + a1 * (channels-1)
    factor = (1/a1)/100
    life_time = np.sum(spectrum[zero_peak_position-int(20*factor):zero_peak_position+int(20*factor)]) / zero_peak_frequency
    if life_time == 0:
        print('error in %s \nlife_time is ZERO\nsetting life_time to 1'%file_path)
        life_time = 1
    parameters = np.array([a0,a1,Fano,FWHM, life_time, max_energy, gating_time, real_time])
    spectrum = np.divide(spectrum, life_time)
    ### save data if decided
    if save_results:
        folder_path = os.path.split(file_path)[0]
        try: os.mkdir('%s/data/'%folder_path)
        except: pass
        if os.path.exists(f'{folder_path}/data/data.h5'):
            with h5py.File(f'{folder_path}/data/data.h5', 'r+') as tofile:
                if file_name in tofile.keys():
                    del tofile[file_name]
        else:
            empty_file = h5py.File(f'{folder_path}/data/data.h5', 'w')
            empty_file.close()
        with h5py.File(f'{folder_path}/data/data.h5', 'r+') as tofile:
            tofile.create_dataset(f'{file_name}/spectra',
                                  data = spectrum,
                                  compression = 'gzip', compression_opts = 9)
            tofile.create_dataset(f'{file_name}/sum spec', data = spectrum,
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/counts', data = [np.sum(spectrum)],
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/parameters', data = parameters,
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/max pixel spec', data = spectrum,
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/position dimension', data = np.array([1,1,1]),
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/tensor positions', data = np.array([[0,0,0]]),
                                  compression="gzip")
            tofile.create_dataset(f'{file_name}/positions', data = np.array([[0,0,0]]),
                                  compression="gzip")
    return spectrum, parameters
      
def many_spx2spec_para(folder_path, signal = None ,worth_fit_threshold = 200,
                       save_sum_spec = True, save_spectra = True, 
                       save_counts = True, save_parameters = True,
                       save_any = True, print_warning = False,
                       save_spec_as_dict = True,
                       return_values = False):
    ''' 
    This function reads out the spectrum of a .spx-file and reads out the
    detector parameters given in the .spx-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.
    index : int
        the number of the spectrum in the measurement set
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    file_name = os.path.split(folder_path)[1]
    if os.path.exists(f'{folder_path}/data/data.h5'):
        with h5py.File(f'{folder_path}/data/data.h5', 'r+') as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    worth_fit = []
    counts = []
    parameters = []
    # spectra = {}
    zero_peak_frequency = 1e4
    signal_progress = signal
    folder_size = 0
    zero_peak_position = 96
    zero_peak_frequency = 10000
    machine_memory = psutil.virtual_memory().total
    start = t.time()
    ### derive the position of the spx-file
    if save_spec_as_dict == False:
        positions = spx_tensor_positions(folder_path)
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
    sorted_folder = ns.natsorted(glob('%s/*.spx'%folder_path))
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    
    folder_size = sum(os.path.getsize(f) for f in sorted_folder if os.path.isfile(f))
    life_time = False
    if (folder_size / machine_memory) < 0.7:                              ### for machines with big memory
        print('machine memory big enough. creating spectra dict')
        for file_nr, spx_file in enumerate(sorted_folder):
            with open(spx_file,'r', encoding = 'ISO-8859-1') as infile:
                for line in infile:
                    if '<LifeTime>' in line:
                        line = line.replace('<LifeTime>','').replace('</LifeTime>','').replace('\r\n','').replace('    ','').replace(',','.')
                        life_time = int(line) / 1000.0
                    elif '<RealTime>' in line:
                        real_time = int(line.replace('<RealTime>','').replace('</RealTime>','').replace('\r\n','').replace('    ','')) / 1000.0
                    elif '<ZeroPeakPosition>' in line:
                            zero_peak_position = int(line.replace('<ZeroPeakPosition>','').replace('</ZeroPeakPosition>','').replace('\r\n','').replace('    ',''))
                    elif '<ZeroPeakFrequency>' in line:
                            zero_peak_frequency = int(line.replace('<ZeroPeakFrequency>','').replace('</ZeroPeakFrequency>','').replace('\r\n','').replace('    ',''))
                    elif '<PulsePairResTimeCount>' in line:
                        gating_time  = int(line.replace('<PulsePairResTimeCount>','').replace('</PulsePairResTimeCount>','').replace('\r\n','').replace('    ','')) * 1e-6
                        if gating_time == 0.0:
                            gating_time = 3e-6 
                    elif '<ChannelCount>' in line:
                        channels = int(line.replace('<ChannelCount>','').replace('</ChannelCount>','').replace('\r\n','').replace('    ',''))
                    elif '<CalibAbs>' in line:
                        a0 = float(line.replace('/','').replace('<CalibAbs>','').replace('\r\n','').replace(',','.'))
                    elif '<CalibLin>' in line:
                        a1 = float(line.replace('/','').replace('<CalibLin>','').replace('\r\n','').replace(',','.'))
                    elif '<SigmaAbs>' in line:
                        FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(float(line.replace('/','').replace('<SigmaAbs>','').replace('\r\n','').replace(',','.')))  
                    elif '<SigmaLin>' in line:
                        Fano = float(line.replace('/','').replace('<SigmaLin>','').replace('\r\n','').replace(',','.'))/(3.85e-3) # 3.85eV is die mittlere Energie zur Erzeugung eines Elektron Loch Paares
                    elif '<Channels>' in line:
                        spectrum = line.replace('<Channels>','').replace('</Channels>','').replace('\r\n','').replace('    ','').split(',')
                        break

            try: spectrum = [int(intensity) for intensity in spectrum]
            except: spectrum = [float(intensity) for intensity in spectrum]
            
            ### calculate the life time from the zero peak
            ### detector is set in 10keV, 20keV or 40keV and maipulate the
            ### spectrum in order to represent an 40keV spectrum
            factor = (1/a1)/100
            if not (94 < zero_peak_frequency < 99):
                zero_peak_position = int(zero_peak_position/factor)
            a1 *= factor
            max_energy = a0 + a1 * (channels-1)

            life_time = np.sum(spectrum[zero_peak_position-int(20/factor):zero_peak_position+int(20/factor)]) / zero_peak_frequency * factor
            if life_time == 0:
                print('factor:\t', factor)
                print('zero peak position and frequency:\t', zero_peak_position, zero_peak_frequency)
                print('error in %s \nlife_time is ZERO\nsetting life_time to 1'%spx_file)
                life_time = 1
            parameters.append([a0, a1, Fano, FWHM, life_time, max_energy, gating_time, real_time])
            spectrum = np.divide(spectrum, life_time)
            ### sometimes the number of channels are corrupted, add or remove channels
            ### to comply with 4096
            if channels < 4096:
                spectrum = np.r_[spectrum, np.zeros(4096-channels)]
            elif channels > 4096:
                spectrum = spectrum[:4096]
            
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
            with h5py.File('%s/data/data.h5'%folder_path, 'r+') as tofile:
                tofile.create_dataset(f'{file_name}/spectra',
                                      data = np.array(list(spectra.values())),
                                      compression = 'gzip', compression_opts = 9)
            if save_spec_as_dict:
                max_pixel_spec = np.max(np.array(list(spectra.values())), axis = 0)
                sum_spec = calc_sum_spec(spectra)
            else:
                max_pixel_spec = np.max(spectra.reshape((np.prod(spectra.shape[:-1]),4096)), axis = 0)
                sum_spec = np.sum(spectra.reshape((np.prod(spectra.shape[:-1]),4096)), axis = 0)
        
    else:                                                                       #for machines with low memory
        print('machine memory to small. creating single spectra')
        try: os.mkdir('%s/single_spectra/'%folder_path)
        except: pass
        for file_nr, spx_file in enumerate(sorted_folder):
            with open(spx_file,'r', encoding = 'ISO-8859-1') as infile:
                for line in infile:
                    if '<LifeTime>' in line:
                        life_time = int(line.replace('<LifeTime>','').replace('</LifeTime>','').replace('\r\n','').replace('    ','')) / 1000.0
                    elif '<RealTime>' in line:
                        real_time = int(line.replace('<RealTime>','').replace('</RealTime>','').replace('\r\n','').replace('    ','')) / 1000.0
                    elif '<PulsePairResTimeCount>' in line:
                        gating_time  = int(line.replace('<PulsePairResTimeCount>','').replace('</PulsePairResTimeCount>','').replace('\r\n','').replace('    ','')) * 1e-6
                        if gating_time == 0.0:
                            gating_time = 3e-6 
                    elif '<ZeroPeakPosition>' in line:
                         zero_peak_position = int(line.replace('<ZeroPeakPosition>','').replace('</ZeroPeakPosition>','').replace('\r\n','').replace('    ',''))
                    elif '<ZeroPeakFrequency>' in line:
                         zero_peak_frequency = int(line.replace('<ZeroPeakFrequency>','').replace('</ZeroPeakFrequency>','').replace('\r\n','').replace('    ',''))
                    elif '<ChannelCount>' in line:
                        channels = int(line.replace('<ChannelCount>','').replace('</ChannelCount>','').replace('\r\n','').replace('    ',''))
                    elif '<CalibAbs>' in line:
                        a0 = float(line.replace('/','').replace('<CalibAbs>','').replace('\r\n',''))
                    elif '<CalibLin>' in line:
                        a1 = float(line.replace('/','').replace('<CalibLin>','').replace('\r\n',''))
                    elif '<SigmaAbs>' in line:
                        FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(float(line.replace('/','').replace('<SigmaAbs>','').replace('\r\n','')))   
                    elif '<SigmaLin>' in line:
                        Fano = float(line.replace('/','').replace('<SigmaLin>','').replace('\r\n',''))/(3.85e-3) # 3.85eV is die mittlere Energie zur Erzeugung eines Elektron Loch Paares
                    elif '<Channels>' in line:
                        spectrum = line.replace('<Channels>','').replace('</Channels>','').replace('\r\n','').replace('    ','').split(',')
                        break
            
            max_energy = a0 + a1*(channels-1)
            
            try: spectrum = [int(intensity) for intensity in spectrum]
            except: spectrum = [float(intensity) for intensity in spectrum]
            
            ### calculate the life time from the zero peak and check if the
            ### detector is set in 10keV, 20keV or 40keV and maipulate the
            ### spectrum in order to represent an 40keV spectrum
            factor = (1/a1)/100
            life_time = np.sum(spectrum[zero_peak_position-int(20*factor):zero_peak_position+int(20*factor)]) / zero_peak_frequency
            parameters.append([a0, a1, Fano, FWHM, life_time, max_energy, gating_time])
            spectrum = np.divide(spectrum, life_time)

            sum_spec = sum(spectrum)
            counts.append(sum_spec)
            if sum_spec > worth_fit_threshold:
                worth_fit.append(True)
            else:
                worth_fit.append(False)
            
            np.save('%s/single_spectra/spectrum_%d'%(folder_path,file_nr),spectrum)  
            if signal_progress != None:
                signal_progress.emit(file_nr)
        sum_spec = sum_from_single_files(folder_path)
    
    ### transform all data objects to array
    parameters = np.array(parameters)
    print('spx_loading: parameters_shape:\t', np.array(parameters).shape)
    with h5py.File('%s/data/data.h5'%folder_path, 'r+') as tofile:
        tofile.create_dataset(f'{file_name}/sum spec', data = sum_spec,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/counts', data = counts,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/parameters', data = parameters,
                              compression="gzip")
        tofile.create_dataset(f'{file_name}/max pixel spec', data = max_pixel_spec,
                              compression="gzip")
    print('spx loadingtime - %f'%(t.time()-start))
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
    
def spx2life_time(file_path):
    ''' 
    This function reads out the life-time in seconds of a .spx-file.
    '''
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '<LifeTime>' in line:
                life_time = int(line.replace('<LifeTime>','').replace('</LifeTime>','').replace('\r\n','').replace('    ',''))
                life_time /= 1000.0
                break
            if '<RealTime>' in line:
                life_time = int(line.replace('<RealTime>','').replace('</RealTime>','').replace('\r\n','').replace('    ',''))
                life_time /= 1000.0
                break
    return life_time
    
def spx2real_time(file_path):
    ''' 
    This function reads out the real-time in seconds of a .spx-file.
    '''
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '<RealTime>' in line:
                real_time = int(line.replace('<RealTime>','').replace('</RealTime>','').replace('\r\n','').replace('    ',''))
                real_time /= 1000.0
                break
    return real_time
    
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

def spx2channels(file_path):
    '''
    This function reads out the channe numbers.
    '''
    with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '<ChannelCount>' in line:
                channels = int(line.replace('<ChannelCount>','').replace('</ChannelCount>','').replace('\r\n','').replace('    ',''))
                break
    return channels

def spx_log_content(file_path):
    '''
    This function reads out all the parameters of the scan saved in the .log_file.
    
    Parameters
    ----------
    folder_path: str - path of the folder
    
    Returns
    -------
    [[scan_width],[start],[end],[positions]] = spx_log_content(file_path)
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
   
def spx_tensor_position(file_path):
    '''
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = spx_tensor_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    '''
    if file_path[-4:] == '.spx':
        with open(file_path, 'r', encoding = 'ISO-8859-1') as infile:
            for line in infile:
                    if '<Data' in line:
                        encoded = line.split('>')[1].split('<')[0]
                        data = codecs.decode(encoded.encode('ascii'), 'base64')
                        if len(data) == 162:
                            return np.frombuffer(data[1:-1], dtype = np.float64)[15:18]
                        elif len(data) == 161:
                            return np.frombuffer(data[1:], dtype = np.float64)[15:18]
                        else:
                            return np.frombuffer(data[1:], dtype = np.float64)[15:18]
    if len(file_path.split('(')) != 1:
        if file_path[-4:] == '.txt':
            position = file_path.split('(')[2].replace(').txt','')
        elif file_path[-4:] == '.spx':
            position = file_path.split('(')[2].replace(').spx','')
        position = position.split(',')
        return [float(position[i]) for i in range(3)]          
    elif len(file_path.split('(')) == 1:
        return [0,0,float(file_path.split('_')[-1].replace('z','').replace('.spx',''))]

def spx_tensor_positions(folder_path, file_type = '.spx'):
    '''
    This function reads out the tensor position of all spx/txt-files in the 
    given folder_path
    
    Parameters
    ----------
    folder_path: str - path of the folder containing the spx or txt files
    
    Returns
    ------- 
    positions = np.array([x0,y0,z0], [x0,y1,z0], ..., [xn,ym,zk])
    '''
    files = glob(folder_path+'*'+file_type)
    positions = []
    for file in files:
        positions.append(spx_tensor_position(file))
    return np.array(positions)
    
def log_file_type(log_file):
    ''' This function defines the type of the given .log-file by reading the 
    header and scanning it for a keyphrase
    '''
    with open(log_file, 'r', encoding = 'ISO-8859-1') as log_file:
        log_file_type = None
        for line in log_file:
            if 'Scan started' in line:
                log_file_type = 'Louvre'
                break
    return log_file_type

def Louvre_log_file_content(log_file):
    ''' This function reads out the width, start and end position of the 
    measurement as stated in the .log-file
    '''
    with open(log_file,'r', encoding = 'ISO-8859-1') as log_file:
        parameters = [2, 0, 1]    ### ['width','start','end']
        width_start_end = []
        axes_parameters = []
        for line in log_file:
            if 'Start' in line:
                line = line.split()
                line = [convert_string(line[2]),convert_string(line[4]),convert_string(line[6])]
                axes_parameters.append(line)
                if len(axes_parameters) == 3:                                       ### avoid reading the complete log_file
                    log_file.close()
                    break
        for i in range(3):
            width_start_end.append([axes_parameters[j][parameters[i]] for j in range(3)])
        width_start_end.append([int(round(abs((width_start_end[2][i]-width_start_end[1][i])/width_start_end[0][i])+1))for i in range(3)])
   
    return width_start_end

def Louvre_tensor_position(log_file):

    tensor_position =  {}     
    temp = 0
    with open(log_file, 'r+', encoding = 'ISO-8859-1') as log_file:
        for line in log_file:
            if 'X  Y  Z' in line: 
                temp = 1
            elif temp == 1:
                positions = line.split()
                positions = [float(positions[i])/1000 for i in range(len(positions))]  
                temp = 2
            elif temp == 2:
                tensor_position[int(line.replace('corresponding to spectrum No', ''))] = positions  
                temp = False
    return tensor_position

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
