# -*- coding: utf-8 -*-
"""
This programm will extract the channels of the given .mca-file and save them
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
import h5py
import psutil ### Module to determine system memory properties
import matplotlib.pyplot as plt
plt.ion()

###############################################################################
        
        
def mca2spec_para(file_path,  XANES = True, print_warning = False):
    ''' 
    This function reads out the spectrum and detector parameters given in the 
    .mca-file depending on which kind of .mca file is loaded (see function 
    check_mca_type).
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .mca-file.
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    ### define file independent default values
    gating_time = 3e-6
    spectrum = []
    data_start = False
    ### check the mca_file type
    mca_type = check_mca_type(file_path)
    if mca_type == 'XPS-FP2':
        a0 = 0.5
        a1 = 0.01
        FWHM = 0.08
        Fano = 0.112
        channels = 4096
        with open(file_path,'r') as infile:
            for line in infile:
                if '<<END>>' in line:
                    ### end of data
                    break
                elif data_start:
                    ### read out spectrum
                    spectrum.append(int(line.replace('\n','')))
                elif '<<DATA>>' in line:
                    ### data starts
                    data_start = True
                elif 'LIVE_TIME' in line:
                    life_time = float(line.split('-')[-1])
                    real_time = float(line.split('-')[-1])
    elif mca_type == 'measurement':
        a0 = 0.0
        a1 = 0.0092
        FWHM = 0.045
        Fano = 0.148
        with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
            for line in infile:
                if data_start:
                    ### read out spectrum
                    for data in line.replace('\\','').split():
                        spectrum.append(int(data))
                elif '@A' in line:
                    ### start of the spectrum
                    for data in line.replace('@A','').replace('\\','').split():
                        spectrum.append(int(data))
                    ### set the start of the data to true
                    data_start = True
                elif '@CTIME' in line:
                    ### read out the measurement time
                    life_time = float(line.split()[2])
                    real_time = float(line.split()[-1])
                elif '@CHANN' in line:
                    ### read out the channels
                    channels = int(line.split()[1])
    life_time_mca = life_time
    
    ### determine life_time via ROI-methode. Sum over spectrum_tmp[75:116] and
    ### divide by zero peak frequency (10000 per second for M4)
    ### check if detector is set in 10keV, 20keV or 40keV and manipulate the
    ### spectrum in order to represent a 40keV spectrum
    max_energy = a0 + a1 * (channels-1)
    parameters = [a0,a1,Fano,FWHM, life_time, max_energy, gating_time, real_time]
    spectrum = np.divide(spectrum, life_time)

    if print_warning:
        if np.abs(1-life_time_mca / life_time) > 0.04:
            print('high difference in life_time calculation in file:\t', file_path)
            print('difference:\t %.3f %%'%(np.abs(1-life_time_mca / life_time)*100))
            print('mca life-time:\t %.3f s'%life_time_mca)
            print('ROI life-time:\t %.3f s'%life_time)
    return np.array(spectrum), np.array(parameters)
      

def many_mca2spec_para(folder_path, XANES = False, signal = None ,
                       worth_fit_threshold = 200,
                       save_sum_spec = True, save_spectra = True, 
                       save_counts = True, save_parameters = True,
                       save_any = True, print_warning = False,
                       save_spec_as_dict = True,
                       return_values = False):
    ''' 
    This function reads out the spectrum of a .mca-file and reads out the
    detector parameters given in the .mca-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .mca-file.
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
    signal_progress = signal
    folder_size = 0
    a0 = 0.0
    a1 = 0.0102
    FWHM = 0.06
    Fano = 0.120
    gating_time = 3e-6
    machine_memory = psutil.virtual_memory().total * 1E-9
    start = t.time()
    ### derive the position of the mca-file
    if save_spec_as_dict == False:
        positions, len_scans = mca_tensor_positions(folder_path, XANES = XANES)
        x = np.unique(positions[:,0])
        z = np.unique(positions[:,1])
        monoE = np.unique(positions[:,2])
        ### calculate the stepsizes in every direction
        try: x_steps = x[1] - x[0]
        except: x_steps = 1
        try: z_steps = z[1] - z[0]
        except: z_steps = 1
        try: monoE_steps = monoE[1] - monoE[0]
        except: monoE_steps = 1
        ### calculate the tensor positions by dividing positions by steps
        tensor_positions = np.copy(positions)
        ### now subtract to 0
        tensor_positions[:,0] -= x[0]
        tensor_positions[:,1] -= z[0]
        tensor_positions[:,2] -= monoE[0]
        tensor_positions /= [x_steps, z_steps, monoE_steps]
        tensor_positions = np.array(tensor_positions, dtype = int)
    sorted_folder = np.sort(glob('%s/*.mca'%folder_path))
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    
    ### read out ionization current from .spec file
    spec_file_path = glob('/'.join(folder_path.split('/')[:-1])+'/*.spec')[0]
    nr_scans, len_scans = read_nr_len_scans_from_spec(spec_file_path)
    scan_0, _ = [int(tmp.replace('.mca','')) for tmp in sorted_folder[0].split('/')[-1].split('_')[-2:]]
    ionization_current = read_ionization_spec(spec_file_path, nr_scans, 
                                              len_scans = len_scans,
                                              scan_0 = scan_0)
    # print('len_scans:\t', len_scans)
    ### read out spectra
    folder_size = sum(os.path.getsize(f) for f in sorted_folder if os.path.isfile(f)) * 1E-9 
    life_time = False
    spectra = {}
    scan_tmp, scan_offset = 0, 0
    iterator = 0
    if (folder_size / machine_memory) < 0.7:                              ### for machines with big memory
        print('machine memory big enough. creating spectra dict')
        for file_nr, mca_file in enumerate(sorted_folder):
            ### determine number of scan which the file belongs to
            scan, point = [int(tmp.replace('.mca','')) for tmp in mca_file.split('/')[-1].split('_')[-2:]]
            if len_scans['#S %d'%scan] == 0:
                continue
            file_nr = point
            for i in np.arange(scan_0, scan-1-scan_offset):
                file_nr += (len_scans['#S %d'%(i)]+1)
            scan -= (scan_0 -1 +scan_offset)
            if np.abs(np.subtract(scan, scan_tmp)) > 1:
                scan_offset += 1
                scan -= 1
            # print('scan, scan_tmp, scan_offset:\t', scan, scan_tmp, scan_offset)
            scan_tmp = int(scan)
            # print('file_nr, scan, point:\t', file_nr, scan, point)
            spectrum = []
            data_start = False
            with open(mca_file,'r', encoding = 'ISO-8859-1') as infile:
                for line in infile:
                    if data_start:
                        for data in line.replace('\\','').split():
                            spectrum.append(int(data))
                    elif '@A' in line:
                        for data in line.replace('@A','').replace('\\','').split():
                            spectrum.append(int(data))
                        data_start = True
                    elif '@CTIME' in line:
                        life_time = float(line.split()[2])
                        real_time = float(line.split()[-1])
                    elif '@CHANN' in line:
                        channels = int(line.split()[1])
            spectrum = np.asarray(spectrum)            
            
            max_energy = a0 + a1 * (channels-1)
            parameters.append([a0, a1, Fano, FWHM, life_time, max_energy, gating_time, real_time])
            spectrum = np.divide(spectrum, life_time)
            ### norm spectrum to ionization current
            # print('ion_cur[%d][%d]'%(scan, point), ionization_current[scan][point], ionization_current[scan].shape )
            spectrum = np.divide(spectrum, ionization_current[str(scan-1)][point])
            #print(life_time, ionization_current[scan-1][point], scan-1, point)
            sum_spec_tmp = sum(spectrum)
            counts.append(sum_spec_tmp)
            if sum_spec_tmp > worth_fit_threshold:
                worth_fit.append(True)
            else:
                worth_fit.append(False)
            ### now we try to rebuild specfit load in routine to support array
            ### spectra
            if save_spec_as_dict:
                spectra['%i'%iterator] = spectrum
            else:
                if file_nr == 0:
                    spectra = np.empty((len(x), len(z), len(monoE), 4096))
                x_pos, z_pos, monoE_pos = tensor_positions[file_nr]
                spectra[x_pos][z_pos][monoE_pos] = spectrum
            if signal_progress != None:
                signal_progress.emit(file_nr)
            iterator += 1
        with h5py.File('%s/data/data.h5'%folder_path, 'w') as tofile:
            tofile.create_dataset('spectra',
                                  data = np.array(list(spectra.values())),
                                  compression = 'gzip', compression_opts = 9)
        max_pixel_spec = np.max(np.array(list(spectra.values())), axis = 0)
        sum_spec = calc_sum_spec(spectra)
        
    else:                                                                       #for machines with low memory
        print('machine memory to small. get a better machine you loser')
    
    with h5py.File('%s/data/data.h5'%folder_path, 'a') as tofile:
        tofile.create_dataset('sum spec', data = sum_spec,
                              compression="gzip")
        tofile.create_dataset('counts', data = counts,
                              compression="gzip")
        tofile.create_dataset('parameters', data = parameters,
                              compression="gzip")
        tofile.create_dataset('max pixel spec', data = max_pixel_spec,
                              compression="gzip")
    print('mca loadingtime - %f'%(t.time()-start))
    if return_values:
        if save_spec_as_dict == False:
            return spectra, parameters, positions, tensor_positions
        else:
            return spectra, parameters


def read_ionization_spec(spec_file_path, nr_scans, len_scans, scan_0 = 1):
    #ionization_current = [[]]*len(len_scans)
    ionization_current={}
    for la in range(len(len_scans)):
        ionization_current[str(la)]=[]
    read_line = False
    
    with open(spec_file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '#S' in line:
                ### sometimes the scan has no values stored, blind value
                scan = ' '.join(line.split()[:2])
            if '#L' in line:
                try: where = line.split().index('AS_IC')#'ionch1')  ######### edit: AS_IC mittelt über Messzeit
                except: where = line.split().index('ionch1')
                nr_variables = len(line.split())-1
                read_line = True
                continue
            elif '#C' in line:
                read_line = False
                continue
            elif line == '\n':
                read_line = False
                continue
            if read_line:
                if len(line.split()) != nr_variables: continue
                if len_scans[scan] != 0:
                    #ionization_current[int(scan.split()[-1])-1].append(float(line.split()[where-1]))
                    ## edit: AS_IC is in units of 10e-10, nomalize here
                    ionization_current[str(int(scan.split()[-1])-1)].append(float(line.split()[where-1])/10E-10)  
                    #print(float(line.split()[where-1]), int(scan.split()[-1])-1)

    for scan in ionization_current:
        if scan == []:
            ionization_current.remove([])
            nr_scans -= 1
    #norm_ion_cur = np.abs([ionization_current[scan]/np.max(ionization_current[scan]) for scan in range(nr_scans)])
    #ionization_current=np.divide(ionization_current,1E-10)
    #print(ionization_current, ionization_current.shape)
    return ionization_current #norm_ion_cur  #### edit: nicht auf 1 normieren, sondern auf 10e-10, s. Z 271
    

def read_nr_len_scans_from_spec(spec_file_path):
    nr_scans = 0
    len_scans = {}
    current_scan_nr = ''
    with open(spec_file_path,'r', encoding = 'ISO-8859-1') as infile:
        for line in infile:
            if '#S' in line: 
                line = line.split()
                current_scan_nr = ' '.join(line[:2]) 
                if line[2] == 'acquire':
                    len_scans[current_scan_nr]=1
                if line[2] == 'eigerloopscan':
                    len_scans[current_scan_nr]=1
                    #print(current_scan_nr)
                else:
                    len_scans[current_scan_nr] = int(line[-2])
                nr_scans += 1
                continue
            elif 'aborted after' in line:
                len_scan = int(line.split()[-2])
                len_scans[current_scan_nr] = len_scan
                if len_scan == 0:
                   nr_scans -= 1 
    return nr_scans, len_scans


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

    
def mca2life_time(file_path):
    ''' 
    This function reads out the life-time in seconds of a .mca-file.
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
    

def mca2real_time(file_path):
    ''' 
    This function reads out the real-time in seconds of a .mca-file.
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


def mca2channels(file_path):
    '''
    This function reads out the channe numbers.
    '''
    mca_type = check_mca_type(file_path)
    if mca_type == 'XPS-FP2':
        return 4096
    elif mca_type == 'measurement':
        with open(file_path,'r', encoding = 'ISO-8859-1') as infile:
            for line in infile:
                if '@CHANN' in line:
                    channels = int(line.split()[1])
                    break
        return channels


def mca_tensor_position(file_path, XANES = False):
    '''
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = mca_tensor_position(file_path, XANES = bool)
    It determines whether it is a line scan or a 3D-Scan.
    If XANES = True, the position is read out from the monoE parameter
    '''
    file_name = file_path.split('/')[-1].split('_')
    point = int(file_name[-1].replace('.mca', ''))
    if file_path[-4:] == '.mca':
        with open(file_path, 'r', encoding = 'ISO-8859-1') as infile:
            for line in infile:
                if '#S' in line:
                    lines = line.split()
                    if lines[2] == 'eigermesh':
                        for line in infile:
                            if '#P0' in line:
                                line = line.split()
                                x = np.round(float(line[5]),3)
                                y = np.round(float(line[6]),3)
                            elif '#P7' in line:
                                line = line.split()
                                z = np.round(float(line[3]),3)
                                # print('x,y,z', x,y,z)
                                return[x,y,z]
                    if lines[3] == 'm1':
                        # print('larve')
                        for line in infile:
                            if '#P0' in line:
                                line = line.split()
                                z = np.round(float(line[1]),3)
                                y = np.round(float(line[2]),3)
                                x = np.round(float(line[3]),3)
                                #print ('test', x, y, z)
                                return[x,y,z]
                    
                    if lines[3] == 'monoE':
                        # print('XANES')
                        for line in infile:
                            if '#P8' in line:
                                line = line.split()
                                z = np.round(float(line[3]),4)                                
                                y = np.round(float(line[1]),4)
                                x = np.round(float(line[2]),4)
#                                y = np.round(float(line[1]),4)
#                                z = np.round(float(line[3]),4)
                                print ('test', x, y, z)
                                return[x,y,z]
                    if lines[3] == 'm2':
                        print('coins') 
                        for line in infile:
                            if '#P0' in line:
                                line = line.split()
                                x = np.round(float(line[2]),3)
                                y = np.round(float(line[3]),3)
                                z = np.round(float(line[1]),3)
                                #print ('test', x, y, z)
                                return[x,y,z]
                    else:
                        # print('biene')
                        for line in infile:
                            if '#P0' in line:
                                line = line.split()
                                x = np.round(float(line[1]),3)
                                y = np.round(float(line[2]),3)
                                z = np.round(float(line[3]),3)
                                #print ('test', x, y, z)
                                return[x,y,z]


def mca_tensor_positions(folder_path, file_type = '.mca', XANES = False):
    '''
    This function reads out the tensor position of all mca/txt-files in the 
    given folder_path
    
    Parameters
    ----------
    folder_path: str 
        path of the folder containing the mca or txt files
    file_type : str
        type of the file
    XANES : bool
        if true, monoE is used as z coordinate
    
    Returns
    ------- 
    positions = np.array([x0,y0,z0], [x0,y1,z0], ..., [xn,ym,zk])
    '''
    if type(folder_path) == str:
        files = sorted(glob(folder_path+'*'+file_type))
    elif type(folder_path) == list:
        files = folder_path
    nr_scans = np.unique([int(file.split('/')[-1].split('_')[-2]) for file in files])
    len_scans = np.zeros(len(nr_scans), dtype = int)
    for file in files:
        len_scans[int(file.split('/')[-1].split('_')[-2])-nr_scans[0]] += 1
    positions = []
    for file in files:
        positions.append(mca_tensor_position(file, XANES = XANES))
    # print('positions, len_scans:\t', positions[0], len_scans[0])
    return positions, len_scans
    

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

def check_mca_type(mca_file_path):
    """
    Function to decide what kind of .mca file is provided. Up to date only 
    measurement files from beamlines and simulation files generated by the 
    software XPS-FP2 are supported and destinguished.

    Parameters
    ----------
    mca_file_path : str
        Absolute path to the .mca file.

    Returns
    -------
    str
        description of the file, either 'XPS-FP2' as coming from this software
        or 'measurment' coming from beamlines
    """
    with open(mca_file_path, 'r') as mca_file:
        for line in mca_file:
            if 'PMCA SPECTRUM' in line:
                return 'XPS-FP2'
            else:
                return 'measurement'
    
    
