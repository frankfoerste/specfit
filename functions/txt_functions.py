# -*- coding: utf-8 -*-
"""
This programm will extract the channels of the given .spx-file and save them
to a .txt-file with the same name.
"""
###############################################################################
import psutil
import natsort as ns
from glob import glob
from glob import iglob

import os
import time as t
import numpy as np
import pickle
import specfit_GUI_functions as sfunc
###############################################################################
def txt2spec_para(file_path):
    '''
    This function reads out the txt_files set up by 3D-measurements. It returns
    the spectrum and the parameters:
        spectrum, parameters = spx_txt2spec_para(file_path)
    '''
    spectrum = []
    a0 = 0.
    a1 = 0.001
    Fano = 0.115
    FWHM = 0.2
    life_time = 1
    gating_time = 3e-6
    
    start = 100 ### value to be sure, that the spectrum doesn't start beforehand
    FWHM = None
    with open(file_path, 'r') as infile:
        for i, line in enumerate(infile):
            if 'lin.' in line:
                a1 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
            elif 'abs.' in line:
                a0 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
            elif 'FWHM' in line:
                try:
                    FWHM = float(line.split()[1]) / 1000.0
                except:
                    FWHM = float(line.split()[-1]) / 1000.0
            elif 'Fano' in line:
                try:
                    Fano = float(line.split()[1])
                except:
                    Fano = float(line.split()[-1])
            elif 'Life time:' in line:
                life_time = float(line.split()[-1].strip()) / 1000.0
            elif 'Life-Zeit' in line:
                life_time = float(line.split()[1].strip()) / 1000.0
            elif 'Channels:' in line or( 'Kan' in line and 'le:' in line):
                channels = int(line.split()[1].strip())
            elif 'Energy' in line or 'Energie' in line:
                start = int(i)
            elif i == 0:
                try: float(line.split()[0])
                except: continue
                line = line.split()
                try: spectrum.append(int(line[1]))
                except: spectrum.append(int(float(line[0])))
                try:
                    line[1]
                    a0 = float(line[0])
                except: pass
                start = 0
            elif i > start:
                line = line.split()
                try: spectrum.append(int(line[1]))
                except: spectrum.append(int(float(line[0])))
                if i == 1:
                    try: 
                        line[1]
                        a1 = np.round(float(line[0])-a0,3)
                    except: pass
#    assert len(spectrum) == channels
#    spectrum = np.divide(spectrum,life_time)
    
    if FWHM == None:
        parameters = [a0, a1, 0.115, 0.2, 1, a0 + a1 * i, gating_time]
    else: 
        parameters = [a0,a1,Fano,FWHM, life_time, a0 + a1 * (channels-1), gating_time]
    return np.asarray(spectrum), parameters


def many_txt2spec_para(folder_path, signal):
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
  # worth_fit = []
    counts = []
    parameters = []
    spectra = {}
    file_nr = 0
    signal_progress = signal
    folder_size = 0
    machine_memory = psutil.virtual_memory().total * 1E-9
    start_time = t.time()
    try: os.mkdir('%s/data/'%folder_path)
    except: pass
    sorted_folder = ns.natsorted(glob('%s/*.txt'%folder_path))
    with open('%s/data/sorted_SpecFit.dat'%folder_path, 'w+', encoding="ISO-8859-1") as f:
        for i in range(len(sorted_folder)):
            f.write('(%d) %s \n'%(i,sorted_folder[i]))
            
    for f in os.listdir(folder_path):
        if os.path.isfile('%s/%s'%(folder_path, f)) == True:
            folder_size += os.path.getsize('%s/%s'%(folder_path, f)) * 1E-9

    if (folder_size / machine_memory) < 0.12:                              ### for machines with big memory
        print('machine memory big enough. creating spectra dict')
        for txt_file in ns.natsorted(glob('%s/*.txt'%folder_path)):
            spectrum_tmp = []
            start = 100
            with open(txt_file,'r', encoding="ISO-8859-1") as infile:
                for i, line in enumerate(infile):
                    if 'lin.' in line:
                        a1 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif 'abs.' in line:
                        a0 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif 'FWHM' in line:
                        try:
                            FWHM = float(line.split()[-1]) / 1000.0
                        except:
                            FWHM = float(line.split()[1]) / 1000.0
                    elif 'Fano' in line:
                        try:
                            Fano = float(line.split()[-1])
                        except:
                            Fano = float(line.split()[1])
                    elif 'Energy' in line or 'Energie' in line:
                        start = i
                    elif 'Life time:' in line:
                        life_time = float(line.split()[-1].strip()) / 1000.0
                    elif 'Life-Zeit' in line:
                        life_time = float(line.split()[1].strip()) / 1000.0
                    elif 'Channels:' in line or ('Kan' in line and 'le:' in line): #annoying problems with ä
                        channels = int(float(line.split()[1].strip()))
                    elif i > start:
                        line = line.split()
                        spectrum_tmp.append(int(float(line[1])))
            gating_time = 3e-6
            parameters_tmp = [a0, a1, Fano, FWHM, life_time, a0 + a1 * (channels-1), gating_time]
            counts.append(sum(spectrum_tmp))
         #   if sum(spectrum_tmp) > 200:
        #        worth_fit.append(True)
       #     else:
       #         worth_fit.append(False)
            spectra['%i'%file_nr] = np.divide(spectrum_tmp,life_time)
#            np.save('%s/single_spectra/spectrum_%d'%(folder_path,file_nr),spectrum)  
            parameters.append(parameters_tmp)
            file_nr += 1
            signal_progress.emit(file_nr)
        pickle.dump(spectra, open('%s/data/spectra.pickle'%(folder_path),'wb'), protocol = pickle.HIGHEST_PROTOCOL)
        sum_spec = sfunc.sum_spec(spectra)
        max_pixel_spec = np.max(np.array(list(spectra.values())), axis = 0)
        np.save('%s/data/max_pixel_spec'%(folder_path),max_pixel_spec)
        np.save('%s/data/sum_spec'%(folder_path),sum_spec)
        
        
        del spectra
    else:                                                                       #for machines with low memory
        print('machine memory to small. creating single spectra')
        try: os.mkdir('%s/single_spectra/'%folder_path)
        except: pass
        for spx_file in ns.natsorted(glob('%s/*.txt'%folder_path)):
            spectrum = []
            start = 100
            with open(spx_file,'r', encoding="ISO-8859-1") as infile:
                for i, line in enumerate(infile):
                    if 'lin.' in line:
                        a1 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif 'abs.' in line:
                        a0 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif 'FWHM' in line:
                        try:
                            FWHM = float(line.split()[-1]) / 1000.0
                        except:
                            FWHM = float(line.split()[1]) / 1000.0
                    elif 'Fano' in line:
                        try:
                            Fano = float(line.split()[-1])
                        except:
                            Fano = float(line.split()[1])
                    elif 'Energy' in line or 'Energie' in line:
                        start = i
                    elif 'Life time:' in line or 'Life-Zeit' in line:
                        life_time = float(line.split()[1].strip()) / 1000.0
                    elif 'Channels:' in line or 'Kanäle:' in line:
                        channels = int(float(line.split()[1].strip()))
                    elif i > start:
                        line = line.split()
                        spectrum.append(int(line[1]))
            gating_time = 3e-6
            parameters_tmp = [a0, a1, Fano, FWHM, life_time, a0 + a1*(channels-1), gating_time]
        #    if sum(spectrum) > 200:
       #         worth_fit.append(True)
      #      else:
   #             worth_fit.append(False)
            spectrum = np.divide(spectrum, life_time)
            counts.append(sum(spectrum))
            np.save('%s/single_spectra/spectrum_%d'%(folder_path,file_nr),spectrum)  
            parameters.append(parameters_tmp)
            file_nr += 1 
            signal_progress.emit(file_nr)
        sum_spec = sum_from_single_files(folder_path)
        np.save('%s/data/sum_spec'%folder_path,sum_spec)
      #  np.save('%s/data/counts'%(folder_path),counts)
                    
 #   np.save('%s/data/fit_bool'%(folder_path),worth_fit)
    np.save('%s/data/counts'%(folder_path),counts)
    np.save('%s/data/parameters'%(folder_path),parameters)
    print('txt loadingtime - %f'%(t.time()-start_time))
    

def txt2energy(file_path):
    '''
    This function reads out the txt_files set up by 3D-measurements. It returns
    the energy:
        energy = spx_txt2energy(file_path)
    '''
    energy = []
    start = 100
    with open(file_path, 'r', encoding="ISO-8859-1") as infile:
        for i, line in enumerate(infile):
            if 'Energ' in line:
                start = i
            elif i > start:
                line = line.split()
                energy.append(float(line[0]))
    return energy

def txt2channels(file_path):
    '''
    This function reads out the txt_files set up by 3D-measurements. It returns
    the channels:
        channels = spx_txt2energy(file_path)
    '''
    channels = None
    with open(file_path, 'r', encoding="ISO-8859-1") as infile:
        for line in infile:
            if 'Kan' in line:
                channels = int(line.split()[1])
                break
            elif 'Channels' in line:
                channels = int(line.split()[1])
                break
    if channels == None: channels = 4096
    return channels

def txt2life_time(file_path):
    '''
    This function reads out the txt_files set up by 3D-measurements. It returns
    the channels:
        channels = spx_txt2energy(file_path)
    '''
    with open(file_path, 'r', encoding="ISO-8859-1") as infile:
        for line in infile:
            if 'Life time:' in line or 'Life-Zeit' in line:
                life_time = float(line.split()[1].strip()) / 1000.0
                break
    print( 'returned life-time', life_time)
    return life_time
    
def spx_tensor_position(file_path):
    '''
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = spx_tensor_position(file_path)
    '''
    position = file_path.split('(')[-1].replace(').txt','')
    position = position.split(',')
    return position

def sum_from_single_files(folder_path):
    assert os.path.exist('%s/single_spectra/*.npy'%folder_path)
    first_spec = True
    for single_spec_file in iglob('%s/single_spectra/*.npy'%folder_path):
        if first_spec == True:
            sum_spec = np.load(single_spec_file)
            first_spec = False
        else:
            sum_spec += np.load(single_spec_file)   
    return sum_spec
