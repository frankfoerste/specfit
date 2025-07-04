# -*- coding: utf-8 -*-
"""
Specfit_GUI_functions
"""
import numpy as np
import pickle


def norm2sec(spectrum, time):
    '''
    This function normalizes the given spectrum to seconds based on life or real time.
    
    Parameters
    ----------
    spectrum : list
    time : float
    
    Returns
    -------
    list of the normalized spectrum
    '''
    spectrum = [float(i) for i in spectrum]
    spectrum[:] = [i / time for i in spectrum]
    return spectrum


def sum_spec(spectrum):
    '''This function calculates the sum spectrum of all given spectra.'''  
    values = np.asarray(list(spectrum.values()))
    sum_spec = values.sum(axis=0)
    return sum_spec


def get_file_and_folder_name(path):
    'This function returns the folderpath and the filename'
    file_name_length = len(path.split('/')[-1:][0])
    folder_path = path[:-file_name_length]
    file_name = path[-file_name_length:]
    return folder_path, file_name


def save_dict_pickle(save_path, dictionary):
    ''' This function saves the dictionary to the given path using pickle'''
    with open(save_path, 'wb') as saveto:
        pickle.dump(dictionary, saveto, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def open_dict_pickle(load_path):
    ''' This function loads a dictionary from a pickle-file'''
    with open(load_path, 'rb') as loadfrom:
        return pickle.load(loadfrom)
