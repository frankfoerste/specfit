# -*- coding: utf-8 -*-
"""
This programm will extract the channels of the given .spx-file and save them
to a .txt-file with the same name.
"""
###############################################################################
import numpy as np
import os
import h5py
import itertools
import dask.array as da

###############################################################################
        
        
def hdf2spec_para(file_path):
    ''' 
    This function reads out the spectrum of a .hdf5-file and reads out the
    detector parameters given in the .hdf5-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .hdf5-file.
    
    Returns
    -------
    array containing the spectrum
    array containing the detector parameters [a0, a1, FANO, FWHM]
    '''
    
    file_type = file_path.split('.')[-1]
    save_path = '/'.join(file_path.split('/')[:-1])
    try: os.mkdir('%s/data/'%save_path)
    except: pass

    with h5py.File(file_path, 'r') as hdf5_file:
    
        if file_type == 'h5':
            spectra = hdf5_file['Raw'][()]
            shape = spectra.shape
            # print(shape)
            if len(shape) == 3:
                spectra = np.reshape(spectra, (shape[0]*shape[1], shape[2]))
            else:
                spectra = np.reshape(spectra, (shape[0]*shape[1]*shape[2], shape[3]))
            sum_spec = hdf5_file['Spectrum'][()]
            counts = np.sum(spectra, axis=-1)
            max_pixel_spec = np.sum(spectra, axis=0)
            parameter = hdf5_file['Header'][()]
            h,m,s = parameter[0][2].decode('ascii').split(':')
            
            a0 = parameter[0][12]/1000
            a1 = parameter[0][13]/1000
            Fano = parameter[0][15]
            FWHM = 0.14
            life_time = int(h)*3600 + int(m)*60 + int(s)
            real_time = int(h)*3600 + int(m)*60 + int(s)
            spectra = np.divide(spectra, life_time) ### norm the measurement to life_time
            sum_spec = np.divide(sum_spec, life_time)
            channels = parameter[0][19]
            gating_time = 3e-6
            parameters = np.array([a0, a1, Fano, FWHM, np.mean(life_time), a0 + a1 * (channels-1), gating_time, real_time])
            print('parameters', parameters.shape)
            with h5py.File(f'{save_path}/data/data.h5', 'w') as tofile:
                # print('### saving max pixel spec to data.h5')
                # tofile.create_dataset('max pixel spec', data = max_pixel_spec)
                print('### saving sum spec to data.h5')
                tofile.create_dataset('sum spec', data = sum_spec,
                                      compression="gzip")
                print('### saving parameters to data.h5')
                tofile.create_dataset('parameters', data=parameters,
                                      compression="gzip")
            print('### saving counts to data.h5')
            da.to_hdf5(f'{save_path}/data/data.h5', {'counts': counts},
                       compression="gzip")  
            print('### saving spectra to data.h5')
            da.to_hdf5(f'{save_path}/data/data.h5', {'spectra': spectra},
                       compression="gzip")  
        
            
        elif file_type == 'hdf5':
            spectra = da.from_array(hdf5_file['SDD/measurement/detector'])
            print('### read out spectra')
            sum_spec = np.squeeze(hdf5_file['SDD/sum spectra/sumS'], axis = 1)
            print('### read out sum spectra')
            counts = np.sum(spectra, axis=-1)
            max_pixel_spec = hdf5_file['SDD/maximum pixel spectrum/mps'][()]
            a0 = -0.235825
            a1 = 0.002515
            Fano = 0.112
            FWHM = 0.045
            life_time = np.sum(spectra[:,:125], axis=-1)
            print('### calculated life time')
            real_time = 1
            channels = int(len(spectra[0]))
            gating_time = 1e-6
            ### now norm to the life_time. For the QUAD Detector the gating time 
            ### is set to 4*1e6
            print('### normalize spectra')
            spectra = spectra/life_time[:,np.newaxis]/4/gating_time
            print('### normalized spectra')
            parameters = np.array([a0, a1, Fano, FWHM, np.mean(life_time), a0 + a1 * (channels-1), gating_time, real_time])
            print('parameters', parameters.shape)
            with h5py.File(f'{save_path}/data/data.h5', 'w') as tofile:
                print('### saving max pixel spec to data.h5')
                tofile.create_dataset('max pixel spec', data = max_pixel_spec,
                           compression="gzip")
                print('### saving sum spec to data.h5')
                tofile.create_dataset('sum spec', data = sum_spec,
                           compression="gzip")
                print('### saving parameters to data.h5')
                tofile.create_dataset('parameters', data = parameters,
                           compression="gzip")
            print('### saving counts to data.h5')
            da.to_hdf5(f'{save_path}/data/data.h5', {'counts': counts},
                       compression="gzip")  
            print('### saving spectra to data.h5')
            da.to_hdf5(f'{save_path}/data/data.h5', {'spectra': spectra},
                       compression="gzip")  

    return spectra, parameters, sum_spec, channels
      
   
def hdf_tensor_positions(file_path):
    '''
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = spx_tensor_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    '''
    file_type = file_path.split('.')[-1]
    save_path = '/'.join(file_path.split('/')[:-1])
    hdf5_file = h5py.File(file_path, 'r')
    if file_type == 'hdf5':
        line_breaks = hdf5_file['SDD/scan index log/line breaks'][()]
        positions = [line_breaks[-2,2]+1, line_breaks[1,1], 1]
        tensor_positions = np.asarray(list(itertools.product(range(positions[0]),
                                                     range(positions[1]))),
                                dtype = np.uint32)
        tensor_positions = np.hstack((tensor_positions, 
                                      np.zeros((len(tensor_positions),1), dtype = np.uint32)))
    elif file_type == 'h5':
        shape = hdf5_file['Raw'].shape
        if len(shape) == 3:
            positions = [shape[0], shape[1], 1]
            tensor_positions = np.asarray(list(itertools.product(range(positions[0]),
                                                                 range(positions[1]))),
                                          dtype = np.uint32)
            tensor_positions = np.hstack((tensor_positions, 
                                          np.zeros((len(tensor_positions),1),
                                                   dtype = np.uint32)))
        else:
            positions = list(shape[:-1])
            tensor_positions = np.asarray(list(itertools.product(range(positions[0]),
                                                                 range(positions[1]),
                                                                 range(positions[2]))),
                                          dtype = np.uint32)
    with h5py.File('%s/data/data.h5'%save_path, 'a') as tofile:
        tofile.create_dataset('tensor positions', data = tensor_positions,
                   compression="gzip")
        tofile.create_dataset('positions', data = tensor_positions,
                   compression="gzip")
        tofile.create_dataset('position dimension', data = positions,
                   compression="gzip")
        
    return positions, tensor_positions
