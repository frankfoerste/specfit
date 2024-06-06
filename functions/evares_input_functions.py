# -*- coding: utf-8 -*-
"""
This module is made for input functions in the context of angle resolved measurements.
"""
###############################################################################

import os
import sys
from axp_tools import two_d
try:
    from axp_tools.xrf_library import detectors as det, optics as o
except:
    from axp_xrf_library import ccd_functions as det
    print('use axp_xrf_library')

try:
    from axp_tools.xrl_fupa import fupas
    fp = fupas()
except:
    try:    import xrlxraylib, xrlTUB
    except: 
        from axp_fupa import xrlxraylib, xrlTUB
        print('use axp_fupa')
    fp = xrlxraylib.xrlxraylib()

import numpy as np

def get_CCD_from_file(data_path,full_chip=False):
    """
    reads and extracts data from angular calibration files (plain txt) e.g. created
    with EVARES_ANG.py
    
    Parameters
    ----------
    data_path : str
        path to the data file
    """
    
    # load text from file
    keys_types = [['dx:',int,None],
                  ['n(y):',int,None],
                  ['pixel size:',float,None],
                  ['chip thickness:',float,None],
                  ['detection filters:',list,None],# detection filters calculated from window... and ambient...
                  ['x(0deg):',float,None],
                  ['y(0deg):',float,None],
                  ['distance:',float,None],
                  ['phi(CCD):',float,0],
                  ['omega(CCD):',float,0],
                  ['theta(sample):',float,0],
                  ['phi(sample):',float,0],
                  ['defect map:',str,None],
                  ###not directly used in det.ccd()
                  ['n(x):',int,None],
                  ['x0:',int,None],
                  ['window material:',str,None],
                  ['window thickness:',float,None],
                  ['window density:',str,None],
                  ['ambient compound:',str,None],
                  ['ambient distance:',float,None],
                  ['ambient density:',float,None]]
                  
    CCD_data = {}
    for k,t,default in keys_types:
        CCD_data[k] = default
    try:
        with open(data_path,'r', encoding="ISO-8859-1") as f:
            for line in f:
                line_key = line.split(':')[0].strip()+':'
                for k,t,default in keys_types:
                    if line_key == k:
                        try:    CCD_data[line_key] = t(line.split(':')[1].split('#')[0].strip())
                        except: pass
    except FileNotFoundError:
        print('IOError: File '+data_path+' not found.')
    else:
        if full_chip:
            x, dx = 0, CCD_data['n(x):']
        else:
            x, dx = CCD_data['x0:'], CCD_data['dx:']
        CCD_data['dx:'] = dx
        CCD_data['x0:'] = x
        CCD_data['x(0deg):'] -= x
        CCD_data['phi(CCD):'] += 180
        
        try:
            dm = load_defect_map(CCD_data['defect map:'])
        except FileNotFoundError:
            try:
                dm = load_defect_map(os.path.join(os.path.dirname(data_path),CCD_data['defect map:']))
            except FileNotFoundError:
                print('Defect map not found in \n{} nor in\n{}\n ... no defect map loaded'.format(CCD_data['defect map:'],os.path.join(os.path.dirname(data_path),CCD_data['defect map:'])))
                dm = []
                dm = []
        try:
            dm = dm[x:x+dx][:]
        except:
            dm = []
            dm = []
        CCD_data['defect map:'] = dm
            
        detection_filters = []
        # for window like Be window in front of the chip
        win_mat = str(CCD_data['window material:'])
        try:    win_thick = float(CCD_data['window thickness:'])
        except:
            print('NOTE: no window thickness given! Set to zero!')
            win_thick = ''
        win_dens = str(CCD_data['window density:'])
        if not (win_mat == '' or win_thick == '' or win_dens == ''):
            if win_dens.lower() == 'fp':
                win_dens = fp.density(win_mat)
            else:
                win_dens = float(win_dens)
            detection_filters.append(o.absorption_filter(win_mat,win_thick*1e-4,win_dens))
         
        # for air paths of excitation to CCD
        try:
            air_mat = str(CCD_data['ambient compound:'])
            air_thick = float(CCD_data['ambient distance:'])
            air_dens = float(CCD_data['ambient density:'])
        except:
            air_mat, air_thick, air_dens = '', '', ''
        if not (air_mat == '' or air_thick == '' or air_dens == ''):
            detection_filters.append(o.absorption_filter(air_mat,air_thick,air_dens))
        CCD_data['detection filters:'] = detection_filters
    
        ccd_parameters = [CCD_data[key] for [key,tmp1,tmp2] in keys_types[:13]]+[False]
    #        print(ccd_parameters)
        ccd = det.ccd(*ccd_parameters)
        return ccd


def load_defect_map(data_path):
    defect_map = two_d.load_image_file(data_path)
#    print(np.shape(defect_map))
    if len(np.shape(defect_map)) == 3:
        if np.shape(defect_map)[2] == 4 or np.shape(defect_map)[2] == 3:    # image is rgb
#                    try:
            defect_map = defect_map[:,:,2]     # blau muss null sein
#                    except:
#                        self.defect_map = defect_map[:,:,2]     # ohne Alpha Kanal
#                        tmp_map = np.array(self.defect_map)
#                        self.defect_map[tmp_map==0]=255
#                        self.defect_map[tmp_map==255]=0
        else:
            defect_map = defect_map[:,:,np.shape(defect_map)[3]-1]
    elif len(np.shape(defect_map)) == 2:
        defect_map = defect_map
    else:
        defect_map = None
    return defect_map
