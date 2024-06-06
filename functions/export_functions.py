#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 08:36:24 2023

@author: frank
"""
### import packages needed
import h5py
from PIL import Image
from PyQt5 import QtWidgets

def h5_to_tiff():
    """
    This function creates tiff images from all arrays in provided h5 file

    """
    ### define path where h5 file is stored
    data_path = QtWidgets.QFileDialog().getOpenFileName(filter = '*.h5')[0]
    folder_path = "/".join(data_path.split("/")[:-1])
    file_name = data_path.split("/")[-1].replace(".h5","")
    ### define at how many rows the data should be splitted
    with h5py.File(data_path, "r") as f:
        for key in f:
            data = f[key][()].squeeze()
            file_path = folder_path+f"/{file_name}_{key}.tiff"
            image = Image.fromarray(data)
            # Save the array as a .tif file
            image.save(file_path, "tiff")
