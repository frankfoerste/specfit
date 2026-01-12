import h5py
from PIL import Image
from PyQt6 import QtWidgets
from pathlib import Path

lines = ["K", "Ka", "Kb", "L", "L3", "L2", "L1", "M"]

def h5_to_tiff():
    """
    This function creates tiff images from all arrays in provided h5 file
    """
    # define path where h5 file is stored
    data_path = Path(QtWidgets.QFileDialog().getOpenFileName(filter = "*.h5")[0])
    folder_path = data_path.parent
    (folder_path/"tiff").mkdir(parents=True, exist_ok=True)
    file_name = data_path.stem
    # define at how many rows the data should be splitted
    with h5py.File(data_path, "r") as f:
        for meas in f:
            if isinstance(f[meas], h5py.Group):
                for line in f[meas]:
                    data = f[meas][line][()].squeeze()
                    file_path = folder_path/"tiff"/f"{meas.split('.')[0]}_{line}.tiff"
                    image = Image.fromarray(data)
                    # Save the array as a .tif file
                    image.save(file_path, "tiff")
            else:
                data = f[meas][()].squeeze()
                file_path = folder_path/"tiff"/f"{file_name}_{meas}.tiff"
                image = Image.fromarray(data)
                # Save the array as a .tif file
                image.save(file_path, "tiff")
