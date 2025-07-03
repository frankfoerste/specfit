import h5py
import numpy as np
from PyQt6 import QtWidgets

def split_h5():
    """
    This function splits the data.h5 created when loading the measurement into
    single h5 files which which consist of 

    Returns
    -------
    None.

    """
    # define path where h5 file is stored
    data_path = QtWidgets.QFileDialog().getOpenFileName(filter = '*.h5')[0]
    # define at how many rows the data should be splitted
    rows_split = 10
    # now open the h5 file
    with h5py.File(data_path, 'r') as f:
        # read out the dimension of the measurement
        dimensions = f['position dimension'][()]
        # create a list of splits from the row_split and first dimension of the data
        if dimensions[0]%rows_split != 0:
            list_of_splits = [rows_split]*(dimensions[0]//rows_split)+[dimensions[0]%rows_split]
        else:
            list_of_splits = [rows_split]*(dimensions[0]//rows_split)
        # now iterate over the list and split the data accordingly
        for i, split in enumerate(list_of_splits):
            # calculate start and end index, since the data is stored as one long
            # list (e.g. np.prod(dimensions)), the index has to be calculated 
            # accordingly
            start_idx = np.sum(list_of_splits[:i] * np.prod(dimensions[1:]), dtype = int)
            end_idx = np.sum(list_of_splits[:i+1] * np.prod(dimensions[1:]), dtype = int)
            # create a split name for the splitted data
            split_name = data_path.replace('.h5', f'_{i}.h5')
            split_dimensions = np.array([split] + list(dimensions[1:]))
            split_positions = f['positions'][start_idx:end_idx]
            split_positions[:,0] -= (i*rows_split)
            split_tensor = f['tensor positions'][start_idx:end_idx]
            split_tensor[:,0] -= (i*rows_split)
            with h5py.File(split_name, 'w') as split_file:
                    # Create the datasets
                    split_file.create_dataset('counts', data = f['counts'][start_idx:end_idx])
                    split_file.create_dataset('max pixel spec', data = f['max pixel spec'])
                    if f['parameters'].ndim == 1:
                        split_file.create_dataset('parameters', data = f['parameters'])
                    else:
                        split_file.create_dataset('parameters', data = f['parameters'][start_idx:end_idx])
                    split_file.create_dataset('position dimension', data = split_dimensions)
                    split_file.create_dataset('positions', data = split_positions)
                    split_file.create_dataset('spectra', data = f['spectra'][start_idx:end_idx])
                    split_file.create_dataset('sum spec', data = f['sum spec'])
                    split_file.create_dataset('tensor positions', data = split_tensor)
    print('# splitting done #')
