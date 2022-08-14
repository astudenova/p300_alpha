import numpy as np
import json
import os.path as op


def save_json_from_numpy(filename, folder, var):
    """
    Save numpy array as json file

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :param var: variable to save
    """

    with open(op.join(folder, filename), "w") as f:
        json.dump(var.tolist(), f)


def load_json_to_numpy(filename, folder):
    """
    Load json file and convert to a numpy array

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :return: var - variable
    """

    with open(op.join(folder, filename), "r") as f:
        saved_data = json.load(f)

    var = np.array(saved_data)
    return var


def save_json(filename, folder, var):
    """
    Save list as json file

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :param var: variable to save
    """

    with open(op.join(folder, filename), "w") as f:
        json.dump(var, f)


def load_json(filename, folder):
    """
    Load list from json file

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :return: var - variable
    """

    with open(op.join(folder, filename), "r") as f:
        saved_data = json.load(f)

    var = saved_data
    return var


def save_pickle(filename, folder, var):
    """
    Save array as pickle file
    by Mina Jamshidi Idaji (minajamshidi91@gmail.com)

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :param var: variable to save
    """

    import pickle
    with open(op.join(folder, filename), "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(filename, folder):
    """
    Load array from pickle file
    by Mina Jamshidi Idaji (minajamshidi91@gmail.com)

    :param str filename: the name of the saved file
    :param str folder: the name of the folder to save
    :return: var - variable
    """

    import pickle
    with open(op.join(folder, filename), "rb") as input_file:
        var = pickle.load(input_file)
    return var


def eta_squared(aov):
    """
    Compute effect size based on output of a model

    :param aov: model (anova, lsq, ets.)
    :return: aov - model
    """

    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def list_from_many(ids, dir_read, file_ext, type_read='json'):
    """
    Load several files containing participants data and convert into array

    :param numpy.ndarray ids: ids of participants for whom loading is needed
    :param str dir_read: directory with files
    :param str file_ext: the appendix of the file, for instance, for subj000_erp, _erp is an appendix
    :param str type_read: either 'json' or 'pickle', type of file the derivatives are stored in
    :returns:
        output (numpy.ndarray) - data from all participants in ids
        not_found (list) - list of subject ids for which data were not found
    """
    output = []
    not_found = []
    for i_subj, subj in enumerate(ids):
        try:
            if type_read == 'json':
                output.append(load_json_to_numpy(subj + file_ext, dir_read))
            elif type_read == 'pickle':
                output.append(load_pickle(subj + file_ext, dir_read))
            else:
                print('please specify correct type')
        except ValueError:
            print(subj + ' is not found.')
            not_found.append(subj)

    return np.array(output), not_found


def scaler_transform(data, scaler='standard'):
    """
    Scaler data according to chosen scaler

    :param numpy.ndarray data: data to apply scaler to, should be samples x features (dimensions - 2D)
    :param str scaler: either 'standard' or 'minmax', type of scaler, for 'minmax' scaler range is set to [-1,1]
    :returns: data_scale (numpy.ndarray, 2D) - data scaler-transformed
    """

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # if data is one dimensional vector (1 feature), add a dimension
    if len(data.shape) == 1:
        reshape_flag = 1
        data = data.reshape((-1, 1))
    else:
        reshape_flag = 0

    if scaler == 'standard':
        scaler = StandardScaler()
        data_scale = scaler.fit_transform(data)
        if reshape_flag:
            data_scale = data_scale.reshape((-1))
    elif scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scale = scaler.fit_transform(data)
        if reshape_flag:
            data_scale = data_scale.reshape((-1))
    else:
        raise ValueError('Scaler can be \'standard\' or \'minmax\'')

    return data_scale


def scale_to_zero_one(data):
    """
    Scales data in [0,1] range

    :param numpy.ndarray data: data to apply scaler to, should be samples x 1 (dimnesions - 1D)
    :returns: data_scale (numpy.ndarray, 1D) - data transformed
    """
    data_scale = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_scale
