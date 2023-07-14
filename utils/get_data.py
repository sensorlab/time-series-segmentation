import os, sys
import numpy as np
import pandas as pd
import pyedflib
import mne
import h5py

def get_versions_TSSB(path_to_desc="datasets/TSSB/"):
    """
    Gets the versions of TSSB

    Args:
        path_to_desc: a str that defines the path to the names of versions
    
    Returns:
       versions: a list of possible versions for TSSB
    """
    
    versions = os.listdir(path_to_desc + "TS/")
    versions = list(filter(lambda x: ".ipynb_checkpoints" not in x, versions))
    return versions
    
def get_TSSB(name_of_X, path_to_desc="datasets/TSSB/"):
    """
    Gets the X and mask of a chosen TSSB time series

    Args:
        name_of_X: name of the chosen TS we want and its mask
        path_to_desc: a str that defines the path to the names of versions
    
    Returns:
       _X: a 1D array containing the time steps of the chosen TSSB TS
       mask: a 1D array containing the mask for the chosen TSSB TS
    """
    
    data = []
    with open(path_to_desc+"desc.txt") as f:
        for line in f:
            row = line.strip().split(",")
            data.append(row)

    lists = {}
    for row in data:
        name = row[0]
        values = row[2:]
        lists[name] = values
    _X = np.loadtxt(path_to_desc + "TS/" + name_of_X)
    mask = np.zeros((len(_X)))
    for values in lists[name_of_X[:-4]]:
        mask[int(values)] = 1 
    return _X, mask

def get_Rutgers(DRConfig):
    """
    Gets the X and mask of a chosen TSSB time series

    Args:
        DRConfig: contains the path to the data of rutgers dataset
    
    Returns:
       _X: a 2D array containing the time steps of the rutgers dataset
       mask: a 2D array containing the mask for node classification for the rutgers dataset
       Y: a 1D array containing the mask for graph classification for the rutgers dataset
    """
    
    if DRConfig["len_type"] == "un/cut":
    
        df = pd.read_csv(DRConfig["path_main"])  
        del df['Unnamed: 0']
        df.index, df.columns = [range(df.index.size), range(df.columns.size)]
        length_rss = int((df.columns.stop-2)/2)
        
        X = df.loc[:,df.columns[:length_rss]].to_numpy() # x values for every sample
        #X = np.round_(X,1)
        Y = df[length_rss+1].to_numpy(dtype=np.uint8) # types of anomalies
        X_mask = df.loc[:,df.columns[length_rss+2:]].to_numpy() # binary location of anomalies
        # for i in range(len(Y)):
        #     X_mask[i][X_mask[i] == 1] = Y[i]
        
    # preparation for random graphs
    elif DRConfig["len_type"] == "random":
        dataset_rss = np.load(DRConfig["path_main"], allow_pickle=True)['arr_0']
        dataset_properties = np.load(DRConfig["path_properties"], allow_pickle=True)['arr_0']
        dataset_mask = np.load(DRConfig["path_mask"], allow_pickle=True)['arr_0']

        for i in range(len(dataset_properties)):
            dataset_properties[i,1] = int(dataset_properties[i,1])
        
        X = dataset_rss # x values for every sample
        X_mask = dataset_mask # binary location of anomalies
        Y = dataset_properties[:,2] # types of anomalies
        # Y_len = dataset_properties[:,0] # length of every sample
        
    return X, X_mask, Y

def get_versions_UTime(dataset_l,path_to_desc="datasets/U-Time/sleep-cassette/"):
    """
    Gets the versions of U-Time

    Args:
        path_to_desc: a str that defines the path to the names of versions
    
    Returns:
       versions: a list of possible versions for U-Time
    """
    if dataset_l == "cassette":
        versions = os.listdir("datasets/U-Time/sleep-cassette/")
    elif dataset_l == "dcsm":
        versions = os.listdir("datasets/U-Time/dcsm/")
    versions.sort()
    versions = list(filter(lambda x: ".ipynb_checkpoints" not in x, versions))
    return versions

def get_UTime_cassette(Container,version, key=1):
    # import warnings
    # warnings.filterwarnings("ignore")

    ##__________________________

    converte = {
        "Sleep stage W" : 0,
        "Sleep stage 1" : 1,
        "Sleep stage 2" : 2,
        "Sleep stage 3" : 3,
        "Sleep stage 4" : 3,
        "Sleep stage R" : 4,
        "Movement time" : 5,
        "Sleep stage ?" : 5
    }

    H_PSG = os.listdir("datasets/U-Time/sleep-cassette/" + version)
    print(version)
    H_PSG.sort()
    edf_file_path_PSG = "datasets/U-Time/sleep-cassette/" + version +"/"+ H_PSG[0]
    with pyedflib.EdfReader(edf_file_path_PSG) as edf_file_PSG:
        num_signals = edf_file_PSG.signals_in_file
        signal_labels = edf_file_PSG.getSignalLabels()

        signals = []
        for i in range(num_signals):
            signal = edf_file_PSG.readSignal(i)
            signals.append(signal)

        sample_rate = int(edf_file_PSG.getSignalHeaders()[key]["sample_rate"]*30)
    # edf_file_PSG.close()

    edf_file_path_H = "datasets/U-Time/sleep-cassette/" + version +"/"+ H_PSG[1]
    with pyedflib.EdfReader(edf_file_path_H) as edf_file_H:
        info=edf_file_H.readAnnotations()
    # edf_file_H.close()

    X_mask_markers = info[0]
    X_mask_Labels = info[2]
    
    for i in range(len(info[2])):
        X_mask_Labels[i] = converte[info[2][i]]
    X_mask_true = []

    for i in range(len(X_mask_markers)):
        X_mask_true=np.append(X_mask_true,[X_mask_Labels[i]]*int(info[1][i])*(100 if sample_rate == 3000 else 1)) #*100)

    # X_true = np.stack((signals[3], signals[4], signals[5]),axis=1).reshape(-1,30,3)
    # np.transpose(X_true[0])
    X_true = signals[key].reshape((-1, sample_rate)) #3000

    X_mask_true = X_mask_true.reshape((-1,sample_rate))
    Y_true=[]
    
    for i in range(len(X_true)):
        Y_true=np.append(Y_true,X_mask_true[i][1])
    Y_true=Y_true.astype("int")
    X_mask_true = X_mask_true.astype("int")
        
    return X_true, X_mask_true, Y_true, info

def get_UTime_dcsm(Container,version, key=1):
    converte = {
        "W" : 0,
        "N1" : 1,
        "N2" : 2,
        "N3" : 3,
        "REM" : 4,
    }

    H_PSG = os.listdir("datasets/U-Time/dcsm/" + version)
    H_PSG = list(filter(lambda x: ".ipynb_checkpoints" not in x, H_PSG))
    print(version)
    H_PSG.sort()
    
    ids_file_path_H = "datasets/U-Time/dcsm/" + version +"/"+ H_PSG[0]
    with open(ids_file_path_H , 'r') as f:
        lines = f.readlines()
    ids_array = [line.strip().split(',') for line in lines]
    info = [[int(num) if num.isdigit() else num for num in inner_lst] for inner_lst in ids_array]

    ids_file_path_PSG = "datasets/U-Time/dcsm/" + version +"/"+ H_PSG[1]
    with h5py.File(ids_file_path_PSG, "r") as f:
        channels_group = f["channels"]
        keys = list(channels_group.keys())
        print("num of keys: {}, current key: {}".format(len(keys), keys[key]))
        signal = channels_group[keys[key]][()]
        
    sample_rate = 30
    for i in range(len(info)):
        info[i][2] = converte[info[i][2]]
    
    X_mask_true = []
    for i in range(len(info)):
        X_mask_true=np.append(X_mask_true,[info[i][2]]*info[i][1]*256)
        
    X_true = signal.reshape((-1, sample_rate*256))
    X_mask_true = X_mask_true.reshape((-1,sample_rate*256))
    
    print(len(X_mask_true),len(X_true))
    Y_true=[]
    for i in range(len(X_true)):
        Y_true=np.append(Y_true,X_mask_true[i][1])
    Y_true=Y_true.astype("int")
    X_mask_true = X_mask_true.astype("int")
        
    return X_true, X_mask_true, Y_true, info