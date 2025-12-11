from sklearn import preprocessing
import numpy as np
import scipy.io as scio
from collections import Counter

def swap_samples(X_list, unaligned_rate=1):

    swapped_X_list = []
    n, d = X_list[0].shape
    num_to_swap = int(unaligned_rate * n)
    swap_indices = np.random.choice(n, num_to_swap, replace=False)
    i = 0
    for X in X_list:
        if i==0:
            swapped_X_list.append(X)
            i = i+1
            continue
        shuffled_indices = np.random.permutation(swap_indices)
        X_swapped = X.copy()
        X_swapped[swap_indices, :] = X[shuffled_indices, :]

        swapped_X_list.append(X_swapped)

    return swapped_X_list

def preprocess_data(dataset, unaligned_rate=1, base=0):
    if dataset=="UCI6":
        print("UCI digits handwritten view6 shuffle")
        min_max_scaler = preprocessing.MinMaxScaler()
        data = scio.loadmat('datasets/handwritten.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][0]) # n*d
        X2 = min_max_scaler.fit_transform(data['X'][0][1])
        X3 = min_max_scaler.fit_transform(data['X'][0][2])
        X4 = min_max_scaler.fit_transform(data['X'][0][3])
        X5 = min_max_scaler.fit_transform(data['X'][0][4])
        X6 = min_max_scaler.fit_transform(data['X'][0][5])
        Y = data['Y'][:,0]
        X = [X1, X2, X3, X4, X5, X6]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset == 'Caltech101-7':
        print("Caltech7 shuffle")
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('datasets/Caltech101-7.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][0])
        X2 = min_max_scaler.fit_transform(data['X'][0][1])
        X3 = min_max_scaler.fit_transform(data['X'][0][2])
        X4 = min_max_scaler.fit_transform(data['X'][0][3])
        X5 = min_max_scaler.fit_transform(data['X'][0][4])
        X6 = min_max_scaler.fit_transform(data['X'][0][5])
        Y = data['Y'][:,0] - 1
        X = [X1, X2, X3, X4, X5, X6]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset=="Scene15":
        print("Scene15 view3 shuffle")
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)) #
        data = scio.loadmat('datasets/Scene15.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][1]) # n*d
        X2 = min_max_scaler.fit_transform(data['X'][0][2])
        X3 = min_max_scaler.fit_transform(data['X'][0][0])
        Y = data['Y'][:,0] - 1
        X = [X1, X2, X3]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset == 'Office31':
        print("Office31 view 3 shuffle")
        data = scio.loadmat('datasets/Office31.mat')
        X1 = data['X'][0][0] # n*d
        X2 = data['X'][0][1]
        X3 = data['X'][0][2]
        X = [X1, X2, X3]
        X[0], X[base] = X[base], X[0]
        print("X1: ",X1.shape)
        Y = data['Y'][:,0]
        print("Y: ", sorted(Counter(Y)))
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset == 'cub_googlenet_doc2vec_c10':
        print("CUB shuffle")
        min_max_scaler = preprocessing.MinMaxScaler()
        data = scio.loadmat('datasets/cub_googlenet_doc2vec_c10.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][0])  # n*d
        X2 = min_max_scaler.fit_transform(data['X'][0][1])
        Y = data['gt'][:,0] - 1
        X = [X1, X2]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset == "LandUse-21":
        print("LandUse21 view3 shuffle")
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('datasets/LandUse-21.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][1])  # n*d
        X2 = min_max_scaler.fit_transform(data['X'][0][2])
        X3 = min_max_scaler.fit_transform(data['X'][0][0])
        Y = data['Y'][:, 0] - 1
        X = [X1, X2, X3]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return X_swapped, Y
    if dataset == 'Hdigit':
        print("Hdigits shuffle")
        min_max_scaler = preprocessing.MinMaxScaler()
        data = scio.loadmat('datasets/Hdigit.mat')
        X1 = min_max_scaler.fit_transform(data['data'][0][0].T)  # n*d
        X2 = min_max_scaler.fit_transform(data['data'][0][1].T)
        Y = data['truelabel'][0][0][0] -1
        X = [X1, X2]
        X[0], X[base] = X[base], X[0]
        X_swapped = swap_samples(X, unaligned_rate)
        return  X_swapped, Y

