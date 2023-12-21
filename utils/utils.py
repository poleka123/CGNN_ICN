import os.path
import csv
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dtaidistance import dtw
from fastdtw import fastdtw

files = {
    'pems03': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pems07': ['PEMS07/PEMS07.npz', 'PEMS07/PEMS07.csv'],
    'pems08': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv'],
    'randomuniformity': ['RandomUniformity/V_flow_50.npz', 'RandomUniformity/V_flow_50.csv'],
    'smallscaleaggregation': ['SmallScaleAggregation/V_flow_50.npz', 'SmallScaleAggregation/V_flow_50.csv']
}


def Z_Score(matrix):
    mean, std = np.mean(matrix), np.std(matrix)
    return (matrix - mean) / (std+0.001), mean, std


def Un_Z_Score(matrix, mean, std):
    return (matrix * std) + mean


def get_normalized_adj(W_nodes):
    W_nodes = W_nodes + np.diag(np.ones(W_nodes.shape[0], dtype=np.float32))
    D = np.array(np.sum(W_nodes, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    W_nodes = np.multiply(np.multiply(diag.reshape((-1, 1)), W_nodes),
                         diag.reshape((1, -1)))
    return W_nodes

def generate_adjmatrix(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data_set/"
    data = pd.read_csv(filepath+file[1], header=None).head(8640).to_numpy(np.float32)
    num_node = data.shape[1]
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    data, mean, std = Z_Score(data)
    # 生成dtw的邻接矩阵
    if not os.path.exists(f'data_set/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data_set/{filename}_dtw_distance.npy', dtw_distance)
    dist_matrix = np.load(f'data_set/{filename}_dtw_distance.npy')
    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # use continuous spatial matrix
    # if not os.path.exists(f'data/{filename}_spatial_distance.npy'):
    #     with open(filepath + file[1], 'r') as fp:
    #         dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
    #         file = csv.reader(fp)
    #         for line in file:
    #             break
    #         for line in file:
    #             start = int(line[0])
    #             end = int(line[1])
    #             dist_matrix[start][end] = float(line[2])
    #             dist_matrix[end][start] = float(line[2])
    #         np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)
    #
    # dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    # # normalization
    # std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    # mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    # dist_matrix = (dist_matrix - mean) / std
    # sigma = args.sigma2
    # sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    # sp_matrix[sp_matrix < args.thres2] = 0

    return dtw_matrix




def generate_asist_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features))


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]
    # indices_time = [(i, i+(num_timesteps_input + num_timesteps_output)) for i
    #                 in range(X.sahpe[2]-(num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    features_time, target_time = [], []
    ftmask = np.ones((len(indices), X.shape[0], num_timesteps_input))
    tmask = np.ones((len(indices), X.shape[0], num_timesteps_output))
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        features_timewindow = np.arange(max(0, i), i+num_timesteps_input)
        features_time.append(features_timewindow)

        target.append(X[:, 0, i + num_timesteps_input: j])
        target_timewindow = np.arange(max(0, i+num_timesteps_input), j)
        target_time.append(target_timewindow)
    features_time = np.array(features_time)
    features_time = np.repeat(features_time, X.shape[0], axis=0)
    features_time = features_time.reshape((len(indices), X.shape[0], -1))
    target_time = np.array(target_time)
    target_time = np.repeat(target_time, X.shape[0], axis=0)
    target_time = target_time.reshape((len(indices), X.shape[0], -1))
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target)), \
           torch.tensor(features_time, dtype=torch.float32), torch.tensor(target_time, dtype=torch.float32), \
           torch.tensor(ftmask, dtype=torch.float32), torch.tensor(tmask, dtype=torch.float32)


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def FFT(current_series, current_mean):
    '''
    :param current_series: 一个节点的时序数据
    :param current_mean: 时序数据均值
    :return:
    '''
    # since some series might begin with many 0s, we get <nonzero_idx>
    # in order to get started with non-zero part of series
    if np.sum(current_series == -current_mean)>0:
        try:
            nonzero_idx = np.where(current_series>0)[0][0]
        except:
            nonzero_idx = 0
    else:
        nonzero_idx = 0
    series = current_series[nonzero_idx:]
    # real valid data for FFT
    N = len(series)
    t = np.arange(N)
    dt = 1
    #transform
    fft = np.fft.fft(series)
    fftshift = np.fft.fftshift(fft)
    mo  = abs(fftshift)/N
    phase = np.angle(fftshift)
    fre = np.fft.fftshift(np.fft.fftfreq(d=dt, n=N))
    #series will shift 2*pi*f*nozeros_idx
    apfs = [(mo[i], phase[i]-2*np.pi*fre[i]*nonzero_idx, fre[i]) for i in range(len(mo))]
    return apfs

def get_bestKmask_per_series(K_apfs, series, fft_cutpoint, J):
    # using dtw for matching and selection
    series =series.astype(np.float64)
    criterion = dtw.distance_fast
    bestKsubset = []
    fftinput = np.arange(len(series))
    recover_results = np.zeros(len(series))
    bestdtw = criterion(series[fft_cutpoint:], recover_results[fft_cutpoint:])
    for i, item in enumerate(K_apfs):
        a, p, f = item
        current_fft = a*np.cos(2*np.pi*f*fftinput+p)
        current_dtw = criterion(series[fft_cutpoint:], recover_results[fft_cutpoint:]+current_fft[fft_cutpoint:])
        if current_dtw < bestdtw:
            bestKsubset.append(i)
            bestdtw = current_dtw
            recover_results = recover_results + current_fft
            if len(bestKsubset) >= J:
                break
    # transform bestKsubset into one-hot mask
    selected_results = np.zeros(len(K_apfs))
    for idx in bestKsubset:
        selected_results[idx] = 1
    return selected_results


def warm_PM_paramters_perK(training_values, fft_cutpoint, K=100, J=10):
    # compute for initialization of periodical module
    all_f, all_p, all_a = [], [], []
    all_mean = []
    all_Kmask = []
    for i, current_series in tqdm(enumerate(training_values)):
        all_mean.append(np.mean(current_series))
        current_mean = np.mean(current_series)
        series = current_series - all_mean[-1]
        fftnet_series = series[:fft_cutpoint]
        apfs = FFT(fftnet_series, current_mean)
        sorted_apfs = sorted(apfs, key=lambda x: x[0], reverse=True)
        K_apfs = []
        # remove the same source infomation
        for item in sorted_apfs:
            a, p, f = item
            if len(K_apfs) == 0:
                K_apfs.append(item)
                continue
            if len(K_apfs) == K:
                break
            # since cos(x) = cos(-x), we want to simplify the candidate cos functions of equivalent p and f
            if round(p + K_apfs[-1][1], 4) == 0.0 and round(f + K_apfs[-1][2], 4) == 0.0:
                K_apfs[-1] = (K_apfs[-1][0]+a, K_apfs[-1][1], K_apfs[-1][2])
            else:
                K_apfs.append(item)
        # keep all series have K dimensions source cos
        if len(K_apfs) < K:
            for _ in range(K-len(K_apfs)):
                K_apfs.append((0, 0, 0))
        # get bestK for per series
        Kmask = get_bestKmask_per_series(K_apfs, series, fft_cutpoint, J)
        K_a = [item[0] for item in K_apfs]
        K_p = [item[1] for item in K_apfs]
        K_f = [item[2] for item in K_apfs]
        all_f.append(K_f)
        all_a.append(K_a)
        all_p.append(K_p)
        all_Kmask.append(Kmask)
    return np.array(all_a), np.array(all_p), np.array(all_f), np.array(all_mean), np.array(all_Kmask)


def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))


def SMAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs((v_ - v) / ((torch.abs(v) + torch.abs(v_)) / 2 + 1e-5)))

