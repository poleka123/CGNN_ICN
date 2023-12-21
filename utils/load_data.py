import os

import numpy as np
import torch
import random


def seq2instance(data, P, F, H):
    num_step, dims = data.shape
    num_sample = num_step - H - P - F + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape = (num_sample, F, dims))
    z = np.zeros(shape = (num_sample, H, dims))
    for i in range(num_sample):
        z[i] = data[i : i + H]
        x[i] = data[i + H : i + H + P]
        y[i] = data[i + H + P : i + H + P + F]
    return z, x, y



def loadData(args):
    data_path = './data_set/SmallScaleAggregation/'+args.dataset+'.npz'
    SE_FILE = './data_set/SmallScaleAggregation/'+'SE('+args.dataset+')'+'.txt'
    # 综合数据
    Traffic_flow = np.load(data_path)['data']
    time, num_of_nodes, features = Traffic_flow[0], Traffic_flow[1],  Traffic_flow[2]
    Traffic = Traffic_flow[:, :, 0]
    num_step = Traffic.shape[0]
    # normalization
    mean, std = np.mean(Traffic), np.std(Traffic)
    Traffic = (Traffic - mean) / std

    # train/val/test
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps:]
    test = Traffic[train_steps:]
    # 后面修改
    # val = Traffic[train_steps : train_steps + val_steps]
    # test = Traffic[-test_steps :]

    # Z, X, Y  represents past, present and future tarffic data respectively
    trainZ, trainX, trainY = seq2instance(train, args.P, args.F, args.H)
    valZ, valX, valY = seq2instance(val, args.P, args.F, args.H)
    testZ, testX, testY = seq2instance(test, args.P, args.F, args.H)

    # normalization

    # trainX = (trainX - mean) / std
    # valX = (valX - mean) / std
    # testX = (testX - mean) / std

    # spatial embedding
    f = open(SE_FILE, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]
    # print("SE Shape is: ", SE.shape)

    #temporal embedding
    if args.dataset=='pems04' or 'RandomUniformity' or 'SmallScaleAggregation':
        week = np.arange(7)
    if args.dataset=='pems08':
        week = np.array([4, 5, 6, 0, 1, 2, 3])
    if args.dataset=='pems03':
        week = np.array([5, 6, 0, 1, 2, 3, 4])
    if args.dataset=='pems07':
        week = np.arange(7)
    l = []
    for i in range(num_step//288):
        tmp = week[i%7]
        b = tmp.repeat(288)
        l.append(b)
    dayofweek = np.stack(l, axis=0).reshape(-1, 1)
    #Time.hour
    a = np.arange(24)
    ll = []
    for day in range(num_step//288):
        l = []
        for i in range(24):
            b = a[i].repeat(12)
            l.append(b)
        l = np.stack(l,axis=0).reshape(-1,1)
        ll.append(l)
    Time_hour = np.stack(ll,axis=0).reshape(-1,1)
    # Time.minutes
    a = np.arange(0,60,5).reshape(-1,1)
    b = a.repeat(24*num_step//288,1)
    Time_minute = b.T.reshape(-1,1)
    # Time.seconds
    Time_second = np.zeros([num_step,1])
    timeofday = (Time_hour * 3600 + Time_minute * 60 + Time_second) \
                // 300
    timeofday = np.reshape(timeofday, newshape = (-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis = -1)

    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps:]
    test = Time[train_steps:]
    # val = Time[train_steps : train_steps + val_steps]
    # test = Time[-test_steps :]
    trainTE = seq2instance(train, args.P, args.F, args.H)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.F, args.H)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.F, args.H)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    data_set = {
        'trainZ': trainZ,
        'trainX': trainX,
        'trainTE': trainTE,
        'trainY': trainY,
        'valZ': valZ,
        'valX': valX,
        'valTE': valTE,
        'valY': valY,
        'testZ': testZ,
        'testX': testX,
        'testTE': testTE,
        'testY': testY,
        'SE': SE,
        'data_mean': mean,
        'data_std': std
    }
    return data_set