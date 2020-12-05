# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:14:42 2019

@author: 彭钊

将一维时序数据小波变换成二维时频数据的自编额外扩展包

所需额外扩展包：obspy、numpy、h5py、matplotlib、pywt

"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import pywt

"""
以下为训练事件窗口数据参数，该参数需要根据实际情况进行调整，本程序设定默认参数
"""
# 数据采样率，本处为原始采样率100Hz
# sampling_rate = 100

# 每个窗口的时间序列，本处的窗长为25.01s
# time = 25.01
# t = np.arange(0, time, 1.0 / sampling_rate)



"""
以下为小波变换需要输入的参数，这些参数决定小波变换的效果和输出，本程序设定默认参数
"""
# 小波变换所采用的的小波
# 此为小波变换的关键参数之一，本程序采用cmor小波，参数分别取3,3
# wavename = 'cmor3-3'

# 小波变换后频率域的取值，小波变化后频率域的分布为0-（totalscal-1）
# 小波变换频域的取值范围主要由地震事件的性质和仪器的相应频谱决定
# 程序应用时，该参数需要根据实际情况做调整
# 本程序默认取46，小波变换后的频域为0-45Hz
# totalscal = 46


### 一维时序数据进行小波时频变换
### 输入为一维时频数据data:（m,）
### 输出为列表[cwtmatr, frequencies]，cwtmatr为二维时频矩阵（（totalscal-1）,m）,frequencies为频域（（totalscal-1）,）
def cwt4series(data,wavename = 'cmor3-3',totalscal = 46,sampling_rate = 100):
    # data为一维时序数据
    
    # 需要确保data为（len，）格式
    data = data.reshape((-1,))
    
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    
    return [cwtmatr, frequencies]
    
    
### 对变换后的二维时频矩阵进行下采样，减小其大小，规整其形状，便于机器学习模型进行训练
### 输入为cwt4series函数的输出二维时频矩阵cwtmatr，下采样因子（即缩小倍数）频域为n，时域为m
### 下采样因子，频域默认为 n = 1 ，时域默认为 m = 10
### 输出为下采样后的二维时频矩阵cwtmatr_s
def subs_matr(cwtmatr,n = 1,m = 10):
    
    cwtmatr_s = cwtmatr[::int(n),::int(m)]
    
    return cwtmatr_s
    
    











