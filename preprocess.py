# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:14:42 2019

@author: 彭钊

原始地震波形数据窗口化、去趋势、标准化等预处理操作的自编额外函数扩展包

所需额外扩展包：obspy、numpy

"""

from obspy.core import read
from obspy.core import UTCDateTime
from math import radians, cos, sin, asin, sqrt
import numpy as np

'''
读取地震目录，得到地震的发震时间
根据计算的地震波到达仪器的时间差得到P波或S波的到时
进而确定地震事件窗口化的起始时间。
本软件CCLSN V1.0的初始窗口时长为25s。

还包括使用初始时间偏移方法进行数据扩增的函数
'''
def read_catalog_orig(filename,arriving_time):
    # filename为地震目录
    # arriving_time为地震到时，time为UTCDateTime类，QKT、RHT两个台站大约为10s，TNC大约为12s
    # 原始到时，无数据扩增方法，week1数量为1556，week2数量为5072
    
    cat=[]
    
    with open(filename, 'r') as f:
        for line in f:
            year = int(line[0:4])
            mon = int(line[5:6])
            day = int(line[7:9])
            hour = int(line[10:12])
            mins = int(line[13:15])
            secs = float(line[16:20])
            
            time = UTCDateTime(year, mon, day,hour, mins, secs) - 28800 + arriving_time
            # time为UTCDateTime类,为世界标准时，与北京时间有8h时间差
            cat.append(time)
 
    return cat

def read_catalog_ex4(filename,arriving_time):
    # filename为地震目录
    # arriving_time为地震到时，time为UTCDateTime类，QKT、RHT两个台站大约为10s，TNC大约为12s
    # 4倍数据扩增，时间偏移量为-1s、-0.5s、+0.5s，数据扩增为4倍，week1数量为6224
    
    cat=[]
    
    with open(filename, 'r') as f:
        for line in f:
            year = int(line[0:4])
            mon = int(line[5:6])
            day = int(line[7:9])
            hour = int(line[10:12])
            mins = int(line[13:15])
            secs = float(line[16:20])
            
            time1 = UTCDateTime(year, mon, day,hour, mins, secs) - 28800 + arriving_time
            cat.append(time1)
            
            time2 = UTCDateTime(year, mon, day,hour, mins, secs) - 28800 + arriving_time - 1
            cat.append(time2)
            
            time3 = UTCDateTime(year, mon, day,hour, mins, secs) - 28800 + arriving_time - 0.5
            cat.append(time3)
            
            time4 = UTCDateTime(year, mon, day,hour, mins, secs) - 28800 + arriving_time + 0.5
            cat.append(time4)
 
    return cat



'''
输出背景噪声窗口化的初始时间
'''
# 20140523-24两天噪声窗
def noise_train_orig():
    #starttime为起始时刻，lasttime为结束时间
    #一分钟取6段
    #配合无扩增方法事件集数量，数量为1440
    #时间偏移量为+2，数据扩增为2倍
    
    cat = []
    
    starttime_1 = UTCDateTime(2014,5,22,19,25,0)
    lasttime_1 = UTCDateTime(2014,5,22,20,0,0)
    starttime_2 = UTCDateTime(2014,5,22,21,0,0)
    lasttime_2 = UTCDateTime(2014,5,22,22,25,0)
        
    t = starttime_1
    dt = lasttime_1
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
        
    t = starttime_2
    dt = lasttime_2
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
        
    return cat

def noise_train_ex4():
    #starttime为起始时刻，lasttime为结束时间
    #一分钟取10段
    #配合4倍扩增方法事件集数量，数量为6000
    #时间偏移量为+0.5、+1.0、+1.5、+2，数据扩增为5倍
    
    cat = []
    
    starttime_1 = UTCDateTime(2014,5,22,19,25,0)
    lasttime_1 = UTCDateTime(2014,5,22,20,0,0)
    starttime_2 = UTCDateTime(2014,5,22,21,0,0)
    lasttime_2 = UTCDateTime(2014,5,22,22,25,0)
        
    t = starttime_1
    dt = lasttime_1
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 0.5)
        cat.append(t + 1)
        cat.append(t + 1.5)
        cat.append(t + 2)
        t = t + 6
        
    t = starttime_2
    dt = lasttime_2
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 0.5)
        cat.append(t + 1)
        cat.append(t + 1.5)
        cat.append(t + 2)
        t = t + 6
        
    return cat



def noise_test():
    #starttime为起始时刻，lasttime为结束时间
    #一分钟取6段
    #配合测试事件集数量，数量为5072
    #时间偏移量为+2，数据扩增为2倍，数量为4440
    cat = []
    
    starttime_1 = UTCDateTime(2014,5,22,16,15,0)
    lasttime_1 = UTCDateTime(2014,5,22,17,10,0)
    starttime_2 = UTCDateTime(2014,5,22,17,15,0)
    lasttime_2 = UTCDateTime(2014,5,22,19,15,0)
    starttime_3 = UTCDateTime(2014,5,23,13,30,0)
    lasttime_3 = UTCDateTime(2014,5,23,14,50,0)
    starttime_4 = UTCDateTime(2014,5,23,15,20,0)
    lasttime_4 = UTCDateTime(2014,5,23,16,5,0)
    starttime_5 = UTCDateTime(2014,5,23,18,30,0)
    lasttime_5 = UTCDateTime(2014,5,23,19,40,0)
    
    t = starttime_1
    dt = lasttime_1
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
        
    t = starttime_2
    dt = lasttime_2
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
        
    t = starttime_3
    dt = lasttime_3
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
    
    t = starttime_4
    dt = lasttime_4
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10
    
    t = starttime_5
    dt = lasttime_5
    while dt - t > 0 :
        cat.append(t)
        cat.append(t + 2)
        t = t + 10

    return cat


'''
根据地震理论到时从波形文件中截取波形，并标准化
'''
#截取长度为25s，三分量垂直堆叠
#输入st为obspy读取的Stream对象

def streams_slice(st,arrival_time):
    
    t = arrival_time
    
    tr1 = st[0]             #3分量之一
    tr1 = tr1.slice(t-2,t+23)    #截取波形，时间区间为25秒
    tr1 = tr1.detrend()     #去趋势
    tr1 = tr1.normalize()   #标准化
    
    tr2 = st[1]             #3分量之一
    tr2 = tr2.slice(t-2,t+23)    #截取波形，时间区间为25秒
    tr2 = tr2.detrend()     #去趋势
    tr2 = tr2.normalize()   #标准化
    
    tr3 = st[2]             #3分量之一
    tr3 = tr3.slice(t-2,t+23)    #截取波形，时间区间为25秒
    tr3 = tr3.detrend()     #去趋势
    tr3 = tr3.normalize()   #标准化
    
    data = np.vstack((tr1.data,tr2.data,tr3.data))
    data = data.T
                          
    return data    

###计算地震波的理论到时###
def arrival_time(lng1,lat1,lng2,lat2,depth,time):  
    #lng1,lat1为台站经纬度,lng为纬度，lat为经度
    #lng2,lat2,depth,time为地震震中、深度、发震时刻，由read_catalog读取
    distance = sqrt(geodistance(lng1,lat1,lng2,lat2)**2+float(depth)**2) #震中到台站距离
    vel_p = 6.2  #地区地壳波速
    t = distance/vel_p
    
    return time+t

###根据经纬度计算距离，单位为km###   
def geodistance(lng1,lat1,lng2,lat2):
    
    r = 6371 #地球平均半径
    
    lng1,lat1,lng2,lat2 = map(radians,[float(lng1),float(lat1),float(lng2),float(lat2)])#将经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*r
    
    return distance
    