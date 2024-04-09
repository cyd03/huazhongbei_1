import numpy as np
import pandas as pd

path="data_standard/s.csv"
data=pd.read_csv(path,encoding='utf-8')
# print(data)
# 将position和meet两个指标进行极大化处理

position=np.array(data['position'])
position_0= position[position!=0]
position_0=max(position_0)-position_0

position[position!=0]=position_0

# print(position)


std_meet=np.array(data['std_meet'])
std_meet_0=std_meet[std_meet!=0]
std_meet_0=max(std_meet_0)-std_meet_0
std_meet[std_meet!=0]=std_meet_0
# print(std_meet)

# 正向化处理
standard=np.array([data['extend'],data['len'],data['num']])
standard=np.vstack((standard,position))
standard=np.vstack((standard,std_meet))

standard=standard.transpose()
standard=standard/np.sqrt(np.sum(standard**2,axis=0))
# print(standard)
# 计算概率矩阵
standard=standard+1e-30
n,m=standard.shape
p=standard/np.sum(standard,axis=0)
# print(p)
entropy=-np.sum(p*np.log(p),axis=0)/np.log(n)
weights=(1-entropy)/np.sum(1-entropy)
print(weights)

standard=1-np.sum(standard*weights,axis=1)
print(standard)