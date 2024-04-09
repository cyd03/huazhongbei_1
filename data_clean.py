import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import pywt
from nltk import flatten
import numpy as np
import csv
def xiaobo(maichong_1_0, dian_ya_1_0):
    # 获取脉冲数据中的最大值，即最大传感器数
    res = max(maichong_1_0)
    count = 1
    a = 0
    list1 = []
    list2 = []

    # 找到脉冲数据中每个传感器数的索引
    while count <= res:
        if maichong_1_0[a] == count:
            a += 1
        elif maichong_1_0[a] > count:
            list1.append(a)
            count = count + 1
        if a == len(maichong_1_0):
            list1.append(a)
            break

    # 计算每个区间的临界值
    for i in range(len(list1)):
        if i == 0:
            a1 = 0
            a2 = list1[i]
        else:
            a1 = list1[i - 1]
            a2 = list1[i]
        tmp = ((0.4 / (a2 - a1)))
        list2.append(tmp)

    ecg = dian_ya_1_0
    a = 0
    index = []
    data = []
    bizhi = []
    X = 0

    # 根据临界值对电压数据进行处理
    for i in range(max(list1)):
        if i > list1[a]:
            a = a + 1
        X = X + float(list2[a])
        Y = float(ecg[i])
        index.append(X)
        data.append(Y)

    w = pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print('maximum level is ' + str(maxlev))
    threshold = 1
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)

    # 对每一层的系数进行阈值处理
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    # 小波重构
    datarec = pywt.waverec(coeffs, 'db8')
    mintime = 0
    maxtime = mintime + len(data) + 1
    a = mean(datarec)

    # 计算相对差值百分比
    for i in range(max(list1)):
        Z = float(abs(datarec[i] - a) / a * 100)
        bizhi.append(Z)

    return index, data, datarec, bizhi, mintime, maxtime

def draw(index,data,datarec,bizhi,mintime,maxtime):
    plt.figure
    plt.subplot(3, 1, 1)
    plt.plot(index[mintime:maxtime], data[mintime:maxtime - 1])
    plt.xlabel('time(s)')
    plt.ylabel('voltage')
    plt.title('raw single')
    plt.subplot(3, 1, 2)
    plt.plot(index[mintime:maxtime], datarec[mintime:maxtime - 1])
    plt.xlabel('time(s)')
    plt.ylabel('voltage')
    plt.title('de-noised single use wavelet techniques')
    plt.subplot(3, 1, 3)
    plt.plot(index[mintime:maxtime], bizhi[mintime:maxtime - 1])
    plt.xlabel('time(s)')
    plt.ylabel('voltage')
    plt.title('bizhi tu')
    plt.tight_layout()
    plt.show()

def lianghua(datarec,n):
    bizhi = []
    ymean = mean(datarec)
    for i in range(len(datarec)):
        bizhi.append(abs(datarec[i] - ymean) / ymean * 100)
    outlier_x = []
    overier_x = []
    outlier_n = 0
    lsttype = []
    flag = -1
    b = len(datarec)
    for i in range(b):
        if (bizhi[i] <= 1) and flag != 0:
            flag = 0
            outlier_n = outlier_n + 1
            lsttype.append("正常")
            outlier_x.append(i)
            overier_x.append(i)
        elif (bizhi[i] > 1) and (bizhi[i] <= 1.6) and flag != 1:
            flag = 1
            outlier_n = outlier_n + 1
            lsttype.append("微小断丝")
            outlier_x.append(i)
            overier_x.append(i)
        elif (bizhi[i] > 1.6) and (bizhi[i] <= 2.5) and flag != 2:
            flag = 2
            outlier_n = outlier_n + 1
            lsttype.append("轻度断丝")
            outlier_x.append(i)
            overier_x.append(i)
        elif (bizhi[i] > 2.5) and (bizhi[i] <= 5) and flag != 3:
            flag = 3
            outlier_n = outlier_n + 1
            lsttype.append("中度断丝")
            outlier_x.append(i)
            overier_x.append(i)
        elif (bizhi[i] > 5) and (bizhi[i] <= 14) and flag != 4:
            flag = 4
            outlier_n = outlier_n + 1
            lsttype.append("重度断丝")
            outlier_x.append(i)
            overier_x.append(i)
        elif (bizhi[i] > 14) and flag != 5:
            flag = 5
            outlier_n = outlier_n + 1
            lsttype.append("内部断股")
            outlier_x.append(i)
            overier_x.append(i)
    overier_x.append(len(datarec)-1)
    return outlier_x,overier_x,outlier_n,lsttype

def solve_data(path,num):
    data=pd.read_csv(path)
    dianya= data[[num]]
    maichong= data[['maichong']]
    dianya= flatten(dianya.values.tolist())
    maichong = flatten(maichong.values.tolist())
    return dianya,maichong

def solve_dict(lsttype,outlier_n):
    my_dict = {}
    for i in lsttype:
        if i in my_dict:
            my_dict[i] += 1
        else:
            my_dict[i] = 1
    print(outlier_n)
    print("缺陷数量", outlier_n - my_dict["正常"])
    return my_dict

def get_weizhi(outlier_x,lsttype,overier_x,index):
    alst1 = []
    weizhi = []
    for i in range(len(outlier_x)):
        alst1.append(index[outlier_x[i]])
        alst1.append(lsttype[i])
        weizhi.append(index[overier_x[i + 1] - 1])
    print(alst1)
    print(weizhi)
    return alst1,weizhi


def write_data(weizhi,lsttype,path):
    in_data = np.array([weizhi, lsttype])
    in_data = in_data.transpose()
    f = open(path, "w", encoding="utf-8", newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['weizhi', 'type'])
    csv_writer.writerows(in_data)
    f.close()
