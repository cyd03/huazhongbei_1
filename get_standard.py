import csv
import pandas as pd
import numpy as np
from nltk import flatten
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
def get_len(lsttype,weizhi):
	my_dict_len={}
	for i in range(len(weizhi)):
		if i ==0:
			length=float(weizhi[i])-0
		else:
			length=float(weizhi[i])-float(weizhi[i-1])
		if lsttype[i] in my_dict_len:
			my_dict_len[lsttype[i]]+=length
		else:
			my_dict_len[lsttype[i]]=length
	return my_dict_len

def get_data(path):
	data = pd.read_csv(path, encoding='utf-8')
	weizhi = data[['weizhi']]
	ttype = data[['type']]
	weizhi = flatten(weizhi.values.tolist())
	ttype = flatten(ttype.values.tolist())
	return weizhi,ttype
def get_standard_extend(my_dict,my_dict_len,ttype):
	num_loss = np.array(list(my_dict.values()))
	len_loss = np.array(list(my_dict_len.values()))
	num_loss = num_loss / sum(num_loss)
	len_loss = len_loss / sum(len_loss)
	tylist = []
	for i in my_dict:
		if i == '正常':
			tylist.append(1)
		elif i == '微小断丝':
			tylist.append(2)
		elif i == '轻度断丝':
			tylist.append(4)
		elif i == '中度断丝':
			tylist.append(8)
		elif i == '重度断丝':
			tylist.append(16)
		elif i =='内部断股':
			tylist.append(32)
	type_loss = np.array(tylist)
	standard_extend = sum(type_loss * num_loss * len_loss)
	# print(standard_extend)


	standard_num = len(ttype) - my_dict['正常']

	standard_len = sum(my_dict_len.values()) - my_dict_len['正常']

	np_weizhi = []
	for i in range(len(ttype)):
		if ttype[i] != '正常':
			np_weizhi.append(weizhi[i])
	np_weizhi = np.array(np_weizhi)
	standard_position = sum(abs(np_weizhi - 363.2) / 363.2 / standard_len)
	if np_weizhi.size!=0:
		standard_meet = np_weizhi.std()
	else:
		standard_meet=0
	save_data = np.array([standard_extend, standard_len, standard_num, standard_position, standard_meet])
	return save_data
def save_standard(save_data):
	f = open("data_standard/s.csv", "a+", encoding="utf-8", newline='')
	csv_writer = csv.writer(f)
	csv_writer.writerow(save_data)
	f.close()

f = open("data_standard/s.csv", "w", encoding="utf-8", newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(["extend", "len", "num", "position", "std_meet"])
f.close()
for i in range(6):
	path="clean_data/s_"+str(i+1)+".csv"
	weizhi,ttype=get_data(path)
	standard_quality=1
	if standard_quality== 1:
		my_dict = solve_dict(ttype, len(ttype))
		my_dict_len = get_len(ttype, weizhi)
		print(my_dict)
		print(my_dict_len)
		save_data=get_standard_extend(my_dict,my_dict_len,ttype)
		print(save_data)
		save_standard(save_data)

# 获取程度指标

# 获得异常数量指标


# 获得异常长度指标



# 位置危险系数指标，极大化处理

# 缺陷集中程度指标，极大化处理






