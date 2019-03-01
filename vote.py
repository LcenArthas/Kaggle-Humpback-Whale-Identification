import pandas as pd
import numpy as np
import os

path = "pre_vote/"
dir = os.listdir(path)

pic = pd.read_csv(path + dir[0])["Image"].values.tolist() #pic的列表存储是pic_name
pic = np.array(pic).T
pic = pic[:, np.newaxis]

top_11 = []
top_22 = []
top_33 = []
top_44 = []
top_55 = []
for file_name in dir:                              #对每个csv文件处理
    csv = pd.read_csv(path + file_name)["Id"].values.tolist()
    top_1 = []
    top_2 = []
    top_3 = []
    top_4 = []
    top_5 = []
    for evr_id in csv:                             #对每个csv里面的每张图片的结果进行处理
        ids = evr_id.split()
        top_1.append(ids[0])                       #每个top_n中存储着每张图片的top_n，长度是7960
        top_2.append(ids[1])
        top_3.append(ids[2])
        top_4.append(ids[3])
        top_5.append(ids[4])
    top_11.append(top_1)                           #每个top_nn中存储着每个文件的top_n，长度是文件数
    top_22.append(top_2)
    top_33.append(top_3)
    top_44.append(top_4)
    top_55.append(top_5)
#经过上面得到的结构：top_11[[top_1], [top_1],...,[top_1]]

print("finished[1/2]")

#函数：返回列表中出现次数最多的元素
def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str

#函数：返回集成好的top_n
def subm(lt):
    submit_1 = []                                   #submit_n存储每张图片的集成后的top_n的预测，长度为7960
    for i in range(len(lt[0])):
        temp = []
        for result in lt:
            temp.append(result[i])
        submit_1.append(max_list(temp))
    return submit_1

top_1 = np.array(subm(top_11))[:, np.newaxis]
top_2 = np.array(subm(top_22))[:, np.newaxis]
top_3 = np.array(subm(top_33))[:, np.newaxis]
top_4 = np.array(subm(top_44))[:, np.newaxis]
top_5 = np.array(subm(top_55))[:, np.newaxis]

print(top_1.shape)
print(pic.shape)

result = pd.DataFrame(np.hstack((pic, top_1, top_2, top_3, top_4, top_5)))
result.to_csv(path + "final_output.csv", index=0)

print("finished[2/2]")

