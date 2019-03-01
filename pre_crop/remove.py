from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

#建立query文件夹，移除了所有new——whale
path = '/mnt/sdb2/liucen/whale/'
pathlist = os.listdir(path + 'new_train/')

for name in tqdm(pathlist):
    id = name.split('.')[1]
    if id == 'new_whale':
        continue
    else:
        img = Image.open(path + '/new_train/' + name)
        img.save(path + '/query/' + name)

#统计只有一张鲸鱼图片的id
id_one = []
train = pd.read_csv(path + 'train.csv')
dict = {'name': train['Id'].value_counts().index, 'num' : train['Id'].value_counts().values}
dict = pd.DataFrame(dict)                 #serais类型转变成dataframe
dict_country = dict.set_index('name').T.to_dict()
for name in dict_country:
    if dict_country[name]['num'] == 1:
        id_one.append(name)

#生成pre——train文件夹，移除了所有只有一张鲸鱼图片的id的图
pathlist = os.listdir(path + 'query/')
for name in tqdm(pathlist):
    id = name.split('.')[1]
    if id in id_one:
        continue
    else:
        img = Image.open(path + '/query/' + name)
        img.save(path + 'pre_train/' + name)