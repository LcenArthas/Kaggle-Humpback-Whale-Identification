from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

path = '/mnt/sdb2/liucen/whale/'
pathlist = os.listdir(path + 'train/')
# pathlist = os.listdir(path + 'test/')

csv = pd.read_csv(path + 'bounding_boxes.csv')
csv = pd.DataFrame(csv)
dict = csv.set_index('Image').T.to_dict('list')            #转成字典

id_csv = pd.read_csv(path + 'train.csv')
id_csv = pd.DataFrame(id_csv)
id_dict = id_csv.set_index('Image').T.to_dict()

for name in tqdm(pathlist):
    img = Image.open(path + '/train/' + name)
    # img = Image.open(path + '/test/' + name)
    img = np.array(img)
    x0, y0, x1, y1 = dict[name][0], dict[name][1],dict[name][2],dict[name][3]

    if x0 >= 50:
        x0 -= 50
    else:
        x0 = 0
    if y0 >= 50:
        y0 -= 50
    else:
        y0 = 0
    if x1 <= img.shape[1]-50:
        x1 += 50
    else:
        x1 = img.shape[1]
    if y1 <= img.shape[0]-50:
        y1 += 50
    else:
        y1 = img.shape[0]

    img = img[y0: y1, x0: x1]
    img = Image.fromarray(img)

    id = id_dict[name]['Id']

    img.save(path + '/new_train/' + name.split('.')[0] + '.' + id + '.jpg')
    # img.save(path + '/new_test/' + name)


