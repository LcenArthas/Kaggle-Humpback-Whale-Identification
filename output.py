import data
import loss
import torch
import model
from trainer import Trainer
import pandas as pd
import numpy as np

from option import args
import utils.utility as utility

ckpt = utility.checkpoint(args)                           #一个class，load，save，写日志

loader = data.Data(args)
model = model.Model(args, ckpt)
model.load("result/", pre_train='', resume=130)
loss = None
trainer = Trainer(args, model, loss, loader, ckpt)

dis = trainer.test()

#按照提交格式，剔除重复预测
(n, m) = dis.shape  #n=7960,m=15697
predict = []
for i in range(n):
    a = list(dis[i, :])
    b = []
    for j in a:
        if not j in b:
            b.append(j)
            if len(b) == 5:
                break
    predict.append(b)

predict = np.array(predict)

new_whale = 0
for i in range(predict.shape[0]):
    if predict[i][0] == 'new_whale':
        new_whale += 1
print("新鲸鱼的预测比例", new_whale / 7960.0)

pic_name = pd.read_csv(args.datadir + '/sample_submission.csv')['Image'].values.tolist()
pic_name = np.array(pic_name).T
pic_name = pic_name[:, np.newaxis]

result = pd.DataFrame(np.hstack((pic_name, predict)))
result.to_csv("result/output_new11.csv", index=0)
