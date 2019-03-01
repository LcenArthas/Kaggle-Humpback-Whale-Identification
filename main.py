import data
import loss
import torch
import model
from trainer import Trainer

from option import args
import utils.utility as utility

ckpt = utility.checkpoint(args)                           #一个class，load，save，写日志

loader = data.Data(args)
model = model.Model(args, ckpt)
loss = loss.Loss(args, ckpt) if not args.test_only else None
trainer = Trainer(args, model, loss, loader, ckpt)

n = 0
while not trainer.terminate():
	n += 1
	trainer.train()
	# if args.test_every!=0 and n%args.test_every==0:
	# 	trainer.test()
	if n % 10 == 0:               #每10个epoach保存下模型
		model.save("result/", n)

