import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
from loss.prototypical_loss import prototypical_loss as loss_fn
import pandas as pd

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.trainset = loader.trainset
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch,
                self.args.epochs,
                batch + 1,
                len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))

    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        # self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()
        threshold = 0.5

        # self.ckpt.add_log(torch.zeros(1, 5))
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        if self.args.re_rank:
            g_q_dist = np.dot(gf, np.transpose(qf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(g_q_dist, g_g_dist, q_q_dist)

            print("dist的大小",dist.shape)
        else:
            dist = cdist(gf, qf)

        new_dist = np.array([threshold] * dist.shape[0])                   #阈值
        new_dist = new_dist[:, np.newaxis]
        dist = np.hstack((dist, new_dist))

        indices = np.argsort(dist, axis=1)

        dist_sort = np.array(np.sort(dist, axis=1))                         #来查看距离矩阵，确定阈值
        result = pd.DataFrame(dist_sort[:, :10])
        result.to_csv("result/dist2.csv", index=0, header=0)

        ids = self.queryset.ids
        ids.append("new_whale")
        ids = np.asarray(ids)
        return ids[indices]
        # r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
        #         separate_camera_set=False,
        #         single_gallery_shot=False,
        #         first_match_break=True)
        # m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
        #
        # self.ckpt.log[-1, 0] = m_ap
        # self.ckpt.log[-1, 1] = r[0]
        # self.ckpt.log[-1, 2] = r[2]
        # self.ckpt.log[-1, 3] = r[4]
        # self.ckpt.log[-1, 4] = r[9]
        # best = self.ckpt.log.max(0)
        # self.ckpt.write_log(
        #     '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
        #     m_ap,
        #     r[0], r[2], r[4], r[9],
        #     best[0][0],
        #     (best[1][0] + 1)*self.args.test_every
        #     )
        # )
        # if not self.args.test_only:
        #     self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W  得到的是一个从384到1的list
        return inputs.index_select(3,inv_idx)                  #把图片的第三个维度进行从384到0从新排序，即进行反转

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            ff = torch.FloatTensor(inputs.size(0), 4096).zero_()     #这个数值要随时更改
            # for i in range(2):
            #     if i==1:
            #         inputs = self.fliphor(inputs)
            input_img = inputs.to(self.device)             #这行之后的四行前进一个退格符
            outputs = self.model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
            print(features.shape)
        return features

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
