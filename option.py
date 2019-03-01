import argparse

parser = argparse.ArgumentParser(description='MGN')

parser.add_argument('--nThread', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs')

parser.add_argument("--datadir", type=str, default="/mnt/sdb2/liucen/whale/", help='dataset directory')
parser.add_argument('--data_train', type=str, default='Market1501', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Market1501', help='test dataset name')

parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument("--epochs", type=int, default=500, help='number of epochs to train')
parser.add_argument('--test_every', type=int, default=20, help='do test per every N epochs')
parser.add_argument("--batchid", type=int, default=12, help='the batch for id')                 #before:16
parser.add_argument("--batchimage", type=int, default=4, help='the batch of per id')
parser.add_argument("--batchtest", type=int, default=16, help='input batch size for test')      #before:32
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

parser.add_argument('--model', default='MGN', help='model name :MGN & PN')
parser.add_argument('--loss', type=str, default='1*CrossEntropy+2*Triplet', help='loss function configuration')

parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pool', type=str, default='avg', help='pool function')
parser.add_argument('--feats', type=int, default=512, help='number of feature maps')            #before:256
parser.add_argument('--height', type=int, default=256, help='height of the input image')        #before:128
parser.add_argument('--width', type=int, default=768, help='width of the input image')          #before:384
parser.add_argument('--num_classes', type=int, default=2931, help='')


parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD','ADAM','NADAM','RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=60, help='learning rate decay per N epochs')

parser.add_argument("--margin", type=float, default=1.2, help='')
parser.add_argument("--re_rank", action='store_true', help='')
parser.add_argument("--random_erasing", action='store_true', help='')                #随机擦除
parser.add_argument("--probability", type=float, default=0.5, help='')               #随机擦出比例

parser.add_argument("--savedir", type=str, default='saved_models', help='directory name to save')
parser.add_argument("--outdir", type=str, default='out', help='')
parser.add_argument("--resume", type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load', type=str, default='', help='file name to load')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')

#prototypical
parser.add_argument('--classes_per_it_tr', type=int, help='number of random classes per episode for training, default=60', default=60)
parser.add_argument('--num_support_tr', type=int, help='number of samples per class to use as support for training, default=5', default=1)
parser.add_argument('--num_query_tr', type=int, help='number of samples per class to use as query for training, default=5', default=1)
parser.add_argument('--iterations', type=int, help='number of episodes per epoch, default=100', default=100)

args = parser.parse_args()    #进行解析

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
