from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from torch.utils.data import dataloader
from data.prototypical_sampler import PrototypicalBatchSampler

class Data:
    def __init__(self, args):

        train_list = [
            # transforms.RandomResizedCrop(size=(args.height, args.width),scale=(0.97, 1.0)),   #随机剪裁，0.97
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),                                                       #随机角度旋转+-5
            transforms.RandomAffine(5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),               #亮度随机变化0.5
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # if args.random_erasing:
        #     train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')   #调用market1501类，第二个括号里面是参数
            # 为了propotypical 服务
            # path = [j for j in self.trainset.imgs]
            # y = []
            # for _, i in enumerate(path):
            #     y.append(self.trainset._id2label[self.trainset.id(i)])
            # y = tuple(y)
            # sampler = PrototypicalBatchSampler(labels=y,
            #                                     classes_per_it=args.classes_per_it_tr,
            #                                     num_samples=args.num_support_tr + args.num_query_tr,
            #                                     iterations=args.iterations)
            self.train_loader = dataloader.DataLoader(self.trainset,
                            sampler=RandomSampler(self.trainset, args.batchid, batch_image=args.batchimage),  #MGN的smaple
                            #batch_sampler=sampler,                                                           #prototypical的smaple
                            batch_size=args.batchid * args.batchimage,                                        #MGN
                            num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        