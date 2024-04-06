import argparse
import os
import torch

from apex import amp

from lib.Network import Network
from utils.DataLoader import get_loader
from utils.trainer import trainer, adjust_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=130,
                        help='epoch number, default=30')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--decay-epoch', type=float, default=120, metavar='N',  # 学习率衰减周期
                        help='epoch interval to decay LR')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.05, metavar='RATE',  # 学习率衰减率
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--batchsize', type=int, default=15,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training Image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='gradient clipping margin')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--save_model', type=str, default='./model/')

    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Image/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')
    parser.add_argument('--train_eg_dir', type=str, default='./Dataset/TrainDataset/Edge/')

    opt = parser.parse_args()

    model = Network().cuda()
    print('-' * 30, model, '-' * 30)

    params = model.parameters()

    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)

    LogitsBCE = torch.nn.BCEWithLogitsLoss()  # 损失函数

    # 对model_SINet模型和优化器 optimizer 进行 AMP 初始化
    net, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # 获取训练数据集
    dataset_train = get_loader(opt.train_img_dir, opt.train_gt_dir, opt.train_eg_dir, batchsize=opt.batchsize,
                               trainsize=opt.trainsize)
    total_step = len(dataset_train)

    print('-' * 30,
          "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\neg_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
          "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.train_eg_dir, opt.lr,
                                                      opt.batchsize, opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)

        trainer(train_loader=dataset_train, model=model,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
