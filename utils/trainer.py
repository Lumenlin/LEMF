import numpy as np
import torch
from torch.autograd import Variable
from datetime import datetime
import os
from apex import amp
import torch.nn.functional as F


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    every_epoch_loss = 0
    model.train()
    size_rates = [0.75, 1, 1.25]

    for step, data_pack in enumerate(train_loader):
        for rate in size_rates:

            optimizer.zero_grad()
            images, gts, egs = data_pack
            images = images.cuda()
            gts = gts.cuda()
            egs = egs.cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                egs = F.interpolate(egs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            cam1, cam2, cam3, edge_out = model(images)

            loss_edge = loss_func(edge_out, egs)

            loss1 = structure_loss(cam1, gts)
            loss2 = structure_loss(cam2, gts)
            loss3 = structure_loss(cam3, gts)

            loss_obj = loss1 + loss2 + loss3
            loss_total = loss_obj + loss_edge

            # 计算每个epoch的平均loss
            every_epoch_loss += loss_total.item()

            with amp.scale_loss(loss_total, optimizer) as scale_loss:
                scale_loss.backward()

            optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => Loss_obj: {:.4f}, Loss_edge: {'
                  ':.4f}] Loss_all: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_obj.data, loss_edge.data,
                         loss_total.data))

    # 得到每个epoch的平均loss
    every_epoch_loss = every_epoch_loss / total_step

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch + 1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'model_%d.pth' % (epoch + 1))

    return every_epoch_loss
