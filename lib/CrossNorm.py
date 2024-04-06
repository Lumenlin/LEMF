import functools

import numpy as np
import torch
from torch import nn

def calc_ins_mean_std_v2(x1, x2, eps=1e-5):
    """extract feature map statistics and perform cross normalization"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size1, size2 = x1.size(), x2.size()
    assert len(size1) == 4 and len(size2) == 4
    N1, C1 = size1[:2]
    N2, C2 = size2[:2]

    # Calculate mean and std for x1
    var1 = x1.contiguous().view(N1, C1, -1).var(dim=2) + eps
    std1 = var1.sqrt().view(N1, C1, 1, 1)
    mean1 = x1.contiguous().view(N1, C1, -1).mean(dim=2).view(N1, C1, 1, 1)

    # Calculate mean and std for x2
    var2 = x2.contiguous().view(N2, C2, -1).var(dim=2) + eps
    std2 = var2.sqrt().view(N2, C2, 1, 1)
    mean2 = x2.contiguous().view(N2, C2, -1).mean(dim=2).view(N2, C2, 1, 1)

    return mean1, std1, mean2, std2


def instance_norm_mix_v2(content_feat, style_feat):
    """perform cross normalization"""
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()

    # Calculate mean and std for content_feat
    content_mean, content_std = calc_ins_mean_std_v2(content_feat, style_feat)

    # Calculate mean and std for style_feat
    style_mean, style_std = calc_ins_mean_std_v2(style_feat, content_feat)

    # Normalize content_feat using style statistics
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    # Rescale using style statistics
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# 随机生成一个矩形边界框（bounding box），用于图像裁剪。
def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2


# 实现了一个带有裁剪和通道操作的两个实例交叉归一化（cross-normalization）
def cn_op_2ins_space_chan_v2(x1, x2, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
    """2-instance crossnorm with cropping."""
    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x1.size()[0]).to(x1.device)

    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(x1.size(), beta=beta, bbx_thres=bbx_thres)
        x2_cropped = x2[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2_cropped = x2[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x1.size()[1]).to(x1.device)
        x2_cropped = x2_cropped[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x1_mean, x1_std, x2_mean, x2_std = calc_ins_mean_std_v2(x1, x2_cropped)
        normalized_x1 = (x1 - x1_mean.expand_as(x1)) / x1_std.expand_as(x1)
        normalized_x2 = (x2_cropped - x2_mean.expand_as(x2_cropped)) / x2_std.expand_as(x2_cropped)

        x1_aug = normalized_x2 * x1_std.expand_as(x1) + x2_mean.expand_as(x1)
        x2_aug = normalized_x1 * x2_std.expand_as(x2_cropped) + x1_mean.expand_as(x2_cropped)
    else:
        x1_mean, x1_std, x2_mean, x2_std = calc_ins_mean_std_v2(x1, x2)
        normalized_x1 = (x1 - x1_mean.expand_as(x1)) / x1_std.expand_as(x1)
        normalized_x2 = (x2 - x2_mean.expand_as(x2)) / x2_std.expand_as(x2)

        x1_aug = normalized_x2 * x1_std.expand_as(x1) + x2_mean.expand_as(x1)
        x2_aug = normalized_x1 * x2_std.expand_as(x2) + x1_mean.expand_as(x2)

    if lam is not None:
        x1 = x1 * lam + x1_aug * (1 - lam)
        x2 = x2 * lam + x2_aug * (1 - lam)
    else:
        x1 = x1_aug
        x2 = x2_aug

    return x1, x2


class CrossNorm(nn.Module):

    def __init__(self, crop='neither', beta=None):
        super(CrossNorm, self).__init__()

        self.active = True
        self.cn_op = functools.partial(cn_op_2ins_space_chan_v2,
                                       crop=crop, beta=beta)

    def forward(self, x1, x2):

        if self.training and self.active:
            x1, x2 = self.cn_op(x1, x2)

        self.active = False

        return x1, x2
