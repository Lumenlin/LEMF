import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import argparse
from lib.modify2 import Network
from utils.DataLoader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./model_save/LEMF.pth')
parser.add_argument('--save_path', type=str,
                    default='./results/')
opt = parser.parse_args()

model = Network().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['CAMO', 'COD10K', 'NC4K']:
    save_path = opt.save_path + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)

    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, _, cam, _ = model(image)

        res = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path + name, res * 255)

    print('{} Finish!'.format(dataset))
