import random
import math
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,batch_size=10):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.batch_size = batch_size
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img1, img2, flow = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        #获取 gt
        if self.inp_size is None:
            h_lr = math.floor(img1.shape[-2] / s + 1e-9)
            w_lr = math.floor(img1.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            hr_flow = flow[:, :h_hr, :w_hr]
            img1_down = resize_fn(img1, (h_lr, w_lr))
            img2_down = resize_fn(img2, (h_lr, w_lr))
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img1.shape[-2] - w_hr)
            y0 = random.randint(0, img1.shape[-1] - w_hr)
            hr_img1 = img1[:, x0: x0 + w_hr, y0: y0 + w_hr]
            hr_img2 = img2[:, x0: x0 + w_hr, y0: y0 + w_hr]
            hr_flow = flow[:, x0: x0 + w_hr, y0: y0 + w_hr]
            img1_down = resize_fn(hr_img1, w_lr)
            img2_down = resize_fn(hr_img2, w_lr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = hr_flow

        if self.inp_size is not None:
            # 从 [0, 1, 2, ..., h_hr * w_hr - 1] 这 h_hr * w_hr 个像素点中随机选择 h_lr * w_lr 个不重复的像素索引。
            idx = torch.tensor(np.random.choice(h_hr * w_hr, h_lr * w_lr, replace=False))
            idx, _ = torch.sort(idx)
            # idx,_ = torch.sort(idx)
            # 将高分辨率的坐标展平成二维张量，形状为 (h_hr * w_hr, 2)，即每个像素点对应的 (x, y) 坐标对。
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            # 根据随机选择的索引 idx 从展平后的坐标中挑选出高分辨率的像素坐标。
            hr_coord = hr_coord[idx, :]
            # 将采样后的低分辨率坐标重新调整为 (h_lr, w_lr, 2) 的形状
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])
            # 确保张量在内存中是连续的 将 crop_hr 的形状从 (C, H, W) 变为 (C, H * W)，即将图像的每个通道展平成一个一维向量。
            hr_rgb = hr_flow.contiguous().view(hr_flow.shape[0], -1)
            # 使用这些随机索引在展平后的高分辨率图像（hr_rgb）中选择对应的像素值。每一列对应 idx 中的一个索引。
            hr_rgb = hr_rgb[:, idx]
            # 将刚才从高分辨率图像中随机选取的像素值，重新排列为低分辨率图像的形状 (C, h_lr, w_lr)
            hr_rgb = hr_rgb.view(hr_flow.shape[0], h_lr, w_lr)

        cell = torch.tensor([2 / hr_flow.shape[-2], 2 / hr_flow.shape[-1]], dtype=torch.float32)

        return img1_down, img2_down, hr_rgb, hr_coord, cell
