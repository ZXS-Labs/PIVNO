import argparse
import os
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import matplotlib.pyplot as plt
import pandas as pd
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=2)
    return pred


def eval_psnr(loader, model):
    model.eval()

    pbar = tqdm(enumerate(loader), total=len(loader), leave=False, desc='val')
    dataset_size = len(loader.dataset)
    epe_array = np.zeros((dataset_size, 5))

    with torch.no_grad():  # 使用 no_grad 避免计算图的保存
        for i_batch, (img1, img2, flow, hr_coord, cell) in pbar:
            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            hr_coord = hr_coord.cuda(non_blocking=True)
            cell = cell.cuda(non_blocking=True)

            # 模型前向传播
            pred = model(img1, img2, hr_coord, cell)[-1]

            B, C, H, W = img1.size()
            epe_list = []

            test_epe_loss = torch.sum((pred - flow) ** 2, dim=1).sqrt().view(-1).mean().item()
            epe_list.append(test_epe_loss)

            epe_array[i_batch * B:i_batch * B + B] = test_epe_loss

    mean_epe_list = np.mean(epe_array)
    return mean_epe_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test-genlmages.yaml')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=1, pin_memory=True, shuffle=False)

    model_path=config['model_path']

    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    import time
    t1 = time.time()
    res = eval_psnr(loader, model)
    t2 = time.time()

    test_type=spec['dataset']['args']['type']
    formatted_res = '{:.4f}'.format(res)  # 将 res 格式化为四位小数
    print(f"AEE results: {test_type}={formatted_res}")
