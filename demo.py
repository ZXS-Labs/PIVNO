import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import models
import matplotlib.pyplot as plt
from utils import make_coord

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input_images')
    parser.add_argument('--output', default='./output_images')
    parser.add_argument('--model', default='./save/_train_PIV-genImages/epoch-best.pth')
    parser.add_argument('--scale_max', type=int, default=1, help='Maximum scale factor')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    scale_max = args.scale_max
    # 获取文件夹路径
    input_folder = args.input
    # 创建 output 文件夹
    os.makedirs(args.output, exist_ok=True)

    # 初始化两个列表，分别存储 img1 和 img2 开头的图片
    img1_files = []
    img2_files = []

    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, filename)

        # 检查文件名中是否包含 'img1' 或 'img2'，并且是图片文件
        if 'img1' in filename and filename.endswith(('.png','.bmp', '.jpg', '.jpeg', '.tif')):
            img1_files.append(filepath)
        elif 'img2' in filename and filename.endswith(('.png','.bmp', '.jpg', '.jpeg', '.tif')):
            img2_files.append(filepath)

    # 确保 img1 和 img2 数量一致
    assert len(img1_files) == len(img2_files), "img1 和 img2 数量不匹配"

    for i, (img1_path, img2_path) in enumerate(zip(img1_files, img2_files)):
        # 加载图片并转换为灰度张量
        transform = transforms.ToTensor()
        img1 = transform(Image.open(img1_path).convert('L'))  # 加载为灰度
        img2 = transform(Image.open(img2_path).convert('L'))  # 加载为灰度

        # 将图像移动到 GPU
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

        # 计算缩放尺寸
        h = int(img1.shape[-2] * scale_max)
        w = int(img1.shape[-1] * scale_max)
        scale = h / img1.shape[-2]

        # 生成坐标
        coord = make_coord([h, w], flatten=False).cuda()
        cell = torch.ones(2).cuda()
        cell[0] *= 2 / h
        cell[1] *= 2 / w
        cell_factor = max(scale / scale_max, 1)

        # 预测
        with torch.no_grad():
            pred = model(img1.unsqueeze(0), img2.unsqueeze(0), coord.unsqueeze(0), cell_factor * cell.unsqueeze(0))[-1]


        ###不只是用红色的
        # 将预测结果转换为光流图像格式
        flow_reshaped = pred[0].permute(1, 2, 0).cpu().numpy()  # 转换为 [h, w, 2]

        # 获取原始光流图像尺寸
        height, width, channels = flow_reshaped.shape
        xx, yy = np.meshgrid(range(width), range(height))

        # 将背景灰度图 img1 调整为光流图的大小（256x256）
        img1_resized = cv2.resize(img1.squeeze(0).cpu().numpy(), (width, height))  # 调整为 (256, 256)

        # 计算光流的模长（位移长度）
        magnitude = np.sqrt(flow_reshaped[..., 0] ** 2 + flow_reshaped[..., 1] ** 2)

        # 设置箭头间隔为 5
        s = 10

        # 绘制光流图并将调整后的灰度图作为背景
        plt.figure(figsize=(20, 20))
        plt.imshow(img1_resized, cmap='gray')  # 显示灰度背景

        # 映射颜色到位移长度
        # 映射颜色到位移长度
        norm = plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max())  # 归一化位移长度
        colors = plt.cm.viridis(norm(magnitude[::s, ::s])).reshape(-1, 4)  # 确保是 [N, 4] 格式

        # 绘制箭头，颜色随位移长度变化
        plt.quiver(
            xx[::s, ::s], yy[::s, ::s],
            flow_reshaped[..., 0][::s, ::s],
            -flow_reshaped[..., 1][::s, ::s],
            color=colors, #angles='xy', scale_units='xy', scale=1
        )
        # 强制刷新并绘制图形
        plt.draw()

        # 保存图片
        output_path = os.path.join(args.output, f"output_flow_{i:04d}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved flow visualization with background to {output_path}")

        del img1, img2, pred, coord, cell
        torch.cuda.empty_cache()
