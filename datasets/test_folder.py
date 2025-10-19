import os
import os.path as osp
from glob import glob
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('test-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, type):
        # 读取数据列表
        if not osp.exists(root_path):
            raise FileNotFoundError(f"{root_path} does not exist.")

        with open(root_path, 'r') as f:
            data_list = f.readlines()

        # 去除每行末尾的换行符，并按制表符分割
        data_list = [line.strip().split('\t') for line in data_list]
        # 在每个路径前面加上 './'
        data_list = [['./' + path for path in line] for line in data_list]

        # 根据关键词分组
        self.dataset_types = {
            "backstep": [],
            "cylinder": [],
            "DNS_turbulence": [],
            "JHTDB": [],
            "SQG": [],
            "uniform": []
        }

        # 将每一行数据分配到对应的类别
        for item in data_list:
            if "backstep" in item[0]:
                self.dataset_types["backstep"].append(item)
            elif "cylinder" in item[0]:
                self.dataset_types["cylinder"].append(item)
            elif "DNS_turbulence" in item[0]:
                self.dataset_types["DNS_turbulence"].append(item)
            elif "JHTDB" in item[0]:
                self.dataset_types["JHTDB"].append(item)
            elif "SQG" in item[0]:
                self.dataset_types["SQG"].append(item)
            elif "uniform" in item[0]:
                self.dataset_types["uniform"].append(item)

        # 定义图像转换
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 在初始化时加载所有图像和流数据到内存
        self.img1_list = []
        self.img2_list = []
        self.flow_list = []

        # 选择需要加载的某个数据集类型，这里以 'backstep' 为例
        self.selected_data = self.dataset_types[type]  # 可以根据需要更改为其他数据集

        for img1_path, img2_path, flow_path in self.selected_data:
            try:
                # 读取图像，转换为灰度图
                img1 = Image.open(img1_path).convert('L')
                img2 = Image.open(img2_path).convert('L')
                # 图像转换为张量并保存到内存
                img1 = self.transforms(img1)
                img2 = self.transforms(img2)
                # 读取流场数据并保存到内存
                flow_data = self.load_flow(flow_path)
                flow_data = self.transforms(flow_data)
                self.img1_list.append(img1)
                self.img2_list.append(img2)
                self.flow_list.append(flow_data)

            except Exception as e:
                print(f"Error loading data at paths {img1_path}, {img2_path}, {flow_path}: {e}")
                raise e

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        # 直接从内存中返回数据，而不是从文件系统中读取
        img1 = self.img1_list[idx]
        img2 = self.img2_list[idx]
        flow_data = self.flow_list[idx]
        return img1, img2, flow_data

    def load_flow(self, file_path):
        """读取 .flo 文件的函数"""
        with open(file_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)[0]
            if magic != 202021.25:
                raise ValueError(f'Magic number incorrect in file {file_path}')

            width = np.fromfile(f, np.int32, count=1)[0]
            height = np.fromfile(f, np.int32, count=1)[0]

            flow = np.fromfile(f, np.float32, count=2 * width * height)
            flow = np.resize(flow, (height, width, 2))
        return flow
