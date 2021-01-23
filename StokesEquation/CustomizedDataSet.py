# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:38:07 2020

@author: 47584359
"""

# 子类定义  和上面介绍有点小区别，多了一个图像路径参数，因为我的txt中只有文件名！！！
import torch
import numpy as np

class DatasetFromTxt(torch.utils.data.Dataset):
    def __init__(self, filepath,device, input_transform=None, target_transform=None):
        super(DatasetFromTxt, self).__init__()

        self.data_in_file =  torch.Tensor(np.loadtxt(filepath, delimiter=',').astype(np.float32)).to(device)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = self.data_in_file[index][0:2]
        
        target = self.data_in_file[index][2:]
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.data_in_file)