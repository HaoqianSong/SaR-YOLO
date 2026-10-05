### 第五步：创建人员搜索数据集适配器
#python:yolov13-main/ultralytics/data/person_search.py
"""
人员搜索数据集适配器，支持CUHK-SYSU和PRW数据集
"""

import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json
from ultralytics.data.base import BaseDataset
from ultralytics.utils import LOGGER


class PersonSearchDataset(BaseDataset):
    """
    人员搜索数据集类，支持CUHK-SYSU和PRW格式
    
    数据格式要求：
    - 图片路径
    - 检测框标注（YOLO格式：class_id, x_center, y_center, width, height）
    - 人员ID标注（pid）
    """
    
    def __init__(self, img_path, imgsz=640, cache=False, augment=True, hyp=None, prefix="", rect=False, batch_size=16, stride=32, pad=0.5, single_cls=False, classes=None, fraction=1.0):
        self.use_reid = True  # 启用ReID功能
        super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, fraction)
    
    def get_labels(self):
        """获取数据集标签，包含检测框和person ID"""
        self.label_files = self._get_label_files()
        
        labels = []
        for label_file in self.label_files:
            label_data = self._load_label(label_file)
            labels.append(label_data)
        
        return labels
    
    def _get_label_files(self):
        """获取标签文件路径"""
        img_files = self.im_files
        label_files = []
        
        for img_file in img_files:
            # 将图片路径转换为标签路径
            label_file = str(img_file).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            label_files.append(label_file)
        
        return label_files
    
    def _load_label(self, label_file):
        """
        加载单个标签文件
        
        标签格式：
        class_id x_center y_center width height pid
        其中pid是person ID，如果是背景则为-1
        """
        label_data = {
            'cls': np.array([]),
            'bboxes': np.array([]).reshape(0, 4),
            'pids': np.array([]),
            'segments': [],
            'keypoints': None,
            'normalized': True,
            'bbox_format': 'xywh'
        }
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.read().strip().split('\n')
            
            if lines and lines[0]:
                data = np.array([line.split() for line in lines], dtype=np.float32)
                
                if len(data):
                    # 检查是否包含pid列
                    if data.shape[1] >= 6:  # class_id, x, y, w, h, pid
                        label_data['cls'] = data[:, 0].astype(int)
                        label_data['bboxes'] = data[:, 1:5]
                        label_data['pids'] = data[:, 5].astype(int)
                    else:  # 只有检测标注，没有pid
                        label_data['cls'] = data[:, 0].astype(int)
                        label_data['bboxes'] = data[:, 1:5]
                        label_data['pids'] = np.full(len(data), -1, dtype=int)  # 默认背景
        
        return label_data
    
    def __getitem__(self, index):
        """获取单个样本"""
        # 调用父类方法获取基本数据
        item = super().__getitem__(index)
        
        # 添加ReID标签
        if self.use_reid and 'pids' in self.labels[index]:
            item['pids'] = torch.from_numpy(self.labels[index]['pids']).long()
        
        return item
    
    def collate_fn(self, batch):
        """批处理函数，支持ReID标签"""
        # 调用父类的collate函数
        new_batch = super().collate_fn(batch)
        
        # 添加ReID标签的批处理
        if self.use_reid:
            pids_list = []
            for item in batch:
                if 'pids' in item:
                    pids_list.append(item['pids'])
                else:
                    # 如果没有pid，创建空的tensor
                    pids_list.append(torch.tensor([], dtype=torch.long))
            
            # 将所有pid连接起来
            if pids_list:
                # 处理不同长度的pid tensor
                max_len = max(len(pids) for pids in pids_list)
                padded_pids = []
                for pids in pids_list:
                    if len(pids) < max_len:
                        # 用-1填充
                        padded = torch.full((max_len,), -1, dtype=torch.long)
                        if len(pids) > 0:
                            padded[:len(pids)] = pids
                        padded_pids.append(padded)
                    else:
                        padded_pids.append(pids)
                
                new_batch['pids'] = torch.stack(padded_pids)
        
        return new_batch


def create_person_search_dataset(data_dir, split='train'):
    """
    创建人员搜索数据集的便捷函数
    
    Args:
        data_dir: 数据集根目录
        split: 'train', 'val', 或 'test'
    
    Returns:
        PersonSearchDataset实例
    """
    img_path = os.path.join(data_dir, split, 'images')
    return PersonSearchDataset(img_path)





