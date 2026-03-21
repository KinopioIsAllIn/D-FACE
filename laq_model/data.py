from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T
import numpy as np
import os
import random


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            offset = self.offset
            
            folder = self.folder_list[index]
            img_list = os.listdir(os.path.join(self.folder, folder))

            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0][4:]))
            ## pick random frame 
            first_frame_idx = random.randint(0, len(img_list)-1)
            first_frame_idx = min(first_frame_idx, len(img_list)-1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list)-1)
            
            first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
            second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))

class CASME3(Dataset):
    def __init__(self, image_size, mode='train'):
        self.augmentor = None
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.image_size = image_size

        self.extra_info = []
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])
        self.au_list = []
        self.mode = mode
        self.get_images()

    # 上采样的多种策略：Deconv, Deconv+Residual block, Pixel shuffle
    def get_images2(self, path='/home1/yicheng/optical-flow-extraction/dataset/CASME3/part_A/'):
        # path = os.path.join(path, 'samm_25', 'feature_25')
        annotation_path = os.path.join(path, 'annotation', 'CAS(ME)3_part_A_v1.xls')
        data_path = os.path.join(path, 'data/part_A')

        import xlrd
        # 读取xls文件
        workbook = xlrd.open_workbook(annotation_path)
        sheet = workbook.sheet_by_index(0)

        # 获取总行数
        num_rows = sheet.nrows

        # 遍历数据，跳过表头
        for row_idx in range(1, num_rows):
            row = sheet.row_values(row_idx)

            # 过滤第6列是否为macro-expression
            if row[6].strip() != 'Macro-expression':
                continue
            # 获取图像路径 (第0列和第1列)
            subject_id = str(row[0])
            session_id = str(row[1]).lower()
            image_dir = os.path.join(data_path, subject_id, session_id, "color")

            # 获取要选取的两张图片 (第2列和第3列)
            if int(row[2]) == 0 or int(row[3]) == 0:
                continue
            img1_num = str(int(row[2])) + '.jpg'
            img2_num = str(int(row[3])) + '.jpg'
            img1_path = os.path.join(image_dir, img1_num)
            img2_path = os.path.join(image_dir, img2_num)

            # 读取AU label (第5列)
            au_label = str(row[4])

            self.image_list.append([img1_path, img2_path])
            self.au_list.append(au_label)


    def get_images(self, path='/home1/yicheng/optical-flow-extraction/dataset/CASME3/part_A/data/part_A/'):
        # path = os.path.join(path, 'samm_25', 'feature_25')
        subject_list = os.listdir(path)

        for subject in subject_list:
            # if self.mode == 'train' and self.subject == subject:
            #     continue
            # if self.mode == 'test' and self.subject != subject:
            #     continue
            sub_path = os.path.join(path, subject)
            video_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
            for i in video_names:  # video_name
                frame_path = os.path.join(sub_path, i, 'color')
                if not os.path.exists(frame_path):
                    continue
                for j in range(0, 10000, 100):  # frame_name
                    if os.path.exists(os.path.join(frame_path, '%d.jpg' % j)):
                        index_target = random.randint(8, 30)
                        if os.path.exists(os.path.join(frame_path, '%d.jpg' % (j + index_target))):
                            self.image_list.append([os.path.join(frame_path, '%d.jpg' % j),
                                                    os.path.join(frame_path, '%d.jpg' % (j + index_target))])


    def __getitem__(self, index):
        index = index % len(self.image_list)

        img = Image.open(self.image_list[index][0])
        next_img = Image.open(self.image_list[index][1])

        transform_img = self.transform(img).unsqueeze(1)
        next_transform_img = self.transform(next_img).unsqueeze(1)

        cat_img = torch.cat([transform_img, next_transform_img], dim=1)
        return cat_img


    # def __rmul__(self, v):
    #     self.flow_list = v * self.flow_list
    #     self.image_list = v * self.image_list
    #     return self

    def __len__(self):
        return len(self.image_list)