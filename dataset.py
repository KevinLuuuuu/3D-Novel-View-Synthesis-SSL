from torch.utils.data import Dataset
import os
from PIL import Image
import imageio
import numpy as np
import torch
import json 
import csv


class ImageDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform, 
        train_set = True
    ):
        super(ImageDataset).__init__()
        self.data_path = data_path
        self.images = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith(".jpg")]
        self.transform = transform
        self.train_set = train_set
  
    def __len__(self):
        return len(self.images)
  
    def __getitem__(self,index):
        if self.train_set:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            return image
        else:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            image_name = image_name.split('/')[-1]

            return image, image_name

with open('label.json') as f:
    label = json.load(f)

class OfficeImage(Dataset):

    def __init__(
        self,
        data_path,
        transform, 
        train_set = False,
        valid_set = False, 
        test_csv_path = ""
    ):
        super(OfficeImage).__init__()
        self.data_path = data_path
        self.images = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith(".jpg")]
        self.transform = transform
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_csv_path = test_csv_path
        if self.train_set:
            self.train_label = {}
            with open("hw4_data/office/train.csv", 'r') as file:
                csvreader = csv.reader(file)
                for i, row in enumerate(csvreader):
                    if i==0:
                        continue
                    self.train_label[row[1]] = label[row[2]]
        elif self.valid_set:
            self.valid_label = {}
            with open("hw4_data/office/val.csv", 'r') as file:
                csvreader = csv.reader(file)
                for i, row in enumerate(csvreader):
                    if i==0:
                        continue
                    self.valid_label[row[1]] = label[row[2]]
        else: 
            self.id = []
            self.images = []
            with open(test_csv_path, 'r') as file:
                csvreader = csv.reader(file)
                for i, row in enumerate(csvreader):
                    if i==0:
                        continue
                    self.id.append(row[0])
                    self.images.append(row[1])
    def __len__(self):
        return len(self.images)
  
    def __getitem__(self,index):
        if self.train_set or self.valid_set:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            if self.train_set:
                label = self.train_label[image_name.split('/')[-1]]
            elif self.valid_set:
                label = self.valid_label[image_name.split('/')[-1]]
                
            return image, label
        else:
            image_name = self.images[index]
            image = Image.open(os.path.join(self.data_path,image_name))
            if self.transform is not None:
                image = self.transform(image)
            image_name = image_name.split('/')[-1]
            id = self.id[index]

            return image, image_name, id