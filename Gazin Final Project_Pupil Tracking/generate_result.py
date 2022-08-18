import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import random
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.transforms.functional as TF
import cv2

from tqdm.auto import tqdm

import os
root = "./dataset/test"
datas = {}
for series in tqdm(os.listdir(root)):
    print(series)
    for sequence in tqdm(os.listdir(f"{root}/{series}")):
        # print(sequence)
        for file in os.listdir(f"{root}/{series}/{sequence}"):

            if file.endswith(".jpg"):
                if series not in datas:
                    datas[series] = {}
                if sequence not in datas[series]:
                    datas[series][sequence] = []
                datas[series][sequence].append(f"{root}/{series}/{sequence}/{file}")


tfm = transforms.Compose([transforms.ToTensor()])
concat_n = 5

class GanzinDataset(Dataset) :
    def __init__(self, datas, targets, transform=tfm, device="cuda:0", concat_n=concat_n, oversample=1, bc_label=False) :
        self.datas = datas
        self.targets = targets
        self.transform = transform
        self.concat_n = concat_n
        self.oversample = oversample
        self.bc_label = bc_label

    def __len__(self) :
        return len(self.datas*self.oversample)

    def transforms(self, images, mask):
        # Random Resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            mask, scale=(0.8, 1.0), ratio=(0.8, 1.25))
        images = [TF.crop(image, i, j, h, w) for image in images]
        mask = TF.crop(mask, i, j, h, w)

        resize = transforms.Resize(size=(480, 480))
        images = [resize(image) for image in images]
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            images = [TF.hflip(image) for image in images]
            mask = TF.hflip(mask)


        # Transform to tensor
        images = torch.cat([TF.to_tensor(image) for image in images], dim=0)
        mask = TF.to_tensor(mask)
        return images, mask[0].unsqueeze(0)

    def __getitem__(self, idx) :
        data = []
        idx = idx % len(self.datas)
        for i in range(idx-self.concat_n//2, idx+self.concat_n//2+1) :
            if i < 0 :
                i = 0
            if i >= len(self.datas) :
                i = len(self.datas)-1
            data.append(Image.open(self.datas[i]))
        if self.bc_label :
          target = self.targets[idx]
          return torch.cat([self.transform(d) for d in data], dim = 0), target
        else :
          target = Image.open(self.targets[idx])
          return self.transforms(data, target)


from model import *
device = "cuda:0"
model = conv_autoencoder().to(device)
model.load_state_dict(torch.load("./model_best.pt"))
model.eval()
model.requires_grad_(False)


threshold = 128
x, y = np.meshgrid(np.arange(640), np.arange(480))

import cv2.ximgproc as xip
def post_process(out, img) :
    mask = (out>threshold)
    center = np.array([np.mean(x[mask]), np.mean(y[mask])])
    r = np.sqrt(np.sum(mask))
    out[(x-center[0])**2 + (y-center[1])**2 > r**2] = 0
    out = xip.weightedMedianFilter(img.astype(np.uint8), out.astype(np.uint8), r=3)
    return out



def preprocess(img, ks=9):
    #maxpool = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=ks//2)
    return img#-maxpool(-maxpool(1.0*img))

bs = 50
zeros = np.zeros((480, 640)).astype(np.uint8)
for series, sequences in tqdm(datas.items()) :
    for key, seq in tqdm(sequences.items()):
        dataset = GanzinDataset(seq, [0]*len(seq), bc_label=True) 
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        os.makedirs(f"./S5_solution/{key}/", exist_ok=True)
        for i, (img, _) in enumerate(dataloader) :
            img = img.to(device)
            outs = (model(img).detach().cpu())
            for j, (out, img) in enumerate(zip(outs, img[:, 2, :, :].cpu().numpy()*255)) :
                out = (preprocess(out).numpy()*255).astype(np.uint8)
                out = post_process(out.reshape(480, 640), img.reshape(480, 640))
                im = Image.fromarray(np.stack([out, zeros, out], axis=0).transpose((1, 2, 0)))
                im.save(f"{datas[series][key][i*bs+j]}".replace(".jpg", ".png").replace("./dataset/test/S5", "S5_solution"))

def get_avg(data_path, root):
    data_len = len(data_path)
    area = np.zeros(shape=data_len)
    for i in range(data_len):
        print(root + str(i) + ".png")
        img = cv2.imread(root + str(i) + ".png")[:, :, 0]
        area[i] = np.sum(img>128) * 255
    avg_area = np.sum(area) / data_len
    for i in range(data_len):
        if area[i] < avg_area * 0.5:
            area[i] = 0
    return area, avg_area

def filter(area, avg_area):
    window_len = 5
    area_len = area.shape[0]
    result = np.zeros(shape = area_len)
    for i in range(window_len//2):
        if(area[i] > avg_area):
            result[i] = 1.0
        if(area[-i-1] > avg_area):
            result[-i-1] = 1.0
    for i in range(window_len // 2, area_len - window_len // 2):
        window = area[i-window_len//2:i+window_len//2+1]
        local_max = np.max(window)
        if(area[i] > local_max * 0.5):
            result[i] = 1.0
    return result


def median_post_process(result, txt_path):
    result_len = result.shape[0]
    window_len = 5
    copy_data = np.zeros(shape = result_len+2*(window_len//2))
    for i in range(result_len):
        copy_data[i+window_len//2] = result[i]
    for i in range(window_len//2):
        w_l = window_len//2 - i
        copy_data[i] = copy_data[i+w_l*2]
        copy_data[-i-1] = copy_data[(-i-1)-w_l*2]

    result_len = copy_data.shape[0]

    for i in range(window_len//2, result_len-window_len//2):
        window = copy_data[i-window_len//2:i+window_len//2+1]
        result[i-window_len//2] = np.median(window)

    with open(txt_path, 'w') as f:
        for cnt, item in enumerate(result):
            f.write(str(item)+"\n")


root = './S5_solution'
data_path = {}
for sequence in tqdm(os.listdir(f"{root}")):
    for file in os.listdir(f"{root}/{sequence}"):
        if file.endswith(".png"):
            if sequence not in data_path:
                data_path[sequence] = []
            data_path[sequence].append(f"{root}/{sequence}/{file}")
    area, avg_area = get_avg(data_path[sequence], f"{root}/{sequence}/")
    result = filter(area, avg_area)
    median_post_process(result, f"{root}/{sequence}/conf.txt")