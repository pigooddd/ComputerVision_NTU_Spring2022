# Import necessary packages.
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import random
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

"""Preparing for phase 1 training. """

import os


random.seed(0)

data_path = "./dataset/public"
def sort_data(data_path=data_path) :
    datas_1 = []
    targets_1 = []
    datas_0 = []
    targets_0 = []

    count_1 = 0
    count_0 = 0
    for series in tqdm(os.listdir(data_path)) :
        for sequence in tqdm(os.listdir(data_path + "/" + series)) :
            seq_d_1 = []
            seq_t_1 = []
            seq_d_0 = []
            seq_t_0 = []
            for file in os.listdir(data_path + "/" + series + "/" + sequence) :
                if file.endswith(".jpg") :
                    continue
                fullpath = data_path + "/" + series + "/" + sequence + "/" + file
                with Image.open(fullpath) as f :
                    if (np.array(f)[:, :, 0]==255).any() :    
                        seq_t_1.append(fullpath)
                        seq_d_1.append(fullpath.replace("png", "jpg"))
                        count_1 += 1
                    else :
                        seq_t_0.append(fullpath)
                        seq_d_0.append(fullpath.replace("png", "jpg"))
                        datas_1.append(seq_d_1)
                        targets_1.append(seq_t_1)
                        seq_t_1 = []
                        seq_d_1 = []
                        count_0 += 1
                
            datas_1.append(seq_d_1)
            targets_1.append(seq_t_1)
            datas_0.append(seq_d_0)
            targets_0.append(seq_t_0)
    print(count_1, count_0)
    return datas_1, targets_1, datas_0, targets_0

import pickle as pkl
meta_path = "./metadata.pkl"
try : 
    with open(meta_path, "rb") as f :
        datas_1, targets_1, datas_0, targets_0 = pkl.load(f)
except :
    datas_1, targets_1, datas_0, targets_0 = sort_data()
    pkl.dump((datas_1, targets_1, datas_0, targets_0), open(meta_path, "wb"))

tfm = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1.1), shear=(6, 9)),
              transforms.ToTensor()])
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

subsets = [GanzinDataset(datas, targets) for datas, targets in zip(datas_1, targets_1)]
dataset = ConcatDataset(subsets)

from model import *

num_epochs = 40
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

model = conv_autoencoder(concat_n=concat_n).to(device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
optimizer = RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
lamb = 20
alpha = 0.01
RE = nn.MSELoss()

best_loss = float("inf")
for epoch in range(num_epochs):
    avg_loss = []
    #avg_BCE = []
    avg_RE = []
    for i, data in enumerate(tqdm(dataloader)):
        img, y = data
        img = img.to(device)
        # m = torch.max(y.flatten(start_dim=1), dim=1)[0]
        optimizer.zero_grad()
        #output, confidence = model(img)
        output = model(img)
        # confidence = confidence.view(-1).cpu()
        RELoss = RE(output.cpu(), y)
        #BCELoss = BCE(confidence[m==1], m[m==1]) + lamb*BCE(confidence[m==0], m[m==0])
        loss = RELoss# + BCELoss
        loss.backward()
        optimizer.step()
        
        avg_loss.append(loss.item())
        avg_RE.append(RELoss.item())
        #avg_BCE.append(BCELoss.item())

    scheduler.step(np.mean(avg_loss))
    if np.mean(avg_loss) < best_loss :
        best_loss = np.mean(avg_loss)
        torch.save(model.state_dict(), "./model_best.pt")
    print("epoch: {}, loss: {}".format(epoch, sum(avg_loss)/len(avg_loss)))
    print(f"best loss: {best_loss}")