import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from PIL import Image
import json 


def get_cifar10_train_val_set(root, info, ratio=0.9):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    # info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # # apply shuffle to generate random results 
    # np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x] #從0拿到x當train
    val_image = all_images[x:] #從x拿到結束當val

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    # https://blog.csdn.net/qq_42951560/article/details/109852790
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                ## TO DO ##
                # You can add some transforms here
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                transforms.RandomAffine(degrees=5, translate=(0.1,0.1), scale=(0.9,1.1), shear=None),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.07),
                transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.05),
                # transforms.RandomCrop(25),
                # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
                transforms.ToTensor(),
                
                # experimental normalization for image classification 
                transforms.Normalize(means, stds),
            ])
    data_augmentation_transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=1),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                #transforms.RandomRotation(degrees=(-70, 70)),
                transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 3)),
                transforms.RandomAffine(degrees=70, translate=(0.2,0.2), scale=(0.7,1.3), shear=None),
                # transforms.RandomCrop(25),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    data_augmentation_transform2 = transforms.Compose([
                #transforms.RandomHorizontalFlip(p=1),
                #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                ##transforms.RandomRotation(degrees=(-70, 70)),
                transforms.RandomAffine(degrees=50, translate=(0.2,0.2), scale=(0.7,1.3), shear=None),
                transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    # data_augmentation_transform3 = transforms.Compose([
    #             #https://blog.csdn.net/flyfish1986/article/details/108831332
    #             #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.0),
    #             transforms.ToTensor(),
    #             transforms.Normalize(means, stds),
    #         ])
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform)
                        
    # data augmentation 
    aug_set1 = cifar10_dataset(images=train_image[:(len(train_image)//3)*1], labels=train_label[:(len(train_image)//3)*1],transform=data_augmentation_transform1)
    aug_set2 = cifar10_dataset(images=train_image[(len(train_image)//3)*1:(len(train_image)//3)*2], labels=train_label[(len(train_image)//3)*1:(len(train_image)//3)*2],transform=data_augmentation_transform2)
    # aug_set3 = cifar10_dataset(images=train_image[(len(train_image)//5)*3:(len(train_image)//5)*5], labels=train_label[(len(train_image)//5)*3:(len(train_image)//5)*5],transform=data_augmentation_transform3)
    train_set = torch.utils.data.ConcatDataset([train_set, aug_set1, aug_set2])
    print(f'Number of train images after augmentation is {len(train_set)}')

    return train_set, val_set



## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        label = self.labels[idx]
        image = Image.open(os.path.join(self.prefix, self.images[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # You shall return image, label with type "long tensor" if it's training set
        return image, label
        
## TO DO ##
# Define your own cleaning dataset
def get_cleaning_set(root, ratio=0.9, cv=0):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()
    
    ## TO DO ## 
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    ## TO DO ##
    # Complete class cifiar10_dataset
    cleaning_set = cifar10_dataset(images=all_images, labels=all_labels,transform=train_transform)

    return cleaning_set, info
