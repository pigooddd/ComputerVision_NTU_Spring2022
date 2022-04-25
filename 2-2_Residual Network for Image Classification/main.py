
import torch
import os


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet, myModel, myClean
from myDatasets import  get_cifar10_train_val_set, get_cleaning_set
from tool import train, fixed_seed, clean

# Modify config if you are conducting different models
from cfg import LeNet_cfg as cfg


def train_interface():
    
    """ input argumnet """

    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'): #建立並清空.log檔，唯寫模式
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Computer has NVIDIA GPU: ", torch.cuda.is_available())
    print("device:", device)
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    #model = myLeNet(num_out=num_out)
    #model = myResnet(num_out=num_out)
    model = myModel(num_out=num_out)
    # print model's architecture
    # print(model)

    # Get your training Data 
    ## TO DO ##
    #data cleaning

    cleaning_model = myClean(num_out=num_out)
    print("cleaning model: ")
    print(cleaning_model, '\n', '==========================================', '\n')
    cleaning_set, info= get_cleaning_set(root=data_root, ratio=split_ratio)
    cleaning_loader = DataLoader(cleaning_set, batch_size=256, shuffle=True)
    cleaning_val_loader = DataLoader(cleaning_set, batch_size=1, shuffle=False)
    cleaning_optimizer = optim.SGD(cleaning_model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-6, nesterov=True)
    cleaning_scheduler = optim.lr_scheduler.MultiStepLR(cleaning_optimizer,milestones=milestones, gamma=0.1)
    cleaning_criterion = nn.CrossEntropyLoss()
    cleaning_model = cleaning_model.to(device)
    clean_bool_array = clean(model=cleaning_model, train_loader=cleaning_loader, val_loader=cleaning_val_loader, 
          num_epoch=6, log_path=log_path, save_path=save_path,
          device=device, criterion=cleaning_criterion, optimizer=cleaning_optimizer, scheduler=cleaning_scheduler, cleaning_set=cleaning_set)
    #用cleaning_set, clean_bool_array, info做事
    info = info[clean_bool_array]
    # print(info.shape)

    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
    train_set, val_set= get_cifar10_train_val_set(root=data_root, info=info, ratio=split_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
     
    # define your loss function and optimizer to unpdate the model's parameters.
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    
if __name__ == '__main__':
    train_interface()




    