
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torchvision.models as models

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        
        self.nrb1 = no_residual_block(3)
                                                  #3*32*32 
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),       
                             nn.ReLU(),                #6*28*28
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.nrb2 = no_residual_block(6)
                                                  #6*14*14
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),                #16*9*9
                             nn.MaxPool2d(kernel_size=2, stride=2),)
                                                  #16*5*5
        self.nrb3 = no_residual_block(16)
                                                  # = 400
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())            #120
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())              #84
        self.fc3 = nn.Linear(84,num_out)                         #10
        

        # self.conv = nn.Sequential(
        #     # 3*32*32 => 16*14*14
        #     nn.Conv2d(3,16,kernel_size=5, stride=1),                      
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 16*14*14 => 32*5*5
        #     nn.Conv2d(16,32,kernel_size=5),               
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 32*5*5 => 64*3*3
        #     nn.Conv2d(32,64,kernel_size=3),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),  
            


        # )

        # self.fc = nn.Sequential(
        #     # 576 => 300
        #     #nn.Dropout(),
        #     nn.Linear(576, 300), 
        #     nn.ReLU(),
        #     # 300 => 100
        #     #nn.Dropout(),
        #     nn.Linear(300, 100), 
        #     nn.ReLU(),
        #     # 100 => 10 (num_out)
        #     nn.Linear(100,num_out),
        # )

    def forward(self, x):
        
        x = self.nrb1(x)
        x = self.conv1(x)
        x = self.nrb2(x)
        x = self.conv2(x)
        x = self.nrb3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)    
           
        # x = self.conv(x)
        # #print(x.size())
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        # #print(x.size())
        # x = self.fc(x)

        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        residual = x
        x = self.conv1(x)
        x = x + residual
        x = self.bn(x)
        x = self.relu(x)
        out = x
        return out

class no_residual_block(nn.Module):
    def __init__(self, in_channels):
        super(no_residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU()
        
    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)
        out = x
        return out

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        # self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.

        # self.conv = nn.Sequential(
        #     # 3*32*32 => 16*14*14
        #     nn.Conv2d(3,16,kernel_size=5, stride=1),                      
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 16*14*14 => 32*5*5
        #     nn.Conv2d(16,32,kernel_size=5),               
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 32*5*5 => 64*3*3
        #     nn.Conv2d(32,64,kernel_size=3),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),  
        # )
        # self.rb = residual_block(64)
        # self.fc = nn.Sequential(
        #     # 576 => 300
        #     #nn.Dropout(),
        #     nn.Linear(576, 300), 
        #     nn.ReLU(),
        #     # 300 => 100
        #     #nn.Dropout(),
        #     nn.Linear(300, 100), 
        #     nn.ReLU(),
        #     # 100 => 10 (num_out)
        #     nn.Linear(100,num_out),
        # )

        self.rb1 = residual_block(3)
                                                  #3*32*32 
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),       
                             nn.ReLU(),                #6*28*28
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.rb2 = residual_block(6)
                                                  #6*14*14
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),                #16*9*9
                             nn.MaxPool2d(kernel_size=2, stride=2),)
                                                  #16*5*5
        self.rb3 = residual_block(16)
                                                  # = 400
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())            #120
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())              #84
        self.fc3 = nn.Linear(84,num_out)                         #10
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)

        # x = self.conv(x)
        # x = self.rb(x)
        # #print(x.size())
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        # #print(x.size())
        # x = self.fc(x)
        
        x = self.rb1(x)
        x = self.conv1(x)
        x = self.rb2(x)
        x = self.conv2(x)
        x = self.rb3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)    

        out = x
        return out

class myModel(nn.Module):
    def __init__(self, num_out):
        super(myModel, self).__init__()

        # self.conv = nn.Sequential(
        #     # 3*32*32 => 16*14*14
        #     nn.Conv2d(3,16,kernel_size=5, stride=1),                      
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 16*14*14 => 32*5*5
        #     nn.Conv2d(16,32,kernel_size=5),               
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     # 32*5*5 => 64*3*3
        #     nn.Conv2d(32,64,kernel_size=3),
        #     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),  
            


        # )

        # self.fc = nn.Sequential(
        #     # 576 => 300
        #     #nn.Dropout(),
        #     nn.Linear(576, 300), 
        #     nn.ReLU(),
        #     # 300 => 100
        #     #nn.Dropout(),
        #     nn.Linear(300, 100), 
        #     nn.ReLU(),
        #     # 100 => 10 (num_out)
        #     nn.Linear(100,num_out),
        # )
        self.res50 = models.resnet50(pretrained=True, progress=True)
        

    def forward(self, x):
         
        # x = self.conv(x)
        # #print(x.size())
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        # #print(x.size())
        # x = self.fc(x)
        x = self.res50(x)
        out = x
        return out


class myClean(nn.Module):
    def __init__(self, num_out):
        super(myClean, self).__init__()
        self.res50 = models.resnet50(pretrained=False, progress=True)

    def forward(self, x):
        x = self.res50(x)
        out = x
        return out