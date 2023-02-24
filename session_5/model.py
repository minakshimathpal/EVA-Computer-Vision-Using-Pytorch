import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

seed=42
torch.manual_seed(seed)


# dropout_value=0.05

class Net(nn.Module):
  def __init__(self,norm_type="BN",n_groups=0,dropout_value=0.05):
    super(Net,self).__init__()
    self.norm_type=norm_type
    self.n_groups=n_groups
    self.dropout_value=dropout_value

    # Input Block
    self.convblock1= nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding=0,bias=False),       
        nn.ReLU(),
        self.normalize(10),
        nn.Dropout(self.dropout_value)
        ) # output_size=26 | 3

    # CONVOLUTION BLOCK 1
    self.convblock2= nn.Sequential(
        nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=False),
        nn.ReLU(),
        self.normalize(20),
        nn.Dropout(self.dropout_value)
    ) # output_size=24 | 5

    # TRANSITION BLOCK 1
    self.trans1= nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1),padding=0,bias=False),
        nn.MaxPool2d(2,2) # output_size=11 | 14        
        ) # output_size=22 | 7    
    
    self.convblock4= nn.Sequential(
        nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=False),
        nn.ReLU(),
        self.normalize(20),
        nn.Dropout(self.dropout_value)
    ) # output_size= 11 | 16

    # CONVOLUTION BLOCK 2
    self.convblock5= nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=False),
        nn.ReLU(),
        self.normalize(10),
        nn.Dropout(self.dropout_value)
     ) 
    # output_size=9 | 18
    self.convblock6= nn.Sequential(
        nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=False),
        nn.ReLU(),
        self.normalize(20),
        nn.Dropout(self.dropout_value)
    ) # output_size=7 |20

    # OUTPUT BLOCK 
    self.convblock7 = nn.Sequential(
        nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1),padding=1,bias=False),
        nn.ReLU(),
        self.normalize(10),
        nn.Dropout(self.dropout_value) 
    ) # output_size= 7 | 20
    # OUTPUT BLOCK
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=8)
    ) # output_size = 1

    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        # nn.BatchNorm2d(10),
        # nn.ReLU(),
        # nn.Dropout(dropout_value)
                      )         
  def forward(self,x):
    
    x= self.convblock1(x)
    x= self.convblock2(x)
    x= self.trans1(x)
    x= self.convblock4(x)
    x= self.convblock5(x)
    x= self.convblock6(x)
    x= self.convblock7(x)
    x = self.gap(x)
    x= self.convblock8(x)
    x= x.view(-1,10)
    return F.log_softmax(x,dim=-1) 

  def normalize(self,ch_out):
    if self.norm_type=="BN" :
      return nn.BatchNorm2d(ch_out)
    elif (self.norm_type=="GN") or (self.norm_type=="LN"):
      return nn.GroupNorm(self.n_groups,ch_out)  
    else:
      print("Please Enter a valid Normalization type(BN/GN/LN)") 

  def compute_l1_loss(self, w):
      return torch.abs(w).sum()    
