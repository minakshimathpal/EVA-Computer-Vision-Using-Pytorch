
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations
import albumentations.pytorch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import albumentations
import albumentations.pytorch
# from utils.helper_dataset import get_dataloader_cifar10

class Cifar10Dataset(torchvision.datasets.CIFAR10):
  def __init__(self,root="./data",train=True,download=True,transform=None):
    super().__init__(root=root,train=train,download=download,transform=transform)

  def __getitem__(self,index):
    image, label= self.data[index],self.targets[index]
    
    # apply the transformations
    if self.transform is not None :
      transformed = self.transform(image=image) 
      image=transformed["image"]

      return (image, label)

def get_dataloader_cifar10(dataloader_args,
                           validation_fraction=None,
                           train_transform=None,
                           test_transform=None,
                           ):
  
  train_set= Cifar10Dataset(root="./data",train=True,download=True,transform=train_transform)
  valid_set= Cifar10Dataset(root="./data",train=True,download=True,transform=test_transform)
  test_set=  Cifar10Dataset(root="./data",train=False,download=True,transform=test_transform)
  
  if validation_fraction is not None:
      num= int(validation_fraction * 50000)
      train_indices= torch.arange(0,50000 - num)
      valid_indices= torch.arange(50000 - num,50000)

      train_sampler = SubsetRandomSampler(train_indices)
      valid_sampler = SubsetRandomSampler(valid_indices) 

      valid_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                **dataloader_args,
                                                sampler=valid_sampler)
                                               
                                                
      
      train_loader = torch.utils.data.DataLoader(dataset= train_set,
                                                **dataloader_args,                                                
                                                sampler=train_sampler)                                     
                                                
  else:
      print("inside else ")
      train_loader = torch.utils.data.DataLoader(dataset=train_set, **dataloader_args)

  
  test_loader = DataLoader(dataset=test_set,
                           **dataloader_args)
                           
  

  if validation_fraction is None:
      
        return train_loader, test_loader
  else:
        return train_loader, valid_loader, test_loader

def compute_accuracy(model,data_loader,device):
  with torch.no_grad():
    correct_preds,num_examples=0,0

    for i, (images,targets) in enumerate(data_loader):
      images,targets= images.to(device),targets.to(device)
      logits= model(images)

      predicted_labels= logits.argmax(dim=1,keepdims=True)# get the index of the max log-probability
      correct_preds+=(predicted_labels ==targets).sum()
    return correct_preds.float()/len(data_loader.dataset)



