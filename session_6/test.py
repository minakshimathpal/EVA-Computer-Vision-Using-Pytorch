# -*- coding: utf-8 -*-
"""test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Lsm4l5NGBjIND7GdAChlBFGyw4KBtFpX
"""

import torch.nn.Functional as F

class Test:
  def __init__(self,model,dataloader,optimizer,device):

    self.model=model
    self.device=device
    self.dataloader=dataloader
    self.optimizer=optimizer
    self.test_epoch_loss=[]
    self.test_epoch_accuracy=[]
    
  def test(data_loader,model,optimizer,device):
    model.eval()
    progress_bar= tqdm(data_loader)
    running_loss=0.0
    running_correct= 0

    with torch.no_grad():
      for batch_idx,(images,targets) in enumerate(progress_bar):
      # get samples
        images,targets = images.to(device),targets.to(device)

        # forward pass/prediction
        outputs= model(images)
#
        # Calculate loss
        loss= F.nll_loss(outputs,targets)

      # accumulate loss of every batch
        running_loss+=loss.item() # self.train_losses.append(loss.item()) I am doinf it at epochs

        predicted_labels= outputs.argmax(dim=1,keepdims=True)
        running_correct+=predicted_labels.eq(targets.view_as(predicted_labels)).sum().item() 

    loss= running_loss/len(data_loader.dataset)
    accuracy=100*running_correct/len(data_loader.dataset)

    self.test_epoch_loss.append(loss)
    self.test_epoch_accuracy.append(accuracy)
    
    print(f"Validation Loss {loss} |  Validation Accuracy {accuracy:.2f}")

    return loss,accuracy