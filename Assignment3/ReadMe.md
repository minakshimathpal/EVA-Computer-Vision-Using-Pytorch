
## Part1: BackPropogation using Excel
### 1.1 Network Architecture
The document captures the experience of creating a backpropogation using an excel sheet. The Neural network looks like the below. Our Network excepts two inputs,has a hidden layer and an output layer. The sigmoid function has been used as an Activation Function.

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Network%20Architecture.PNG?raw=True)

### 1.2 Steps:
1. forward Pass
2. Calculate the error:  Calculate the error for each output neuron using the squared error function and sum them to get the total error:
   <br> E_total =1/2(target - output)^2
4. Backward Pass: Our goal with backpropagation is to update each of the weights in the network so that they cause the actual output to be closer the target output, thereby minimizing the error for each output neuron and the network as a whole.

### 1.3 Backpropogation using Chain Rule:
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Backward_Pass.PNG?raw=True)

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Hidden_Layer_Gradient_Propogation.PNG?raw=True)

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Formula_1.PNG?raw=True)
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/formula_2.PNG?raw=True)
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Formula_3.PNG?raw=True)

### 1.4 Error Graph with different Learning rates
with Small Learning convergance is slow. Infact it is not necessary that you will converge in  the given number of epochs. thus with small Learning rates training will be slow. As the error rates are increasing the we can see that we start to converge faster.

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/Learning_Rates.PNG?raw=True)


## Part-2 MNIST Training and Validation Datasets

While building real-world machine learning models, it is quite common to split the dataset into three parts:

1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using gradient descent.
2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc.), and pick the best version of the model.
3. **Test set** - used to compare different models or approaches and report the model's final accuracy.

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images. 

Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for testing. We can do this using the `random_spilt` method from PyTorch.

A general "sequential" neural network can be expressed as

$$f(x) = \underset{i=1}{\overset{n}{\Huge{\kappa}}} R_i(A_ix+b_i)$$

where $$\underset{i=1}{\overset{n}{\Huge{\kappa}}}f_i(x) = f_n \circ f_{n-1} ... \circ f_1(x)$$ and the $A_i$ are weight matrices and the $b_i$ are bias vectors. Typically the $R_i$(Activation function) are the same for all the layers (typically ReLU) **except** for the last layer, where $R_i$ is just is just the identity function

## Building a CNN model from scratch
Our network architecture will contain a combination of different layers, namely:

    Conv2d
    MaxPool2d
    Rectified linear unit (ReLU)
    BatchNormalization Layer
    Dropout2d
    
## Feature Extractor & Classifier Combo

Many classical convolutional neural networks are actually a combination of convnets and MLPs. Looking at the architectures of LeNet and AlexNet for instance, one can distinctively see that their architectures are just a couple of convolution layers with linear layers attached at the end.

This configuration makes a lot of sense, it allowed the convolution layers to do what they do best which is extracting features in data with two spatial dimensions. Afterwards the extracted features are passed onto linear layers so they also can do what they are great at, finding relationships between feature vectors and targets.

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Alexnet_architecture.PNG?raw=true)

## A Flaw in the Design

The problem with this design is that linear layers have a very high propensity to overfit to data. Dropout regularization was introduced to help mitigate this problem but a problem it remained nonetheless. Furthermore, for a neural network which prides itself on not destroying spatial structures, the classical convnet still did it anyway, albeit deeper into the network and to a lesser degree.

## Modern Solutions to a Classical Problem

In order to prevent this overfitting issue in convnets, the logical next step after trying dropout regularization was to completely get rid of the linear layers all together. If the linear layers are to be excluded, an entirely new way of down-sampling feature maps and producing a vector representation of equal size to the number of classes in question is to be sought. This exactly is where global pooling comes in.

## Global Average Pooling

Imagine a scenario where we feel our convolution layers are at an adequate depth but we have 8 feature maps of size (3, 3). We can utilize a 1 x 1 convolution layer in order to down-sample the 8 feature maps to 4. Now we have 4 matrices of size (3, 3) when what we actually need is a vector of 4 elements.

One way to derive a 4 element vector from these feature maps is to compute the average of all pixels in each feature map and return that as a single element. This is essentially what global average pooling entails.

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Global_average_poolin.PNG?raw=true)
    
## Model Architecture Used

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Model_architecture.PNG?raw=true)

There are total ```19,236``` trainable parameters. 

## Training logs
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/Assignment3/artifacts/training_logs.PNG)

## Accuracy and Loss Plots
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Accuracy_plot.PNG?raw=true)
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Loss_plot.PNG?raw=true)
=======
>>>>>>> 5c1d781 (Assignment 3 Added)

