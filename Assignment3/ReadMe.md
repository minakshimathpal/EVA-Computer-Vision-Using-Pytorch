
## Training and Validation Datasets

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
    View
    Linear layer
    
## Feature Extractor & Classifier Combo

Many classical convolutional neural networks are actually a combination of convnets and MLPs. Looking at the architectures of LeNet and AlexNet for instance, one can distinctively see that their architectures are just a couple of convolution layers with linear layers attached at the end.

This configuration makes a lot of sense, it allowed the convolution layers to do what they do best which is extracting features in data with two spatial dimensions. Afterwards the extracted features are passed onto linear layers so they also can do what they are great at, finding relationships between feature vectors and targets.

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/master/Assignment3/artifacts/Alexnet_architecture.PNG?raw=true)

## A Flaw in the Design

The problem with this design is that linear layers have a very high propensity to overfit to data. Dropout regularization was introduced to help mitigate this problem but a problem it remained nonetheless. Furthermore, for a neural network which prides itself on not destroying spatial structures, the classical convnet still did it anyway, albeit deeper into the network and to a lesser degree.

## Modern Solutions to a Classical Problem

In order to prevent this overfitting issue in convnets, the logical next step after trying dropout regularization was to completely get rid of the linear layers all together. If the linear layers are to be excluded, an entirely new way of down-sampling feature maps and producing a vector representation of equal size to the number of classes in question is to be sought. This exactly is where global pooling comes in.

## Global Average Pooling

Still on the same classification task described above, imagine a scenario where we feel our convolution layers are at an adequate depth but we have 8 feature maps of size (3, 3). We can utilize a 1 x 1 convolution layer in order to down-sample the 8 feature maps to 4. Now we have 4 matrices of size (3, 3) when what we actually need is a vector of 4 elements.

One way to derive a 4 element vector from these feature maps is to compute the average of all pixels in each feature map and return that as a single element. This is essentially what global average pooling entails.
    


