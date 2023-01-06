## Training and Validation Datasets

While building real-world machine learning models, it is quite common to split the dataset into three parts:

1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using gradient descent.
2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc.), and pick the best version of the model.
3. **Test set** - used to compare different models or approaches and report the model's final accuracy.

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images. 

Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for testing. We can do this using the `random_spilt` method from PyTorch.

### Random number Generation
I have approached the problem as multi label problem. Idea is to create a target column as a list where first element of the list is image label and the second elemnet of the list is summation of randomm number and the target label. Thus the target will be of ```size(n_samples,2)```
This way we will get the ***ground truth for training a regressor along with a classifier***. So our model will have two output heads one for classification and one for regression.


1. Random number is generated using python's random module using random.randint
2. After that a target column is created by adding image label from Mnist dataset and the generated random number
3. ```__getitem__``` returns a dictionary of image and all the associated labels.This dictionary has image, numeric data which is a list of image label and random integer and target which is also a dictionary with image label and sum of the image label and random integer
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/data_representation.PNG?raw=true)
