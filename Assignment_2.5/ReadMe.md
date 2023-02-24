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
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/target_column.PNG?raw=true)

### Model Architecture
We need one convolutional neural network for our image data and a multi-layer perceptron for our tabular data. Both need to be combined and need to return a single prediction value. First, we define a single conv_block, a small custom CNN-layer that expects an input and output dimension. This will save us a bit of repetitive code writing later on.
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/Model_architecture.PNG?raw=true)

### Loss Function
We create a flexible training routine that takes into account all outputs of our model. Therefore, it does not matter whether we have 2, 3 or, for example, 5 classifier heads. We simply use the conventional loss function for multiclassification tasks. We calculate the CrossEntropyLoss for digit classifier and for regression we need MSE and then we sum the Losses. This way we can optimize the weights with a single optimizer step for both the heads

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/loss_function.PNG?raw=true)

### Training Logs

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/training_logs.PNG?raw=true)

### Evaluation
Accuracy and R2 value is used to evaluate the performance of the model
![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/Evaluation.PNG?raw=true)

![alt text](https://github.com/minakshimathpal/EVA-Computer-Vision-Using-Pytorch/blob/main/artifacts/confusion_matrix.PNG?raw=true)


