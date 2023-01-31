### Objective 
Your 5th Assignment is:
- You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):

        Network with Group Normalization
        Network with Layer Normalization
        Network with L1 + BN
- You MUST:

        Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
        Write a single notebook file to run all the 3 models above for 20 epochs each
        Create these graphs:
        
            Graph 1: Test/Validation Loss for all 3 models together
            Graph 2: Test/Validation Accuracy for 3 models together
            graphs must have proper annotation
- Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 
- write an explanatory README file that explains:
            what is your code all about,
            how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)
            your findings for normalization techniques,
            add all your graphs
            your 3 collection-of-misclassified-images 
 - Upload your complete assignment on GitHub and share the link on LMS
 
 ### Strategy Details
 The objective of this assignment is to get well versed with three types of Normalization.
 - Batch Normalization(with L1 regularization)
 - Layer Normalization
 - Group normalization
 ### Normalization
 Normalization is a kind of preprocessing strategy used in Machine Learning and Deep Learning where the weights/parameters of the model/network are scaled to be in a range of 0-1.Normalizing our inputs aims to create a set of features that are on the same scale as each other.For machine learning models, our goal is usually to recenter and rescale our data such that is between 0 and 1 or -1 and 1, depending on the data itself.Normalization can help training of our neural networks as the different features are on a similar scale, which helps to stabilize the gradient descent step, allowing us to use larger learning rates or help models converge faster for a given learning rate.

 


