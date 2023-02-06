### Objective:
1) Run this network Links to an external site..
2) Fix the network above:
   1) change the code such that it uses GPU and
   2) change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here           instead of MP or strided convolution, then 200pts extra!)
   3) total RF must be more than 44
   4) one of the layers must use Depthwise Separable Convolution
   5) one of the layers must use Dilated Convolution
   6) use GAP (compulsory):- add FC after GAP to target #of classes (optional)
   7) use albumentation library and apply:
      1) horizontal flip
      2) shiftScaleRotate
      3) coarseDropout ```(max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset),                mask_fill_value = None)```
   8) achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
   9) upload to Github
   10) Attempt S6-Assignment Solution.
   11) Questions in the Assignment QnA are:
       1) copy paste your model code from your model.py file (full code) [125]
       2) copy paste output of torchsummary [125]
       3) copy-paste the code where you implemented albumentation transformation for all three transformations [125]
       4) copy paste your training log (you must be running validation/text after each Epoch [125]
       5) Share the link for your README.md file. [200]

### Key Idea: Replacing Maxpooling by strided convolutions(i.e convolution layer with a stride of 2 
        `we can think of strided convolutions as learnable pooling`
Replacing maxpoling with convolution layer with stride of 2 will be a little bit more expensive, because we have now more parameters. 
But in that way, we can also simplify the network in terms of having it look simpler, just saying,okay, we only use convolutions, we don't use anything else, if
that's desirable. We still need the activation function, but we don't need pooling layers, for example. I
 will be a bit costly operation but 
 
 ### Model Architecture
 1) Four convolution blocks
    - first block: | **Reciptive Field** =13
      - 3 dilated convolution with kernel size=3,stride=1, padding=2,dilation=2 
    - Second Block: | **Reciptive Field** =33
      - 1 dilated convolution with kernel size=3,stride=2,dilation=2,padding=1
      - 2 depthwise separable convoltion with kernel size=3,dilation=2,stride=1,padding=2
    - third block: | **Reciptive Field** = 57
      - 1 dilated convolution with kernel size=3,stride=1,dilation=2,padding=1
      - 2 depthwise separable kernel with kernel_size=3,stride=1,padding=1
    - Fourth Block: | **Reciptive Field** = 121
      - 1 dilated convolution with kernel size=3,dilation=2,stride=2,padding=2
      - 1 depthwise separable convoltion with kernel size=3,stride=1,padding=1
      - 1 dilated convoltion with kernel_size=3,dilation=2,stride=1,padding=2
      - 1 Normal Convolution kernel_size=(3,3),stride=(1,1),padding=1
  2) GlobalAveragePooling Layer
  3) Normal Convolution with kernel_size=(3,3),stride=(1,1),padding=1   
    
     
    
        
