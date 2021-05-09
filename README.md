Convolution-Neural-Network
==========================

# Image Classification model using Convolution Neural Network using dataset from Kaggle #

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

Credit - [CNN](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) <br>

![Representation of Convolution Neural network model on Images ](https://miro.medium.com/max/4800/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

                                         *Representation of Convolution Neural network model on Images*

<br>

- Data Prepration:
  - [Dataset](https://www.kaggle.com/c/dogs-vs-cats/data?select=test1.zip):- Taken from Kaggle Dog vs Cat competition dataset.
  - Dataset have 25000 Train images and 12500 Test images, but we will ignore those test images set and use only Training data with 25000 images
  - 25000 images are seperated into 22000 Train images and 3000 Test/Validation images
  - Images filenames are in the format <cat*.jpg> or <dog*.jpg>
 Note : Dataset is not added in this repository as the size of the dataset is too large so it is recommended to download the dataset form the link provided above.
 
- Model Workflow:
  - Import packages
  - Load data
  - Compile the Model
  - Train
  - Saving the resulting Model and weights in respective Folders created
  - Plot the history of training and validation/test error and save them under graph folder
