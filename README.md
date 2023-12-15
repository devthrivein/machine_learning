[//]: # (Image References)

[image0]: ./images/sampleDataset.png "SampleDataset"
[image1]: ./images/Conv.png "Conv Accuracy"
[image2]: ./images/Conv2.png "Conv Loss"
[image3]: ./images/Mobilenet.png "Mobilenet Accuracy"
[image4]: ./images/Mobilenet2.png "Mobilenet Loss"
[image5]: ./images/Inception.png "Inception Accuracy"
[image6]: ./images/Inception2.png "Inception Loss"
[image7]: ./images/Resnet.png "Resnet Accuracy"
[image8]: ./images/Resnet2.png "Resnet Loss"
[image9]: ./images/result.png "Result1"
[image10]: ./images/result2.png "Result2"
[image11]: ./images/result3.png "Result3"


## Table of Contents

1. [Project Overview](#projectOverview)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Training Process](#trainingProcess)
5. [Results and Screenshots](#results)

## Project Overview <a name="projectOverview"></a>

In this project, we built image classification models that are deployed in a real-world application. Given an image of an SME store, the algorithm will identify whether the image contains a banner or not. We created several models using CNN and transfer learning with MobileNet, ResNet, and Inception.

These models serve as the MVP feature of the application. Our initial plan was to create a model capable of analyzing and generating design ideas for store logos and banners.



## Requirements <a name="requirements"></a>

To run the code, we will need to install:
- tensorflow==2.15.0
- tensorflow-hub==0.15.0
- numpy==1.23.5
- matplotlib==3.7.1


You can install the libraries with this code:

```
pip install -r requirements.txt
```

## Usage <a name="usage"></a>

1. Clone [this repository](https://github.com/devthrivein/machine_learning.git) to your local machine. 
2. Open the `classify.ipynb` in Jupyter Notebook, or Colab.
3. Modify the values of `input_shape`, `model_path` and `image_path`.
4. Run the notebook
5. The notebook will preprocess the data, make prediction, and visualize the result.



## Training Process <a name="trainingProcess"></a>
- **Dataset:** The model was trained on a dataset of over 1700 images from different SME stores. ![sampleDataset][image0]
- **Preprocessing:** The images were resized to the corresponding pixels expected by the model and normalized to have pixel values between 0 and 1. This process uses the ImageDataGenerator from the TensorFlow library. 
- **Training Parameters:** The model was trained using the Adam optimizer with an adaptive learning rate. The model was trained for over 200 epochs, but only the most optimum metrics state of the model during training was saved. These processes were conducted using the built-in callback API.

### Convolutional Neural Network

Takes an input shape of (450, 450, 3) and uses 4 convolutional, a max pooling layer, followed by flatten and dense layers.

The model achieved over 98% accuracy on training set, but dropped to 72% on the test set

![Conv Accuracy][image1]

![Conv Loss][image2]

### Transfer Learning with Mobilnet v3
Takes an input shape of (224, 224, 3) followed by and dense layers.

The model achieved over 95% accuracy on training set and 85% on the test set. This model is significantly better than the CNN model.

![Mobilenet Accuracy][image3]

![Mobilenet Loss][image4]

### Transfer Learning with Inception v3
Takes an input shape of (299, 299, 3) followed by and dense layers.

The model achieved over 96% accuracy on training set and 86% on the test set, slightly better than the previous model above.

![Inception Accuracy][image5]

![Inception Loss][image6]

### Transfer Learning with Resnet v2
Takes an input shape of (224, 224, 3) followed by and dense layers.

The model achieved over 96% accuracy on training set, but dropped sharply to 82% on the test set.

![Resnet Accuracy][image7]

![Resnet Loss][image8]

## Results and Screenshots <a name="results"></a>

![Result][image9]

![Result2][image10]

![Result3][image11]
