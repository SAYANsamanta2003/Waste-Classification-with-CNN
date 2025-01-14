# Waste-Classification-with-CNN
This project implements a Convolutional Neural Network (CNN) model to classify waste into different categories such as recyclable, organic, and non-recyclable.

![image](https://github.com/user-attachments/assets/fae5e236-4256-46c9-9983-ac28a091d727)

This project implements a CNN-based image classification model to classify waste into two categories: Organic and Recyclable.

# Dataset
Dataset consists of two classes: Organic and Recyclable.
Training and testing directories are pre-organized, with images in respective folders.
**Key Components**<br>
** Data Preprocessing:**
Images are resized to 224x224 for uniformity.
Data augmentation using ImageDataGenerator to improve model generalization.
Normalization is applied by rescaling pixel values to [0, 1].
![image](https://github.com/user-attachments/assets/f33548ad-74e8-43ae-b91a-11927fd6869b)

# CNN Architecture:
Built using Keras Sequential API.
Three convolutional layers with ReLU activation and MaxPooling.
Fully connected layers with dropout for regularization.
Output layer uses softmax activation for multi-class classification.
# Model Training: <br>
**Loss function:** 
categorical_crossentropy for multi-class classification.
_Metrics:_ Accuracy to evaluate model performance.
The model is trained for 20 epochs on the dataset.

# Prediction Functionality:
A custom function to predict whether a given image is Organic or Recyclable.
Outputs the result along with the input image for visual confirmation.
Results
Achieved a training accuracy of ~98% and validation accuracy of ~89%.
The model shows strong potential for real-world waste classification tasks.
# Requirements
Python 3.x
Libraries: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib<br>
**TO USED**<br>
_Run the script to train the model and make predictions._
# Result
![image](https://github.com/user-attachments/assets/668c7b9e-16dc-4a6f-bda0-2a9b5d3e60f6)
![image](https://github.com/user-attachments/assets/56e18d66-92f9-4455-8b34-fe95d3cdb7e9)

# dataset link
https://www.kaggle.com/code/beyzanks/waste-classification-with-cnn

# Any problem you face mail me
my mail id is samanta2003sayan@gmail.com
