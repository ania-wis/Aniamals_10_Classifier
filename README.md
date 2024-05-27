# Aniamals_10_Classifier

This repository contains a deep learning model for classifying images of 10 different animals using TensorFlow and Keras. The model is trained on a dataset of images and can predict the class of new images.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Colab: https://colab.research.google.com/drive/1lhClincgI4J8sJG-jB0KahhCh-OUU0Z8?usp=sharing

Download model: https://drive.google.com/file/d/1RiXlO5mWv11HVnA8shMlHZ1m_XP_UYdh/view?usp=sharing

This project aims to build an image classifier that can distinguish between 10 different animal classes:
- Dog
- Cat
- Cow
- Butterfly
- Sheep
- Spider
- Squirrel
- Horse
- Elephant
- Chicken

The model is trained using a convolutional neural network (CNN) and achieves a reasonable accuracy on the validation dataset.

## Dataset

The dataset consists of images of animals categorized into 10 classes. The images are resized to 128x128 pixels for training and prediction.
Download dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10
## Model Architecture

The model uses a CNN architecture with the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layer to prevent overfitting
- Flatten layer
- Dense layers with ReLU activation
- Output layer with softmax activation

Here is the detailed model architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```
## Training

The model is trained using the following parameters:
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Batch size: 32
- Epochs: 20

## Evaluation

The model is evaluated on a separate validation dataset. The training and validation losses and accuracies are tracked to monitor the model's performance.

## Prediction

The model can predict the class of new images. To classify images from a test dataset, the images are resized to 128x128 pixels for the model to process.

## Requirements
- Python 3.x
- ensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- Pandas
- PIL (Python Imaging Library)

## Install the required packages using:
```bash
pip install tensorflow keras matplotlib numpy pandas pillow

```
## Usage
Training the Model
```python
import tensorflow as tf

# Load and prepare the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.3,
    subset='training',
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.3,
    subset='validation',
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Define the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# Save the model
model.save('animal_classifier_model.keras')
```
## Predicting New Images
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('animal_classifier_model.h5')

# Class names
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Function to display image with prediction
def display_image_with_prediction(img, prediction):
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class]
    
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class_name}")
    plt.axis('off')
    plt.show()
    
    print(f"Prediction: {predicted_class_name}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {prediction[i]*100:.2f}%")

# Load and predict a single image
img_path = 'path/to/test/image.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
display_image_with_prediction(image.load_img(img_path), predictions[0])
```
## Results
After training, the model achieves an accuracy of around 86% on the training data and 69% on the validation data. Further tuning and more data may improve these results.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
