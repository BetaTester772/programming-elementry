#make marchine learning model with COVID_Diffusion.csv with tensorflow as single layer perceptron

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
# import train test splist
from sklearn.model_selection import train_test_split
import pandas as pd

#read csv file
df = pd.read_csv('COVID_Diffusion.csv')
df.head()

#make x, y for train set Date and confirmed case
x = df['Date'].values
y = df['confirmed case'].values

#make train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#make model singlelayer perceptron for accuracy and loss
model = Sequential([
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#compile model
model.compile(optimizer='adam', loss=['accuracy'])

#train model get loos and accuracy
history = model.fit(x_train, y_train, epochs=100)

#make graph for loss and accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

#make graph
plt.scatter(x_train, y_train, label='train')
plt.scatter(x_test, y_test, label='test')
plt.plot(x_train, model.predict(x_train), color='red', label='predict')
plt.legend()
plt.show()

