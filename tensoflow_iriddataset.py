from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from pandas import get_dummies
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset = pd.read_csv('iris.csv')

# transform species to numerics
dataset.loc[dataset.variety == 'Setosa', 'variety'] = 0
dataset.loc[dataset.variety == 'Versicolor', 'variety'] = 1
dataset.loc[dataset.variety == 'Virginica', 'variety'] = 2

train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    get_dummies(dataset.variety.values), test_size=0.3)

train_X = np.array(train_X).astype(np.float32)
test_X = np.array(test_X).astype(np.float32)
train_y = np.array(train_y).astype(np.float32)
test_y = np.array(test_y).astype(np.float32)

print(train_X.shape)
print(train_y)

model = Sequential()

# first input layer with first hidden layer in a single statement
model.add(Dense(100, input_shape=(4,), activation='relu'))
# 10 is the size(no. of neurons) of first hidden layer, 4 is the no. of features in the input layer
# input_shape=(4,)  can also be written as   input_dim=4

model.add(Dense(50, activation='relu'))
# ouput layer
model.add(Dense(3, activation='softmax'))  # 3 = no. of neurons in output layer as three categories of labels are there

# compile method receives three arguments: "an optimizer", "a loss function" and "a list of metrics"
model.compile(Adam(lr=0.04), 'categorical_crossentropy', ['accuracy'])
# we use "binary_crossentropy" for binary classification problems and
# "categorical_crossentropy" for multiclass classification problems
# the compile statement can also be written as:-
# model.compile(optimizer=Adam(lr=0.04), loss='categorical_crossentropy',metrics=['accuracy'])
# we can give more than one metrics like ['accuracy', 'mae', 'mape']

model.summary()

model.fit(train_X, train_y, epochs=100)
y_pred = model.predict(test_X)
model.evaluate(test_X, test_y)
predict_y = y_pred.argmax(axis=-1)
print(predict_y)
test_y_prob = test_y.argmax(axis=-1)
print(test_y_prob)

print('Accuracy', accuracy_score(test_y_prob, predict_y))
print('micro precision', precision_score(test_y_prob, predict_y, average='micro'))
print('macro recall', recall_score(test_y_prob, predict_y, average='macro'))
print('micro recall', recall_score(test_y_prob, predict_y, average='micro'))
print(test_X)
print(test_X[:, 0])

colormap = np.array(['#f50000', '#f58800', '#0b599f'])
pop_a = mpatches.Patch(color='#f50000', label='Setosa')
pop_b = mpatches.Patch(color='#f58800', label='Versicolor')
pop_c = mpatches.Patch(color='#0b599f', label='Virginica')

plt.subplot(2, 1, 1)
plt.scatter(test_X[:, 0], test_X[:, 1], c=colormap[predict_y], cmap='Paired')
plt.title("Keras NN Model predicted target output")

plt.subplot(2, 1, 2)
plt.scatter(test_X[:, 0], test_X[:, 1], c=colormap[test_y_prob], cmap='Paired')


plt.title("Actual target output")

plt.tight_layout()
plt.legend(handles=[pop_a, pop_b, pop_c])

plt.show()
