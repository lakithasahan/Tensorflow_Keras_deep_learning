# Tensorflow_Keras_deep_learning
In this example we use Keras deep learning library to classify  famous IRIS data set. You can download the complete code with data set from download section.

**Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.**

## Use Keras if you need a deep learning library that:

    1) Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
    2) Supports both convolutional networks and recurrent networks, as well as combinations of the two.
    3) Runs seamlessly on CPU and GPU.

## Keras Cheat sheet which will be extreamly useful when creating classification models

![keras](https://user-images.githubusercontent.com/24733068/65365540-8b617500-dc5d-11e9-9765-e8548b50646f.jpeg)

```python

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


```

## OUTPUT plot can be found below
**Precition and Recall=1.0**
![keraspredict](https://user-images.githubusercontent.com/24733068/65365630-5570c080-dc5e-11e9-908b-0dc198429924.png)




