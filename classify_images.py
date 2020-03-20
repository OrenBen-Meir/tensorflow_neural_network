import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random as rand

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255
test_images = test_images/255

# plt.imshow(train_images[1], cmap=plt.cm.binary)
# plt.show()
'''
                neurons
input layer:    780  
hidden layer:   128  
output layer:   9  
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=7)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Tested Accuracy: ', test_acc)
print('Tested Loss: ', test_loss)

prediction = model.predict(test_images)

# predicted_class = [class_names[i] for i in np.argmax(prediction, axis=1)]
for i in range(5, 10):
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual " + class_names[test_labels[i]])
    plt.title('prediction ' + class_names[np.argmax(prediction[i])])
    plt.show()