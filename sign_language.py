import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class SignLanguage:
    def __init__(self):
        self.model = None
        
        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()
    
    def create_model(self):
        
        model = Sequential()
        model.add(Conv2D(filters = 50, kernel_size = (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))
        model.add(layers.Dense(10, activation="relu"))
        model.add(Flatten())
        model.add(Dropout(0.7))
        model.add(Dense(25, activation = "softmax"))

        model.compile('adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        
        self.model = model
    
    def prepare_data(self, images, labels):

        dec = 0.8

        N, M = images.shape 
        batch_size = int(dec*N)
        image_size = int(math.sqrt(M))

        images = (images-images.mean())/(images.std())
        images = images.reshape(N, image_size, image_size, 1)

        test_labels = labels[batch_size:]
        labels = labels[:batch_size]

        test_images = images[batch_size:]
        images = images[:batch_size]

        labels = to_categorical(labels, 25)
        test_labels = to_categorical(test_labels, 25)
        

        self.data = {
            "train": (images, labels), # (x_train, y_train)
            "test" : (test_images, test_labels), # (x_test, y_test)
        }
    
    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
    
        history = None
        x_train = self.data["train"][0]
        y_train = self.data["train"][1]
        history = self.model.fit(x_train, y_train, batch_size = batch_size, 
                                 epochs=epochs, verbose=verbose,
                                 validation_data = self.data["test"])
        return history
    
    def predict(self, data):
        
        mean = data.mean()
        std = data.std()

        data = (data - mean)/std
        data = np.reshape(data, (len(data), 28, 28,1))

        output = self.model.predict(data)
        return np.argmax(output, axis=1)
    
    def visualize_data(self, data):

        if data is None: return
        
        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()

    def visualize_accuracy(self, history):

        if history is None: return
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()

if __name__=="__main__":
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    train_labels, test_labels = train['label'].values, test['label'].values
    train.drop('label', axis=1, inplace=True)
    test.drop('label', axis=1, inplace=True)

    num_classes = test_labels.max() + 1
    train_images, test_images = train.values, test.values

    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

if __name__=="__main__":
    my_model = SignLanguage()
    my_model.prepare_data(train_images, train_labels)

if __name__=="__main__":
    my_model.visualize_data(my_model.data["train"])

if __name__=="__main__":
    history = my_model.train(epochs=30, verbose=1)
    my_model.visualize_accuracy(history)

if __name__=="__main__":
    y_pred = my_model.predict(test_images)
    accuracy = accuracy_score(test_labels, y_pred)
    print(accuracy)