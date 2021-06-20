"""
Tutorial from:
- https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
- https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
- https://keras.io/examples/vision/mnist_convnet/ 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from tensorflow.keras import Sequential
import os

def load_mnist(mnist_path):
    labels = []
    digits = []
    with open(mnist_path) as f:
        f.readline()
        for line in f.readlines():
            data = line.split(",")
            label = int(data[0])
            img = np.array(list(map(int, data[1:]))).reshape((28, 28))
            labels.append(label)
            digits.append(img)
    labels = np.array(labels)
    digits = np.array(digits)
    return digits, labels


# define cnn model
def build_CNN_model():
    # this has 542,230 training parameters ~6MB
    model = Sequential()
    model.add(Conv2D(28, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model    

    

def build_CNN_2_model(settings):
    # this has 34,826 training parameters -> ~411kB
    k1 = settings['conv2d_1']['kernel_size']
    n1 = settings['conv2d_1']['n_nodes']
    k2 = settings['conv2d_2']['kernel_size']
    n2 = settings['conv2d_2']['n_nodes']
    model = Sequential(
        [
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(n1, kernel_size=(k1, k1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(n2, kernel_size=(k2, k2), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    return model
    

def build_NN_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    return model


def load_CNN_model(checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    settings = {
        'conv2d_1': {
            'kernel_size': 5,
            'n_nodes': 16
        },
        'conv2d_2': {
            'kernel_size': 5,
            'n_nodes': 32
        },
    }
    model = build_CNN_2_model(settings)
    model.load_weights(latest)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    input_dir = "../datasets"
    dataset = "74k" # MNIST font 74k combined
    train_path = os.path.join(input_dir, dataset + "_train.csv")
    test_path = os.path.join(input_dir, dataset + "_test.csv")
    checkpoint_path = "models/CNN_4_74k/cp-{epoch:04d}.ckpt"
    
    print("loading data ...")
    train_digits, train_labels = load_mnist(train_path)
    test_digits, test_labels = load_mnist(test_path)
    print("data loaded.")

    ## Convert to required formats for trainings
    x_train = train_digits.astype('float32')/255
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=10)

    y_test = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    x_test = test_digits.astype('float32')/255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print('Number of images in x_train:', x_train.shape[0])
    print('Number of images in x_test: ', x_test.shape[0])

    # build model
    settings_all = {
        "CNN_2": {
            'conv2d_1': {
                'kernel_size': 3,
                'n_nodes': 32
            },
            'conv2d_2': {
                'kernel_size': 3,
                'n_nodes': 64
            },
        },
        "CNN_3": {
            'conv2d_1': {
                'kernel_size': 3,
                'n_nodes': 16
            },
            'conv2d_2': {
                'kernel_size': 3,
                'n_nodes': 32
            },
        },
         "CNN_4": {
            'conv2d_1': {
                'kernel_size': 5,
                'n_nodes': 16
            },
            'conv2d_2': {
                'kernel_size': 5,
                'n_nodes': 32
            },
        },
    }
    model = build_CNN_2_model(settings_all["CNN_4"])
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ## train
    batch_size = 64

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        )
    history = model.fit(
        x=x_train,
        y=y_train, 
        epochs=10,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[cp_callback]
        )

    # Evaluate the model
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("test evaluation")
    print("accuracy: {:5.2f}%".format(100 * acc))
    print("loss:     {:5.2f}".format(loss))

    # plotting the metrics
    fig, axes = plt.subplots(2, 1)
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='train')
    ax.plot(history.history['val_accuracy'], label='val')
    ax.set_title('model training')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper left')
    ax.set_ylim(top=1)
    ax.grid('on')

    ax = axes[1]
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.set_ylabel('loss')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('epoch')
    ax.legend(loc='upper left')
    ax.grid('on')

    plt.show()
