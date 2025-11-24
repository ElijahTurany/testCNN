print("Its running")

# Imports
import os
import glob
import pandas as pd
# Resolves a weird error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import timeit
import importDataset as importData


s = 'print("startpoint")'

# Defs
numClass = 10
numEpochs = 5
    # trainings considered before weights are changed
    # increasing improves accuracy, increases runtime
batchSize = 64
# Data shapes expected by the network
trainDataShape = (60000, 28, 28, 1)
trainLabelsShape = (60000, 10)
valDataShape = (5000, 28, 28, 1)
valLabelsShape = (5000, 10)
testDataShape = (5000, 28, 28, 1)
testLabelsShape = (5000, 10)

# Shape of 1 signal into network
inputShape = (28, 28, 1)

# Our shapes, WIP
_trainDataShape = (60000, 28, 28, 1)
_trainLabelsShape = (60000, 10)
_valDataShape = (10000, 28, 28, 1)
_valLabelsShape = (10000, 10)
_testDataShape = (10000, 28, 28, 1)
_testLabelsShape = (10000, 10)

def build():
    # Loading, catagorizing data
    data = importData.importDataset()
    labels = importData.importLabels()
    print("Data shape: ", data.shape)
    print("Label shape: ", labels.shape)

    train_data, train_labels = data[:624], labels[:624]
    val_data, val_labels = data[624:832], labels[624:832]
    test_data, test_labels = data[:832], labels[:832]

    print("Normalized Train Data Shape: ", train_data.shape)
    print("Normalized Train Label Shape: ", train_labels.shape)
    print("Normalized Test Val Shape: ", val_data.shape)
    print("Normalized Test Val Shape: ", val_labels.shape)
    print("Normalized Test Data Shape: ", test_data.shape)
    print("Normalized Test Label Shape: ", test_labels.shape)


    #(train_data, train_labels), (testVal_data, testVal_labels) = mnist.load_data()
    # print("Train Data Shape: ", train_data.shape)
    # print("Train Label Shape: ", train_labels.shape)
    # print("Test Data Shape: ", testVal_data.shape)
    # print("Test Label Shape: ", testVal_labels.shape)

    # Split validation and test data
    #val_data, val_labels = testVal_data[:5000], testVal_labels[:5000]
    #test_data, test_labels = testVal_data[5000:], testVal_labels[5000:]

    # Image normalization
    #train_data, test_data, val_data = train_data / 255.0, test_data / 255.0, val_data / 255.0
    # Ensure images w/o color have correct array size, comment out for color images
    train_data = train_data.reshape(list(train_data.shape) + [1]) 
    val_data = val_data.reshape(list(val_data.shape) + [1]) 
    test_data = test_data.reshape(list(test_data.shape) + [1]) 


    # Puts labels into one-hot, a binary vector holding class
    train_labels = to_categorical(train_labels, numClass)
    val_labels = to_categorical(val_labels, numClass)
    test_labels = to_categorical(test_labels, numClass)

    print("****************************************")
    print("Normalized Train Data Shape: ", train_data.shape)
    print("Normalized Train Label Shape: ", train_labels.shape)
    print("Normalized Test Val Shape: ", val_data.shape)
    print("Normalized Test Val Shape: ", val_labels.shape)
    print("Normalized Test Data Shape: ", test_data.shape)
    print("Normalized Test Label Shape: ", test_labels.shape)

    # Assures data and label shapes are correct
    assert train_data.shape == trainDataShape
    assert train_labels.shape == trainLabelsShape
    assert val_data.shape == valDataShape
    assert val_labels.shape == valLabelsShape
    assert test_data.shape == testDataShape
    assert test_labels.shape == testLabelsShape

    # Model creation
    model = models.Sequential([
        layers.Input(inputShape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(numClass, activation='softmax')
    ])

    # Compilation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training
    history = model.fit(train_data, train_labels, epochs=numEpochs, batch_size=batchSize, validation_data=(val_data, val_labels))

    # Evaluation
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Plotting
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:red')
    trainAcc = ax1.plot(history.history['accuracy'], label='Training Accuracy', color='tab:red')
    valAcc = ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:blue')
    trainLoss = ax2.plot(history.history['loss'], label='Training Loss', color='tab:blue')
    valLoss = ax2.plot(history.history['val_loss'], label='Validation Loss', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines = trainAcc + valAcc + trainLoss + valLoss
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='lower left')

    plt.title('Accuracy/Loss')
    plt.show()

print("Time to run: ", timeit.timeit(setup=s, stmt=build, number=1), "s")