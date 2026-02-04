print("Its running")

# Imports
import os
import glob
import pandas as pd
import numpy as np
import datetime
# Resolves a weird error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = "."
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_ID)
os.makedirs(CKPT_DIR, exist_ok=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timeit
import importDataset as importData


s = 'print("startpoint")'

# Defs
numClass = 2
numEpochs = 1000
lr = 1e-3

    # trainings considered before weights are changed
    # increasing improves accuracy, increases runtime, i think
batchSize = 128

# numsteps per epoch, not used with batch size
desired_steps = 20

# Data shapes expected by the network
trainDataShape = (624, 64, 1)
trainLabelsShape = (624, 2)
valDataShape = (208, 64, 1)
valLabelsShape = (208, 2)
testDataShape = (208, 64, 1)
testLabelsShape = (208, 2)

# Shape of 1 signal into network
inputShape = (64, 1)

def build():
    # Loading, catagorizing data
    allData = importData.importDataset()
    allData, shuffledIndex = importData.shuffleDataset(allData)
    data, labels = importData.splitDataLabels(allData)

    print("Data shape: ", data.shape)
    print("Label shape: ", labels.shape)
    print("****************************************")

    train_data, train_labels = data[:624], labels[:624]
    val_data, val_labels = data[624:832], labels[624:832]
    test_data, test_labels = data[832:1040], labels[832:1040]

    testIndex = shuffledIndex[832:1040]

    print("Train Data Shape: ", train_data.shape)
    print("Train Label Shape: ", train_labels.shape)
    print("Val Data Shape: ", val_data.shape)
    print("Val Label Shape: ", val_labels.shape)
    print("Test Data Shape: ", test_data.shape)
    print("Test Label Shape: ", test_labels.shape)

    # Image normalization
    #train_data, test_data, val_data = train_data / 255.0, test_data / 255.0, val_data / 255.0
    # Ensure images w/o color have correct array size, comment out for color images
    #train_data = train_data.reshape(list(train_data.shape) + [1]) 
    #val_data = val_data.reshape(list(val_data.shape) + [1]) 
    #test_data = test_data.reshape(list(test_data.shape) + [1]) 
    
    train_data = train_data / np.max(train_data)
    val_data = val_data / np.max(val_data)
    test_data = test_data / np.max(test_data)

    # Puts labels into one-hot, a binary vector holding class
    train_labels = to_categorical(train_labels, numClass)
    val_labels = to_categorical(val_labels, numClass)
    test_labels = to_categorical(test_labels, numClass)
    
    train_data = train_data.reshape(-1, 64, 1)
    val_data = val_data.reshape(-1, 64, 1)
    test_data = test_data.reshape(-1, 64, 1)

    print("****************************************")
    print("Normalized Train Data Shape: ", train_data.shape)
    print("Normalized Train Label Shape: ", train_labels.shape)
    print("Normalized Val Data Shape: ", val_data.shape)
    print("Normalized Val Label Shape: ", val_labels.shape)
    print("Normalized Test Data Shape: ", test_data.shape)
    print("Normalized Test Label Shape: ", test_labels.shape)

    # Assures data and label shapes are correct
    assert train_data.shape == trainDataShape
    assert train_labels.shape == trainLabelsShape
    assert val_data.shape == valDataShape
    assert val_labels.shape == valLabelsShape
    assert test_data.shape == testDataShape
    assert test_labels.shape == testLabelsShape

    # # Model creation
    # model = models.Sequential([
    #     layers.Input(inputShape),

    #     layers.Conv1D(filters=32, kernel_size=8),
    #     layers.Activation("relu"),
    #     layers.MaxPooling1D(pool_size=2),
    #     layers.Dropout(0.2),

    #     layers.Conv1D(filters=64, kernel_size=4),
    #     layers.Activation("relu"),
    #     layers.MaxPooling1D(pool_size=2),
    #     layers.Dropout(0.2),

    #     layers.Flatten(),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(numClass, activation='softmax')
    # ])

    model = models.Sequential([
        layers.Input(inputShape),

        layers.Conv1D(filters=32, kernel_size=8, padding="same"),
        layers.Activation("relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')
    ])

    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Training
    #history = model.fit(train_data, train_labels, epochs=numEpochs, callbacks=make_callbacks(), batch_size=batchSize, validation_data=(val_data, val_labels))
    history = model.fit(train_data, train_labels, epochs=numEpochs, callbacks=make_callbacks(), steps_per_epoch=desired_steps, validation_data=(val_data, val_labels))

    # Evaluation
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Get predictions
    print("\n Generating detailed predictions...")
    y_pred_proba = model.predict(test_data, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # print("Leng test_data: ", len(test_data))
    # print("Leng y_pred: ", len(y_pred))
    # 
    # print("Leng test_labels: ", len(test_labels))
   

    # print("Shuffled Index: (Zero indexed)")
    # print(testIndex)
    # print("Test labels: ")
    # print(test_labels.astype(int))
    # print("Predection: ")
    # print(y_pred)

    # Confusion matrix
    print("\n Confusion Matrix:")
    true = np.argmax(test_labels.astype(int), axis=1)
    predicted = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true, predicted)
    print(cm)
    print(f"\n   True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]} | True Positives:  {cm[1,1]}")
    
    fp_idx = np.where((true == 0) & (predicted == 1))  # false positives
    fn_idx = np.where((true == 1) & (predicted == 0))  # false negatives

    print("False Positives: ", testIndex[fp_idx])
    print("False Negatives:", testIndex[fn_idx])

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

def make_callbacks():
    # Keras 3 requires .weights.h5 for weights-only checkpoints
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CKPT_DIR, "best.weights.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1
    )
    return [ckpt_cb]

print("Time to run: ", timeit.timeit(setup=s, stmt=build, number=1), "s")
print("Its not running")