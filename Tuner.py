print("IM READY TO TUNE BABY")

# Imports
import os
import glob
import pandas as pd
import numpy as np
import datetime
import csv
import time
# Resolves a weird error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = "."
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_ID)
os.makedirs(CKPT_DIR, exist_ok=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timeit
import importDataset as importData
from Models import Models

numClass = 2

epochsMin = 50
epochsMax = 650
epochsStep = 100

lrMin = .0001
lrMax = .01
lrStep = .001

stepsMin = 5
stepsMax = 45
stepsStep = 10 

# Data shapes expected by the network
trainDataShape = (624, 64, 1)
trainLabelsShape = (624, 2)
valDataShape = (208, 64, 1)
valLabelsShape = (208, 2)
testDataShape = (208, 64, 1)
testLabelsShape = (208, 2)

# Shape of 1 signal into network
inputShape = (64, 1)

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

def build_model(num):
    template = Models.modelArray[num]
    model = keras.models.clone_model(template)
    model.build(template.input_shape)
    return model

modelNum = 1
# epochList = [50, 150, 250, 400, 600]
# lrList = [0.0001, 0.0005, 0.001, .002, .003, .004, .005, .006, .007, .008, .009, .01]
# stepList = [5, 10, 20, 30, 40, 50]

epochList = [600]
lrList = [.009, .01]
stepList = [5, 10, 20, 30, 40, 50]


output_csv = "grid_results.csv"
file_exists = os.path.isfile(output_csv)
with open(output_csv, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["modelNum", "epochs", "lr", "steps_per_epoch", "best_val_acc", "test_acc", "test_loss", "false_pos", "false_neg", "runtime"])

    for epochs in epochList:
        for lr in lrList:
            for steps in stepList:
                keras.backend.clear_session()
                start = time.time()
                print("Epochs: ", epochs, " lr: ", lr, " steps: ", steps)
                model = build_model(modelNum)

                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=['accuracy']
                )
                
                # class_weight: {negative_class: weight, positive_class: weight}
                class_weight = {0: 2.0, 1: 1.0}  # Increase 1’s weight to reduce FNs

                history = model.fit(train_data, train_labels, epochs=epochs, callbacks=make_callbacks(), steps_per_epoch=steps, validation_data=(val_data, val_labels), class_weight=class_weight)
                best_val_acc = float(max(history.history['val_accuracy']))
                test_loss, test_acc = model.evaluate(test_data, test_labels)

                # Get predictions
                y_pred_proba = model.predict(test_data, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int)
                true = np.argmax(test_labels.astype(int), axis=1)
                predicted = np.argmax(y_pred, axis=1)
                cm = confusion_matrix(true, predicted)

                print("False Positives: " + str(cm[0,1]))
                print("False Negatives: " + str(cm[1,0]))

                print(f"Best Validation accuracy: {best_val_acc * 100:.2f}%")
                print(f"Test accuracy: {test_acc * 100:.2f}%")

                end = time.time()
                runtime = (end-start)
                print("Runtime: " + str(runtime))

                writer.writerow([int(modelNum), int(epochs), float(lr), int(steps), float(best_val_acc), float(test_acc), float(test_loss), cm[0,1], cm[1,0], float(runtime)])
                f.flush()
