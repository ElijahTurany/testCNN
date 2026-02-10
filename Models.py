# Imports
import os
import glob
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timeit
import importDataset as importData

numClass = 2
inputShape = (64, 1)

class Models:
    # Best so far
    modelA = models.Sequential([
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

    # modelA w/ first layer duplicated
    modelB = models.Sequential([
        layers.Input(inputShape),

        layers.Conv1D(filters=32, kernel_size=8, padding="same"),
        layers.Activation("relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(filters=32, kernel_size=8, padding="same"),
        layers.Activation("relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')
    ])
    modelArray = [modelA, modelB]