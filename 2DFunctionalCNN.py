print("Its running")

# Imports
import os
# Resolves a weird error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timeit

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

def build():
    # Loading, catagorizing data
    (train_data, train_labels), (testVal_data, testVal_labels) = mnist.load_data()
    #signalData = 
    print("Train Data Shape: ", train_data.shape)
    print("Train Label Shape: ", train_labels.shape)
    print("Test Data Shape: ", testVal_data.shape)
    print("Test Label Shape: ", testVal_labels.shape)

    # Split validation and test data
    val_data, val_labels = testVal_data[:5000], testVal_labels[:5000]
    test_data, test_labels = testVal_data[5000:], testVal_labels[5000:]

    # Image normalization
    train_data, test_data, val_data = train_data / 255.0, test_data / 255.0, val_data / 255.0
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

    # Get predictions
    print("\n Generating detailed predictions...")
    y_pred_proba = model.predict(test_data, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("Leng test_data: ", len(test_data))
    print("Leng y_pred: ", len(y_pred))
    print("Leng test_labels: ", len(test_labels))

    # Confusion matrix
    print("\n Confusion Matrix:")
    cm = confusion_matrix(test_labels, y_pred)
    print(cm)
    print(f"\n   True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]} | True Positives:  {cm[1,1]}")

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