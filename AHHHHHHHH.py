print("Its running")

# Imports
import os
# Resolves a weird error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Defs
numClass = 10
# Data shapes expected by the network
trainDataShape = (50000, 32, 32, 3)
trainLabelsShape = (50000, 10)
testDataShape = (10000, 32, 32, 3)
testLabelsShape = (10000, 10)

# Loading, catagorizing data
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
print("Train Data Shape: ", train_data.shape)
print("Train Label Shape: ", train_labels.shape)
print("Test Data Shape: ", test_data.shape)
print("Test Label Shape: ", test_labels.shape)

# Normalization, puts labels into one-hot, a binary vector holding class
train_data, test_data = train_data / 255.0, test_data / 255.0
train_labels = to_categorical(train_labels, numClass)
test_labels = to_categorical(test_labels, numClass)

print("Normalized Train Data Shape: ", train_data.shape)
print("Normalized Train Label Shape: ", train_labels.shape)
print("Normalized Test Data Shape: ", test_data.shape)
print("Normalized Test Label Shape: ", test_labels.shape)

# Assures data and label shapes are correct
assert train_data.shape == trainDataShape
assert train_labels.shape == trainLabelsShape
assert test_data.shape == testDataShape
assert test_labels.shape == testLabelsShape

# Model defs
inputShape = (32, 32, 3)

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
history = model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_data=(test_data, test_labels))

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