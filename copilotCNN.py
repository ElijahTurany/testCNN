
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameters
sequence_length = 64     # your requested input length
n_features = 1           # number of channels per timestep; set >1 if you have multiple features
n_classes = 2            # your requested output size

model = keras.Sequential([
    layers.Input(shape=(sequence_length, n_features)),    # (batch, 64, channels)

    layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    layers.GlobalAveragePooling1D(),

    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),

    # Final layer: 2 outputs
    layers.Dense(n_classes, activation='softmax')  # use 'linear' if you prefer logits
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',   # for integer labels 0..1
    metrics=['accuracy']
)

model.summary()
