# train_1d_cnn.py
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from importDataset import *

# Hyperparameters
sequence_length = 64     # your requested input length
n_features = 1           # number of channels per timestep; set >1 if you have multiple features
n_classes = 2            # your requested output size

model = keras.Sequential([
    layers.Input(shape=(sequence_length, n_features)),    # (batch, 64, channels)

    layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 2-class softmax output
    outputs = layers.Dense(n_classes, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs)

    # Loss for integer labels 0..1
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        'accuracy',
        keras.metrics.SparseCategoricalAccuracy(name='auc')
    ]

    opt = keras.optimizers.Adam(learning_rate=lr)

    # NOTE: Do NOT wrap with LossScaleOptimizer in modern Keras; policy handles it.
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

# ---------------------------
# Callbacks
# ---------------------------
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
    early_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    tb_cb = keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    # Reduce LR on plateau (simple, robust)
    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    return [ckpt_cb, early_cb, tb_cb, lr_cb]

# ---------------------------
# Training orchestration
# ---------------------------
def train():
    # Load data
    X, y = load_data()

    # Train/Val split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Datasets
    ds_train = make_dataset(X_train, y_train, batch_size=batch_size, training=True)
    ds_val = make_dataset(X_val, y_val, batch_size=batch_size, training=False)

    # Class weights (optional; helpful for imbalance)
    class_weight = compute_class_weights(y_train, n_classes=n_classes)

    # Build model
    model = build_model(sequence_length=sequence_length,
                        n_features=n_features,
                        n_classes=n_classes,
                        lr=learning_rate)

    model.summary()

#dataset=importDataset()
#labels=importLabels()
#print("Dataset shape:", dataset.shape)
#print("Labels shape:", labels.shape)
#print("First 40 labels:", labels[:40])