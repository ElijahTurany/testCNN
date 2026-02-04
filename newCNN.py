# train_1d_cnn.py
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import importDataset as iD
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

sequence_length = 64       # your 1D length
n_features = 1             # channels per timestep
n_classes = 2              # 2 outputs (softmax)
batch_size = 64
epochs = 50
learning_rate = 1e-3
val_split = 0.2
use_mixed_precision = False  # set True if using a GPU with Tensor Cores

# Directory setup
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = "."
LOG_DIR = os.path.join(BASE_DIR, "logs", RUN_ID)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_ID)
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models", RUN_ID)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Mixed precision (automatic loss scaling in modern TF/Keras)
if use_mixed_precision:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# ---------------------------
# Data Loading / Simulation
# Replace this with your real loader
# ---------------------------
def load_data():
    X = iD.importDataset()[ :, 0].T.reshape(1040, 64, 1)
    Y = iD.importLabels()

    #print(X.shape)
    #print(Y.shape)
    """
    Replace with your own dataset loader.
    Return:
        X: float32 array (num_samples, sequence_length, n_features) [1040, 64, 1]
        y: int32 array (num_samples,) labels in {0, 1} [1040,]
    """
    
    num_samples = 1040
    X = np.random.randn(num_samples, sequence_length, n_features).astype('float32')

    # Synthetic target: class 1 if mean > 0, else class 0
    y = (X.mean(axis=(1, 2)) > 0).astype('int32')
    return X, y

# ---------------------------
# tf.data pipeline
# ---------------------------
def make_dataset(X, y, batch_size, training=True, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training and shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10_000), seed=SEED, reshuffle_each_iteration=True)

    # Simple normalization/augmentation example for 1D signals

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------
# Class weights (optional)
# ---------------------------
def compute_class_weights(y, n_classes=2):
    """Return a dict usable by model.fit(class_weight=...)."""
    counts = np.bincount(y, minlength=n_classes)
    total = counts.sum()
    weights = {i: total / (n_classes * max(counts[i], 1)) for i in range(n_classes)}
    return weights

# ---------------------------
# Model definition (Functional API)
# ---------------------------
def build_model(sequence_length=64, n_features=1, n_classes=2, lr=1e-3):
    inputs = keras.Input(shape=(sequence_length, n_features))

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

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
        keras.metrics.SparseCategoricalAccuracy(name='accuracy')
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
    return [ckpt_cb, tb_cb, lr_cb]

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

    # Fit
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=make_callbacks(),
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    eval_results = model.evaluate(ds_val, return_dict=True, verbose=1)
    print("\nValidation metrics:", eval_results)

    # ---- Exports ----
    # Preferred full-model Keras format
    keras_path = os.path.join(EXPORT_DIR, "model.keras")
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    # Not sure what this code does so i commented it out

    # # Try exporting a SavedModel for serving (Keras 3 provides model.export)
    # export_path = os.path.join(EXPORT_DIR, "savedmodel")
    # try:
    #     # Works in Keras 3+
    #     model.export(export_path)
    #     print(f"Exported inference model (SavedModel) to {export_path}")
    # except AttributeError:
    #     # Fallback for TF/Keras 2.x
    #     model.save(export_path)
    #     print(f"Exported SavedModel to {export_path}")

    # # Optional legacy H5 (many tools still accept this)
    # h5_path = os.path.join(EXPORT_DIR, "model.h5")
    # model.save(h5_path)
    # print(f"Also saved legacy H5 model to {h5_path}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    y_val_true = y_val
    y_val_pred_prob = model.predict(ds_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))

    print("\nClassification Report:")
    print(classification_report(y_val_true, y_val_pred, digits=4))

    
    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.show()
    return model, (X_val, y_val), ds_val

# ---------------------------
# Inference helper
# ---------------------------
def predict_probabilities(model, X_batch):
    """
    X_batch: np.ndarray of shape (B, 64, n_features)
    Returns: np.ndarray of shape (B, 2) with class probabilities
    """
    # Apply same standardization as training if needed
    # Here we mirror the per-sample standardization:
    Xb = X_batch.astype('float32')
    mean = Xb.mean(axis=1, keepdims=True)
    std = Xb.std(axis=1, keepdims=True) + 1e-6
    Xb = (Xb - mean) / std
    return model.predict(Xb, verbose=0)

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    model, (X_val, y_val), ds_val = train()

    # Example inference on a small batch
    sample = X_val[:8]
    probs = predict_probabilities(model, sample)
    preds = probs.argmax(axis=1)
    print("\nSample predictions:", preds)
    print("Sample probabilities:\n", probs)