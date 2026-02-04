import os
import numpy as np
import pandas as pd

def importDataset():
    # Directory containing the CSV files
    windowed_dir = '02_02_windowed'
    # Number of files
    num_files = 1040
    # Number of rows to extract per file (rows 2-64, i.e., index 1 to 64)
    rows_signal = 64
    # Additional label appended as the 65th datapoint
    rows_total = rows_signal + 1
    # Number of columns per row (we keep the signal column only)
    num_columns = 1

    # Preallocate array: shape (num_files, rows_total, num_columns)
    data_array = np.zeros((num_files, rows_total, num_columns))

    # Load labels
    
    print("CWD:", os.getcwd())
    labels_file = 'signalLabels.csv'
    df_labels = pd.read_csv(labels_file, header=None)
    labels_array = df_labels[1].to_numpy().flatten()

    for i in range(1, num_files + 1):
        file_path = os.path.join(windowed_dir, f'windowed_{i}.csv')
        df = pd.read_csv(file_path, header=None)
        # Extract rows 2-64 (index 1 to 64 inclusive -> iloc[1:65]) from column 1 (second column)
        selected = df.iloc[1:65, 1].to_numpy()  # shape: (64,)
        # Place signal values into first 64 positions
        data_array[i - 1, :rows_signal, 0] = selected
        # Append label as the 65th datapoint
        try:
            data_array[i - 1, rows_signal, 0] = labels_array[i - 1]
        except IndexError:
            # If labels file doesn't align, set to 0 and continue
            data_array[i - 1, rows_signal, 0] = 0

    return data_array
    # data_array is now shape (1040, 65, 1) with label at index 64

def importFullSignal():
    signal_dir = '12_03_signal'
    # Number of files
    num_files = 1040
    # Number of rows to extract per file
    rows_signal = 1500
    # Additional label appended as the 65th datapoint
    rows_total = rows_signal + 1
    # Number of columns per row (we keep the signal column only)
    num_columns = 1

    # Preallocate array: shape (num_files, rows_total, num_columns)
    data_array = np.zeros((num_files, rows_total, num_columns))

    # Load labels
    labels_file = 'signalLabels.csv'
    df_labels = pd.read_csv(labels_file, header=None)
    labels_array = df_labels[1].to_numpy().flatten()

    for i in range(1, num_files + 1):
        file_path = os.path.join(signal_dir, f'signal_{i}.csv')
        df = pd.read_csv(file_path, header=None)
        # Extract rows 2-64 (index 1 to 64 inclusive -> iloc[1:65]) from column 1 (second column)
        selected = df.iloc[1:1501, 4].to_numpy()  # shape: (64,)
        # Place signal values into first 64 positions
        data_array[i - 1, :rows_signal, 0] = selected
        # Append label as the 1501th datapoint
        try:
            data_array[i - 1, rows_signal, 0] = labels_array[i - 1]
        except IndexError:
            # If labels file doesn't align, set to 0 and continue
            data_array[i - 1, rows_signal, 0] = 0

    return data_array
    # data_array is now shape (1040, 1501, 1) with label at index 1501

def shuffleDataset(data):
    idx_col = np.arange(len(data)).reshape(-1, 1)
    print(idx_col)

    order = np.arange(len(idx_col))
    print(order)
    np.random.shuffle(order)
    print(idx_col[order])
  
    return data[order], idx_col[order]

def test_importDataset():
    dataset=importDataset()
    dataset = shuffleDataset(dataset)
    print(dataset.shape)  # Should print (1040, 65, 1)
    for i in range(30):
        print(f"Sample {i+1} data (first 5 values): {dataset[i, :5, 0]}, label: {dataset[i, 64, 0]}")

#test_importDataset()

def splitDataLabels(data):
    X = data[:, :64, :]  # shape: (1040, 64, 1)
    y = data[:, 64, 0].astype(int)  # shape: (1040,)
    return X, y

def splitFullDataLabels(data):
    X = data[:, :1500, :]  # shape: (1040, 1501, 1)
    y = data[:, 1500, 0].astype(int)  # shape: (1040,)
    return X, y

