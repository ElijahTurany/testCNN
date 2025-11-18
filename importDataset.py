import os
import numpy as np
import pandas as pd

def importDataset():
    # Directory containing the CSV files
    windowed_dir = 'windowed'
    # Number of files
    num_files = 1040
    # Number of rows to extract per file (rows 2-64, i.e., index 1 to 63)
    rows_per_file = 64
    # Number of columns per row
    num_columns = 2

    # Preallocate array
    data_array = np.zeros((num_files, num_columns, rows_per_file))

    for i in range(1, num_files + 1):
        file_path = os.path.join(windowed_dir, f'windowed_{i}.csv')
        df = pd.read_csv(file_path, header=None)
        # Extract rows 2-64 (index 1 to 64 exclusive)
        selected = df.iloc[1:65, :].to_numpy().T  # shape: (2, 64)
        data_array[i - 1] = selected

    # data_array is now shape (1040, 2, 64)
