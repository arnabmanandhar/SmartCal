import os
import h5py
import pandas as pd

def extract_losses_from_logs(logs_dir):
    data = {'epoch': [], 'loss': [], 'val_loss': []}

    def process_dataset(dataset, name):
        try:
            data[name].extend(dataset[:].tolist())
        except KeyError:
            print(f"Skipping {name} due to KeyError")

    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".h5"):
                log_file = os.path.join(root, file)
                print(f"Processing file: {log_file}")

                with h5py.File(log_file, 'r') as f:
                    # Access datasets directly without visiting items
                    for name, dataset in f.items():
                        if isinstance(dataset, h5py.Dataset):
                            process_dataset(dataset, name)

    return pd.DataFrame(data)

def save_to_excel(data, excel_filename='losses.xlsx'):
    data.to_excel(excel_filename, index=False)

# Replace 'logs\\object20240125T2147' with the path to your specific logs directory
logs_directory = 'D:\\Aarohi\\logs\\object20240125T2147'

losses_data = extract_losses_from_logs(logs_directory)
print(losses_data)
save_to_excel(losses_data)
