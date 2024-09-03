import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file_path = 'D:\\Aarohi\\logs\\history.csv'
df = pd.read_csv(csv_file_path, index_col=0)

# Select only validation loss columns
val_loss_columns = [col for col in df.columns if 'val_' in col]

# Plotting with different line styles
fig, ax = plt.subplots(figsize=(10, 6))

# Define line styles
line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 2))]

# Plot each validation line with a different style
for i, column in enumerate(val_loss_columns):
    style = line_styles[i % len(line_styles)]
    ax.plot(df.index, df[column], label=f'Validation {column}', linestyle=style)

# Customize the plot
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss Over Epochs')
ax.legend()
plt.grid(True)
plt.show()