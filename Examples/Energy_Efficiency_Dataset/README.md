# scratchANN Case Study: Energy Efficiency Dataset

## Getting Started

- Use the same codes from the <a href="https://github.com/pedrampasandide/scratchANN" target="_blank">scratchANN repository</a>, except for `demo.c`, which must be replaced with `demo_EnergyEfficiency.c`.
- Update `Makefile` to reference `demo_EnergyEfficiency.c` instead of `demo.c`.
- The Energy Efficiency dataset has been downloaded and saved as `energy_efficiency_data.txt`, using the Python code provided in the Appendix. The file is formatted to be compatible with the `ReadDataFromFile()` function.
- The rest is exactly the same. `make` and `./sample` to run.
- For more details, check out the [YouTube **scratchANN** playlist](URL), where the case study trains the model using **scratchANN** and compares the results with an MLP implemented in TensorFlow Keras.


## Appendix

The following Python code was used to download, process, and save the Energy Efficiency dataset into `energy_efficiency_data.txt`:

```python
import pandas as pd
import numpy as np
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
df = pd.read_excel(url)

# Convert the DataFrame to a NumPy array
data = df.to_numpy()

# Prepare column labels
column_labels = df.columns.tolist()

# Verify data shape
print("Shape of the data array:", data.shape)

# Filepath to save on Google Drive
file_path = "/content/drive/My Drive/energy_efficiency_data.txt"

# Save data in txt format with header
with open(file_path, 'w') as file:
# Write header information
file.write("# Energy Efficiency Dataset\n")
file.write("# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx\n")
file.write("# Columns:\n")
file.write("# " + "\t".join(column_labels) + "\n")
file.write("#" + "-" * 127 + "\n")

# Write the data with tab as delimiter
np.savetxt(file, data, fmt='%s', delimiter="\t")

print(f"Data saved to {file_path}")
```
