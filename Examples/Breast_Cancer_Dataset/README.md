# scratchANN Case Study: Breast Cancer Dataset

## Getting Started

- Use the same codes from the <a href="https://github.com/pedrampasandide/scratchANN" target="_blank">scratchANN repository</a>, except for `demo.c`, which must be replaced with `demo_BreastCancer.c`.
- Update `Makefile` to reference `demo_BreastCancer.c` instead of `demo.c`.
- The Breast Cancer dataset has been downloaded and saved as `breast_cancer_data.txt`, using the Python code provided in the Appendix. The file is formatted to be compatible with the `ReadDataFromFile()` function.
- The rest is exactly the same. `make` and `./sample` to run.
- For more details, check out the [YouTube **scratchANN** playlist](https://youtu.be/3YAi5TZyRdw?si=2P4CJl3ooP_-jXgl), where the case study trains the model using **scratchANN** and compares the results with an MLP implemented in TensorFlow Keras.


## Appendix

The following Python code was used to download, process, and save the Breast Cancer dataset into `breast_cancer_data.txt`:

```python
import pandas as pd
import numpy as np
from google.colab import drive
from sklearn.datasets import load_breast_cancer

# Mount Google Drive
drive.mount('/content/drive')

# Load breast cancer dataset
data = load_breast_cancer()

# Convert the data to a pandas DataFrame for easy manipulation
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column to DataFrame
df['target'] = data.target

# Prepare column labels
column_labels = df.columns.tolist()

# Verify data shape
print("Shape of the data array:", df.shape)

# Filepath to save on Google Drive
file_path = "/content/drive/My Drive/breast_cancer_data.txt"

# Save data in txt format with header
with open(file_path, 'w') as file:
# Write header information
file.write("# Breast Cancer Dataset\n")
file.write("# Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html\n")
file.write("#\n")
file.write("# Columns:\n")
file.write("# " + "\t".join(column_labels) + "\n")
file.write("#" + "-" * 127 + "\n")

# Write the data with tab as delimiter
np.savetxt(file, df.to_numpy(), fmt='%s', delimiter="\t")
```
