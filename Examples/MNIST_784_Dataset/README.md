# scratchANN Case Study: MNIST 784 Dataset

## Getting Started

- **IMPORTANT**: The MNIST 784 dataset is large, so it has not been uploaded here. However, using the Python code provided in the Appendix, the dataset can be one-hot encoded for targets, processed, and saved as `mnist_one_hot_data.txt`. Ensure the file is formatted to be compatible with the `ReadDataFromFile()` function.

- Use the same codes from the <a href="https://github.com/pedrampasandide/scratchANN" target="_blank">scratchANN repository</a>, except for `demo.c`, which must be replaced with `demo_MNIST784.c`.
- Update `Makefile` to reference `demo_MNIST784.c` instead of `demo.c`.

- The rest is exactly the same. `make` and `./sample` to run.
- For more details, check out the [YouTube **scratchANN** playlist](URL), where the case study trains the model using **scratchANN** and compares the results with an MLP implemented in TensorFlow Keras.


## Appendix

The following Python code was used to download, process, and save the MNIST 784 dataset into `mnist_one_hot_data.txt` (in case of using another name change the input in `demo_MNIST784.c`):

```python
import numpy as np
from sklearn.datasets import fetch_openml
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Convert X and y to numpy arrays
X = np.array(X, dtype=int)
y = np.array(y, dtype=int)  # Convert y to integers

# Create a one-hot encoded matrix for y
num_classes = 10
y_one_hot = np.zeros((y.shape[0], num_classes), dtype=int)

y_one_hot[np.arange(y.shape[0]), y] = 1

# Concatenate X and the one-hot encoded y
data = np.hstack((X, y_one_hot))

# Verify the shape and content
print("Shape of the data array:", data.shape)

# Prepare the column labels
num_features = X.shape[1]
feature_labels = [f"x{i+1}" for i in range(num_features)]
output_labels = [f"y{i}" for i in range(num_classes)]
column_labels = feature_labels + output_labels

# Filepath to save on Google Drive
file_path = "/content/drive/My Drive/mnist_one_hot_data.txt"

# Write the file
with open(file_path, 'w') as file:
# Write header information
file.write("# MNIST Dataset\n")
file.write("# Reference: https://www.openml.org/d/554\n")
file.write("# " + "\t".join(column_labels) + "\n")
file.write("#" + "-" * 127 + "\n")

# Write the data with tab as delimiter
np.savetxt(file, data, fmt='%d', delimiter="\t")

print(f"Data saved to {file_path}")
```
