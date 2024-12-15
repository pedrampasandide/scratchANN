# Artificial Neural Network in C

## Introduction
**scratchANN** is a lightweight artificial neural network (ANN) library written in C, designed for educational and experimental purposes. It offers a simple yet powerful framework for building and training **multilayer perceptron** models. The library provides essential features to define and train ANNs, allowing users to customize the number of layers, neurons, and activation functions.

### Key Features
- **Flexible Architecture**: Supports user-defined layers and neurons.
- **Mixed Activation Functions**: Allows different activation functions for neurons within the same layer, increasing randomness and flexibility.
- **Supported Activation Functions**: Linear, ReLU, Tanh, Sigmoid.
- **Output Flexibility**: Handles continuous, binary, and categorical data (requires one-hot encoding for categorical data).
- **Optimized for Performance**: Minimizes cache misses and leverages parallel computation via `omp.h`.
- **SGD Optimization**: Implements stochastic gradient descent (SGD) for model training.
- **Essential Utilities**: Functions for data reading, preprocessing, training, evaluation, and model saving/loading.

This library is ideal for scholars and engineers looking for an open-source ANN model to understand basic principles and extend functionality.

---

## Getting Started

### Dependencies
- **Compiler**: A C compiler that supports C11 (e.g., `gcc`).
- **Libraries**:
  - `omp.h`: For parallel computing.
  - `libsodium`: Required for efficient random number generation. (Install via your package manager, e.g., `sudo apt install libsodium-dev` on Ubuntu.)

### Compilation
Use the provided `Makefile` to build the project:

```bash
make
```

This will generate an executable named `sample`.

### Example Usage
To run the demo:

```bash
./sample
```

---

## Core Functions

The following are the main functions provided by **scratchANN**:

1. **Data Handling**:
   - `ReadDataFromFile()`: Reads data from a text file.
   - `splitData()`: Splits data into training and validation/test sets.
   - `standardize_features()`: Standardizes features for better training performance.
   - `shuffleData()`: Shuffles the data to prevent bias during training.

2. **Model Management**:
   - `createModel()`: Creates the MLP model.
   - `freeModel()`: Frees memory allocated for the model.
   - `saveModel()`: Saves the model to disk.
   - `loadModel()`: Loads a saved model from disk.

3. **Training and Evaluation**:
   - `trainModel()`: Trains the model using forward and backpropagation.
   - `Evaluation()`: Computes the outputs of the model.
   - `summary()`: Displays a summary of the model, similar to TensorFlow.

---

## DISCLAIMER

**scratchANN** is provided "as is" without any warranties of any kind, express or implied. The author is not responsible for any damages or losses resulting from the use of this software. Users are advised to validate the functionality of the software in their own environments before deploying it for critical applications.

---

## License

This project is licensed under the GNU General Public License (GPL). See the LICENSE file for more details.

---

## Version

Current version: **v1.0.0**

