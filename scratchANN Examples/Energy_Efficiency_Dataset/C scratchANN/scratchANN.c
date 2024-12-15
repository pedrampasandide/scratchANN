
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h> // necessary for `memcpy()` in `random_double()` with `sodium.h`
#include <stdbool.h>
#include "scratchANN.h"
#include <sodium.h> // For `randombytes_buf()` function
#include <omp.h>    // for parallel computation

// ##########################################################################
// ######################### Activation Function ############################
// ##########################################################################

// Since in back propogation we work with post-activation values (`outputs`), we
// we have to rewrite the derivative based on activation functions, for example, in sigmoid and tanh

// ------------------------ Linear ------------------------------
double linear(double x) { return x; }
double linear_derivative(double x)
{
    (void)x;
    return 1;
}

// ------------------------ Sigmoid -----------------------------
/*
Notes for sigmoid function:
Overflow in exp(-x): For very large negative values of x (e.g., x<<−709),
Underflow in exp(-x): For very large positive values of x (e.g., x>>709),
Loss of Precision: Near x=0, the result of 1.0/(1.0+exp(−x)) is very close to 0.5.
In such cases, there might be minor precision errors due to rounding in the division operation.
Instead of sigmoid(x) = 1.0 / (1.0 + exp(-x)) for all x,
the following format has more numerical stability (I checked the difference in results with MNIST)
*/
double sigmoid(double x)
{
    if (x >= 0)
    {
        return 1.0 / (1.0 + exp(-x)); // avoid Underflow
    }
    else
    {
        double exp_x = exp(x);        // avoid computing two times
        return exp_x / (1.0 + exp_x); // avoid Overflow
    }
}

/*
Notes for sigmoid_derivative function:
double sigmoid_derivative(double x) {return sigmoid(x) * (1 - sigmoid(x));}
This function assumes x is already the output of the sigmoid function.
Rounding errors from the sigmoid function propagate here.
Catastrophic Cancellation: When x is very close to 0 or 1
x*(1-x) when x in [0, 1] seems the same as x-x*x
*/
double sigmoid_derivative(double x) { return x * (1 - x); }

// ------------------------- Tanh -------------------------------
/*
For large values of x, tanh(x) approaches +1 (or −1 for large negative x).
However, the calculation of ex and e−x internally in the tanh function may lead to overflows or underflows.
Loss of Precision: When x is very close to 0, tanh(x) returns values close to x
*/
double tanh_function(double x)
{
    if (x > 20.0)
        return 1.0; // Avoid overflow
    if (x < -20.0)
        return -1.0; // Avoid underflow
    return tanh(x);  // Use math.h implementation for other cases
}

/*
double tanh_derivative(double x) {return 1 - pow(tanh(x), 2);}
This function assumes x is already the output of the tanh function.
Any precision errors in tanh(x) will propagate to this derivative.
Catastrophic Cancellation: When x is close to −1 or +1
With -O3 assembly code generated for both pow(x,2) and x*x seems the same
Instead of double tanh_derivative(double x) {return 1 - pow(x, 2); we use:
*/
double tanh_derivative(double x)
{
    // // Ensure output is within the valid range for numerical stability and Prevent underflow
    // x = fmax(fmin(x, 1.0), -1.0);
    return fmax(0, 1.0 - x * x);
}

// double tanh_derivative(double x) {
//     double tanh_x = tanh(x);
//     return fmax(0.0, 1.0 - tanh_x * tanh_x); // Prevent underflow
// }

// ------------------------- ReLu -------------------------------
double relu(double x) { return x >= 0 ? x : 0; }
double relu_derivative(double x) { return x >= 0 ? 1 : 0; }

// --------------------------------------------------------------
// USER: add more activation functions if needed,
// make sure the acitvation is written in the correct format

// ##########################################################################
// ######################### Utility Functions ##############################
// ##########################################################################

// Random value with sodium
double random_double(double min, double max)
{
    unsigned char buffer[sizeof(uint64_t)]; // Buffer to hold random bytes
    uint64_t random_value;

    // Generate random bytes
    randombytes_buf(buffer, sizeof(uint64_t));

    // Convert the random bytes to a 64-bit unsigned integer value
    memcpy(&random_value, buffer, sizeof(uint64_t));

    // Scale the 64-bit random integer to a double value in the range [0, 1.0]
    // Scale the value to the desired range [min, max]
    return min + (random_value / ((double)UINT64_MAX)) * (max - min);
}

// Function to allocate a 3D array of Output doubles with dimensions [num_data][num_layer][neurons per layer]
double ***allocate3DArray(int *layers, int num_layer, int num_data)
{
    // Allocate memory for the outer array (i.e., array[t] for each data sample)
    double ***array = (double ***)calloc(num_data, sizeof(double **));

    // Check for successful allocation
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Allocate memory for each layer for each data sample
    for (int t = 0; t < num_data; t++)
    {
        array[t] = (double **)calloc(num_layer, sizeof(double *));

        if (array[t] == NULL)
        {
            printf("Memory allocation failed!\n");
            // Deallocate allocated memory before exit
            for (int p = 0; p < t; p++)
                free(array[p]);
            free(array);
            exit(1);
        }

        // Allocate memory for each neuron in each layer
        for (int k = 0; k < num_layer; k++)
        {
            int num_neuron = layers[k + 1]; // skipping the first layer
            array[t][k] = (double *)calloc(num_neuron, sizeof(double));

            if (array[t][k] == NULL)
            {
                printf("Memory allocation failed!\n");
                // Deallocate allocated memory before exit
                for (int p = 0; p <= t; p++)
                {
                    for (int q = 0; q < (p == t ? k : num_layer); q++)
                        free(array[p][q]);
                    free(array[p]);
                }
                free(array);
                exit(1);
            }
        }
    }

    return array;
}

// Function to print the 3D array with dimensions [num_data][num_layer][neurons per layer]
void print3DArray(double ***array, int *layers, int num_layer, int num_data)
{
    for (int t = 0; t < num_data; t++)
    {
        printf("Training Sample %d:\n", t + 1);
        for (int k = 0; k < num_layer; k++)
        {
            int num_neuron = layers[k + 1]; // skip the first layer
            printf("  Layer %d (num_neuron = %d):\n", k + 1, num_neuron);
            for (int i = 0; i < num_neuron; i++)
            {
                printf("    %lf ", array[t][k][i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Function to deallocate the 3D array of doubles
void deallocate3DArray(double ***array, int num_layer, int num_data)
{
    // Loop over each training sample
    for (int t = 0; t < num_data; t++)
    {
        // Loop over each layer for each training sample
        for (int k = 0; k < num_layer; k++)
        {
            free(array[t][k]); // Free the neuron array for layer `k`
        }
        free(array[t]); // Free the layer array for training sample `t`
    }
    free(array); // Free the outer array holding all training samples
}

void assign_activation_functions(Layer *layer, ActivationConfig *configs, int num_configs)
{
    int neuron_index = 0;
    int total_neurons = layer->num_neurons;

    // Sum the proportions to normalize
    double total_proportion = 0;
    for (int i = 0; i < num_configs; i++)
    {
        total_proportion += configs[i].proportion;
    }

    // Assign activation functions based on the proportions
    for (int i = 0; i < num_configs; i++)
    {
        int num_neurons = (int)(total_neurons * (configs[i].proportion / total_proportion));

        for (int j = 0; j < num_neurons && neuron_index < total_neurons; j++)
        {
            layer->neurons[neuron_index].activation_function = configs[i].activation_function;

            // Assign corresponding derivative function
            if (configs[i].activation_function == sigmoid)
            {
                layer->neurons[neuron_index].activation_derivative = sigmoid_derivative;
            }
            else if (configs[i].activation_function == tanh_function)
            {
                layer->neurons[neuron_index].activation_derivative = tanh_derivative;
            }
            else if (configs[i].activation_function == linear)
            {
                layer->neurons[neuron_index].activation_derivative = linear_derivative;
            }
            else if (configs[i].activation_function == relu)
            {
                layer->neurons[neuron_index].activation_derivative = relu_derivative;
            }
            // USER: add more activation functions if there are more added into activation_functions.h
            neuron_index++;
        }
    }

    // Ensure all remaining neurons have the last activation function if any are left
    while (neuron_index < total_neurons)
    {
        layer->neurons[neuron_index].activation_function = configs[num_configs - 1].activation_function;
        layer->neurons[neuron_index].activation_derivative = configs[num_configs - 1].activation_function == sigmoid
                                                                 ? sigmoid_derivative
                                                             : configs[num_configs - 1].activation_function == tanh_function
                                                                 ? tanh_derivative
                                                             : configs[num_configs - 1].activation_function == linear
                                                                 ? linear_derivative
                                                                 : relu_derivative;
        // USER: add more activation functions if there are more added into activation_functions.h
        neuron_index++;
    }
}

void Forward_Pass(NeuralNetwork *nn, double **batch_X_train, int batch_size, double ***outputs)
{
// Loop over each sample in the current batch
#pragma omp parallel for
    for (int sample = 0; sample < batch_size; sample++)
    {
        // Forward pass for the first hidden layer
        for (int neuron = 0; neuron < nn->layers[1].num_neurons; neuron++)
        {

            double sum = 0.0;

            // Compute weighted sum for the first hidden layer, based on batch_X_train as input
            for (int input = 0; input < nn->nInputs; input++)
            {
                // USER: Accumulated Rounding Errors in Summation
                sum += nn->layers[1].neurons[neuron].weights[input] * batch_X_train[sample][input];
            }

            // Add bias and apply activation function
            sum += nn->layers[1].neurons[neuron].bias;
            // USER and NOTE: Precision Loss, for example, sigmoid suffers from precision issues for extreme values of sum.
            outputs[sample][0][neuron] = nn->layers[1].neurons[neuron].activation_function(sum);
            /* USER: Propagation of Errors, The errors in weight-input products and bias addition will propagate
            into outputs and affect subsequent layers.
            */
        }

        // Forward pass for the remaining layers (2nd hidden layer onward)
        for (int layer = 2; layer < nn->num_layers; layer++)
        {
            for (int neuron = 0; neuron < nn->layers[layer].num_neurons; neuron++)
            {
                double sum = 0.0;

                // Compute weighted sum based on the previous layer's outputs
                for (int prev_neuron = 0; prev_neuron < nn->layers[layer - 1].num_neurons; prev_neuron++)
                {
                    // USER: Accumulated Rounding Errors in Summation
                    sum += nn->layers[layer].neurons[neuron].weights[prev_neuron] * outputs[sample][layer - 2][prev_neuron];
                }
                // Add bias and apply activation function
                sum += nn->layers[layer].neurons[neuron].bias;
                // USER: Precision Loss, for example, sigmoid suffers from precision issues for extreme values of sum.
                outputs[sample][layer - 1][neuron] = nn->layers[layer].neurons[neuron].activation_function(sum);
                // USER: Propagation of Errors,
            }
        }
    }
}

void Backward_Pass(NeuralNetwork *nn, double **batch_X_train, double **batch_Y_train, int batch_size,
                   double learning_rate, double ***outputs, double ***PL)
{

    int last_layer = nn->num_layers - 1;
    int output_neurons = nn->layers[last_layer].num_neurons;

// First, calculate PL for the output layer
#pragma omp parallel for
    for (int j = 0; j < batch_size; j++) // Process only the batch
    {
        for (int i = 0; i < output_neurons; i++)
        {
            double output = outputs[j][last_layer - 1][i]; // Output from the forward pass
            double target = batch_Y_train[j][i];           // Corresponding target value

            // Calculate derivative using activation derivative for the output layer neuron
            // USER: Catastrophic Cancellation in Derivatives.
            double derivative = nn->layers[last_layer].neurons[i].activation_derivative(output);
            PL[j][last_layer - 1][i] = (output - target) * derivative;
        }
    }

    // Now, calculate PL for each hidden layer in reverse order, excluding the output layer
    for (int l = last_layer - 1; l > 0; l--)
    {
        int num_neurons = nn->layers[l].num_neurons;
        int next_neurons = nn->layers[l + 1].num_neurons;

#pragma omp parallel for
        for (int j = 0; j < batch_size; j++) // Process only the batch
        {
            for (int i = 0; i < num_neurons; i++)
            {
                double output = outputs[j][l - 1][i];

                // Calculate derivative using activation derivative for hidden layer neuron
                // USER: Catastrophic Cancellation in Derivatives.
                double derivative = nn->layers[l].neurons[i].activation_derivative(output);

                // Sum weighted PLs from the next layer
                double sum = 0.0;
                for (int k = 0; k < next_neurons; k++)
                {
                    // USER: Accumulated Errors in Weighted Sum
                    sum += nn->layers[l + 1].neurons[k].weights[i] * PL[j][l][k];
                }

                // Update PL for this neuron
                // USER: Propagation of Errors
                PL[j][l - 1][i] = derivative * sum;
            }
        }
    }

    /*
    ------------------- Option 1: NO Cache missing V2 (some round-off errors) ---------------------
    Continue in Backward_Pass function (for weights and bias updates)
    */

    // Update weights and biases for the first hidden layer (l = 1)
    int first_layer_neurons = nn->layers[1].num_neurons;
    int input_neurons = nn->nInputs;

// Loop over samples, neurons, and inputs
#pragma omp parallel for
    for (int k = 0; k < batch_size; k++)
    {
        // Loop over each sample
        for (int i = 0; i < first_layer_neurons; i++)
        {
            // Loop over neurons in the first layer
            double delta = learning_rate * PL[k][0][i];

            for (int j = 0; j < input_neurons; j++)
            {
                // Loop over input features
                /*
                USER: Precision Loss in Weight Updates
                especially when learning_rate or batch_X_train[k][j] is very small.
                */
                nn->layers[1].neurons[i].weights[j] -= delta * batch_X_train[k][j];
            }
            // Update bias for the neuron
            // USER: Precision Loss similar to Weight Updates
            nn->layers[1].neurons[i].bias -= delta;
        }
    }

// Update weights and biases for remaining layers (l = 2, ..., num_layers - 1)
#pragma omp parallel for
    for (int k = 0; k < batch_size; k++)
    {
        // Loop over each sample
        for (int l = 2; l < nn->num_layers; l++)
        {
            // Loop over layers
            int num_neurons = nn->layers[l].num_neurons;
            int prev_neurons = nn->layers[l - 1].num_neurons;

            for (int i = 0; i < num_neurons; i++)
            {
                // Loop over neurons in the current layer
                double delta = learning_rate * PL[k][l - 1][i];

                for (int j = 0; j < prev_neurons; j++)
                {
                    // Loop over neurons in the previous layer
                    /*
                    USER: Precision Loss in Weight Updates
                    especially when learning_rate or batch_X_train[k][j] is very small.
                    */
                    nn->layers[l].neurons[i].weights[j] -= delta * outputs[k][l - 2][j];
                }
                // Update bias for the neuron
                // USER: Precision Loss similar to Weight Updates
                nn->layers[l].neurons[i].bias -= delta;
            }
        }
    }
    //------------------------------------------------------------------------------------------------

    /*
    ---------------------- Option 2: With Cache missing (less round-off) --------------------------
    Continue in Backward_Pass function (for weights and bias updates)
    */

    // // Update weights and biases for the first hidden layer (l = 1)
    // int first_layer_neurons = nn->layers[1].num_neurons;
    // int input_neurons = nn->nInputs;

    // for (int i = 0; i < first_layer_neurons; i++)
    // {
    //     for (int j = 0; j < input_neurons; j++)
    //     {
    //         double sum = 0.0;
    //         for (int k = 0; k < batch_size; k++) // Loop over batch
    //         {
    //             sum += PL[k][0][i] * batch_X_train[k][j];
    //         }
    //         nn->layers[1].neurons[i].weights[j] -= learning_rate * sum;
    //     }
    // }

    // // Update biases for the first hidden layer
    // for (int i = 0; i < first_layer_neurons; i++)
    // {
    //     double sum = 0.0;
    //     for (int k = 0; k < batch_size; k++) // Loop over batch
    //     {
    //         sum += PL[k][0][i];
    //     }
    //     nn->layers[1].neurons[i].bias -= learning_rate * sum;
    // }

    // // Update weights and biases for remaining layers (l = 2, ..., num_layers - 1)
    // for (int l = 2; l < nn->num_layers; l++)
    // {
    //     int num_neurons = nn->layers[l].num_neurons;
    //     int prev_neurons = nn->layers[l - 1].num_neurons;

    //     // Update weights for layer `l` based on `PL` and outputs from previous layer
    //     for (int i = 0; i < num_neurons; i++)
    //     {
    //         for (int j = 0; j < prev_neurons; j++)
    //         {
    //             double sum = 0.0;
    //             for (int k = 0; k < batch_size; k++) // Loop over batch
    //             {
    //                 double prev_output = outputs[k][l - 2][j];
    //                 sum += PL[k][l - 1][i] * prev_output;
    //             }
    //             nn->layers[l].neurons[i].weights[j] -= learning_rate * sum;
    //         }
    //     }

    //     // Update biases for layer `l`
    //     for (int i = 0; i < num_neurons; i++)
    //     {
    //         double sum = 0.0;
    //         for (int k = 0; k < batch_size; k++) // Loop over batch
    //         {
    //             sum += PL[k][l - 1][i];
    //         }
    //         nn->layers[l].neurons[i].bias -= learning_rate * sum;
    //     }
    // }
    // ---------------------------------------------------------------------------------------------
}

void metrics_binary(NeuralNetwork *nn, int num_train, int num_val, double **Eval_train, double **Eval_val,
                    double **Y_train, double **Y_val, double **X_train, double **X_val, int ep)
{

    int num_outputs = nn->nOutputs;
    int nLayers = nn->num_layers;

    // --------------------------------------------------------------------------
    for (int i = 0; i < num_train; i++)
    {

        Evaluation(nn, X_train[i]);

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_train[i][j] = (nn->layers[nLayers - 1].neurons[j].output) > threshhold ? 1 : 0;
        }
    }

    double correct_predictions = 0; // when all Eval_train[sample][i] are equal to Y_train[sample][i]
    for (int i = 0; i < num_train; i++)
    {
        int all_correct = 1; // Assume all outputs are correct for this sample
        for (int j = 0; j < num_outputs; j++)
        {
            if ((int)Eval_train[i][j] != (int)Y_train[i][j])
            {
                all_correct = 0;
                break;
            }
        }
        correct_predictions += all_correct;
    }

    double accuracy_train = (double)correct_predictions / num_train * 100.0;

    double sum_squared_diff = 0.0;
    for (int i = 0; i < num_train; i++)
    {
        for (int j = 0; j < num_outputs; j++)
        {
            double diff = Y_train[i][j] - Eval_train[i][j];
            sum_squared_diff += diff * diff;
        }
    }

    // Calculate the cost and divide by num_train
    double cost_train = sum_squared_diff / num_train;

    // -------------------------------------------------------------------------------
    for (int i = 0; i < num_val; i++)
    {

        Evaluation(nn, X_val[i]);

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_val[i][j] = (nn->layers[nLayers - 1].neurons[j].output) > threshhold ? 1 : 0;
        }
    }

    correct_predictions = 0; // when all Eval_val[sample][i] are equal to Y_val[sample][i]
    for (int i = 0; i < num_val; i++)
    {
        int all_correct = 1; // Assume all outputs are correct for this sample
        for (int j = 0; j < num_outputs; j++)
        {
            if ((int)Eval_val[i][j] != (int)Y_val[i][j])
            {
                all_correct = 0;
                break;
            }
        }
        correct_predictions += all_correct;
    }

    double accuracy_val = (double)correct_predictions / num_val * 100.0;

    sum_squared_diff = 0.0;
    for (int i = 0; i < num_val; i++)
    {
        for (int j = 0; j < num_outputs; j++)
        {
            double diff = Y_val[i][j] - Eval_val[i][j];
            sum_squared_diff += diff * diff;
        }
    }

    // Calculate the cost and divide by num_train
    double cost_val = sum_squared_diff / num_val;

    printf("==================================================================================\n");
    printf("| Epoch | Train Cost   | Train Accuracy | Validation Cost | Validation Accuracy |\n");
    printf("==================================================================================\n");
    printf("| %5d | %12.6lf | %13.2f%% | %15.6lf | %18.2f%% |\n",
           ep, cost_train, accuracy_train, cost_val, accuracy_val);
    printf("==================================================================================\n");
}

void metrics_continuous(NeuralNetwork *nn, int num_train, int num_val, double **Eval_train, double **Eval_val,
                        double **Y_train, double **Y_val, double **X_train, double **X_val, int ep)
{
    int num_outputs = nn->nOutputs;
    int nLayers = nn->num_layers;

    double mse_train = 0.0, mape_train = 0.0;
    double mse_val = 0.0, mape_val = 0.0;

    // --------------------------------------------------------------------------
    // Train Metrics
    for (int i = 0; i < num_train; i++)
    {
        Evaluation(nn, X_train[i]);

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_train[i][j] = nn->layers[nLayers - 1].neurons[j].output;
        }
    }

    for (int i = 0; i < num_train; i++)
    {
        for (int j = 0; j < num_outputs; j++)
        {
            double y_true = Y_train[i][j];
            double y_pred = Eval_train[i][j];
            double diff = y_true - y_pred;

            // Accumulate MSE
            mse_train += diff * diff;

            // Accumulate MAPE (avoid division by zero)
            if (y_true != 0.0)
            {
                mape_train += fabs(diff / y_true);
            }
        }
    }

    mse_train /= (num_train * num_outputs);
    mape_train = (mape_train / (num_train * num_outputs)) * 100.0;

    // --------------------------------------------------------------------------
    // Validation Metrics
    for (int i = 0; i < num_val; i++)
    {
        Evaluation(nn, X_val[i]);

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_val[i][j] = nn->layers[nLayers - 1].neurons[j].output;
        }
    }

    for (int i = 0; i < num_val; i++)
    {
        for (int j = 0; j < num_outputs; j++)
        {
            double y_true = Y_val[i][j];
            double y_pred = Eval_val[i][j];
            double diff = y_true - y_pred;

            // Accumulate MSE
            mse_val += diff * diff;

            // Accumulate MAPE (avoid division by zero)
            if (y_true != 0.0)
            {
                mape_val += fabs(diff / y_true);
            }
        }
    }

    mse_val /= (num_val * num_outputs);
    mape_val = (mape_val / (num_val * num_outputs)) * 100.0;

    // --------------------------------------------------------------------------
    // Print Metrics
    printf("=============================================================================================\n");
    printf("| Epoch |   Train MSE    |   Train MAPE    | Validation MSE  | Validation MAPE   |\n");
    printf("=============================================================================================\n");
    printf("| %5d | %10.6lf     |%10.2f%%      | %10.6lf      | %11.2f%%      |\n",
           ep, mse_train, mape_train, mse_val, mape_val);
    printf("=============================================================================================\n");
}

// Categorical metrics calculation
void metrics_categorical(NeuralNetwork *nn, int num_train, int num_val, double **Eval_train, double **Eval_val,
                         double **Y_train, double **Y_val, double **X_train, double **X_val, int ep)
{
    int num_outputs = nn->nOutputs;
    int nLayers = nn->num_layers;

    // --------------------------------------------------------------------------
    double correct_predictions = 0;
    double sum_cross_entropy = 0.0;
    for (int i = 0; i < num_train; i++)
    {
        // Forward pass to calculate output
        Evaluation(nn, X_train[i]);

        double max = nn->layers[nLayers - 1].neurons[0].output;
        int predicted_class = 0;

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_train[i][j] = nn->layers[nLayers - 1].neurons[j].output;
            if (nn->layers[nLayers - 1].neurons[j].output > max)
            {
                max = nn->layers[nLayers - 1].neurons[j].output;
                predicted_class = j;
            }
        }
        // Check if prediction matches the true class
        // Y is one-hot coded. It means all Y_train[i][j] must be equal to 0 except for:
        if ((int)Y_train[i][predicted_class] == 1)
        {
            correct_predictions++;
        }

        // Cross-entropy loss
        // sum_cross_entropy += -log(Eval_train[i][predicted_class]); // OR
        for (int j = 0; j < num_outputs; j++)
        {
            sum_cross_entropy += -Y_train[i][j] * log(Eval_train[i][j]);
        }
    }

    double accuracy_train = correct_predictions / num_train * 100.0;
    double cost_train = sum_cross_entropy / num_train;

    // -------------------------------------------------------------------------------
    correct_predictions = 0;
    sum_cross_entropy = 0.0;

    for (int i = 0; i < num_val; i++)
    {
        // Evaluate to calculate output
        Evaluation(nn, X_val[i]);

        double max = nn->layers[nLayers - 1].neurons[0].output;
        int predicted_class = 0;

        for (int j = 0; j < num_outputs; j++)
        {
            Eval_val[i][j] = nn->layers[nLayers - 1].neurons[j].output;
            if (Eval_val[i][j] > max)
            {
                max = Eval_val[i][j];
                predicted_class = j;
            }
        }

        // Check if prediction matches the true class
        // Y is one-hot coded. It means all Y_val[i][j] must be equal to 0 except for:
        if ((int)Y_val[i][predicted_class] == 1)
        {
            correct_predictions++;
        }

        /*
        sum_cross_entropy += -log(Eval_train[i][predicted_class]); is for when we have softmax,
        if sigmoid is in last layer for each neuron, then Cross-entropy loss:
        */
        for (int j = 0; j < num_outputs; j++)
        {
            sum_cross_entropy += -Y_val[i][j] * log(Eval_val[i][j]);
        }
    }

    double accuracy_val = correct_predictions / num_val * 100.0;
    double cost_val = sum_cross_entropy / num_val;

    printf("==================================================================================\n");
    printf("| Epoch | Train Cost   | Train Accuracy | Validation Cost | Validation Accuracy |\n");
    printf("==================================================================================\n");
    printf("| %5d | %12.6lf | %13.2f%% | %15.6lf | %18.2f%% |\n",
           ep, cost_train, accuracy_train, cost_val, accuracy_val);
    printf("==================================================================================\n");
}

void (*metrics)(NeuralNetwork *nn, int num_train, int num_val, double **Eval_train, double **Eval_val,
                double **Y_train, double **Y_val, double **X_train, double **X_val, int ep) = NULL;

// ##########################################################################
// ########################### Main Functions ###############################
// ##########################################################################

// Function to allocate a 2D array
double **allocate2DArray(int rows, int cols)
{

    double **array = (double **)calloc(rows, sizeof(double *));
    if (array == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++)
    {
        array[i] = (double *)calloc(cols, sizeof(double));
        if (array[i] == NULL)
        {
            printf("Memory allocation failed!\n");
            for (int j = 0; j < i; j++)
            {
                free(array[i]);
            }
            free(array);
            exit(1);
        }
    }

    return array;
}

// Function to deallocate the 2D array
void deallocate2DArray(double **array, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(array[i]);
    }
    free(array);
}

// Function to split the data into training and validation sets
void splitData(double **data, int num_inputs, int num_outputs, int num_train, int num_val,
               double ***X_train, double ***Y_train, double ***X_val, double ***Y_val)
{

    // Allocate memory for training and validation sets
    *X_train = allocate2DArray(num_train, num_inputs);
    *Y_train = allocate2DArray(num_train, num_outputs);
    *X_val = allocate2DArray(num_val, num_inputs);
    *Y_val = allocate2DArray(num_val, num_outputs);

    // Split the data into training set
    for (int i = 0; i < num_train; i++)
    {
        // Copy inputs and outputs for the training set
        for (int j = 0; j < num_inputs; j++)
        {
            (*X_train)[i][j] = data[i][j];
        }
        for (int j = 0; j < num_outputs; j++)
        {
            (*Y_train)[i][j] = data[i][j + num_inputs];
        }
    }

    // Split the data into validation set
    for (int i = 0; i < num_val; i++)
    {
        // Copy inputs and outputs for the validation set
        for (int j = 0; j < num_inputs; j++)
        {
            (*X_val)[i][j] = data[i + num_train][j];
        }
        for (int j = 0; j < num_outputs; j++)
        {
            (*Y_val)[i][j] = data[i + num_train][j + num_inputs];
        }
    }
    printf("The dataset is splitted into train and validation successfully.\n");
    printf("Number of samples for train:      %d\n", num_train);
    printf("Number of samples for validation: %d\n", num_val);
    printf("\n");
}

void shuffleData(double **data, int num_samples, int num_columns)
{
    srand((unsigned int)time(NULL)); // Seed for random number generation
    for (int i = num_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1); // Generate a random index
        // Swap row i with row j
        for (int k = 0; k < num_columns; k++)
        {
            double temp = data[i][k];
            data[i][k] = data[j][k];
            data[j][k] = temp;
        }
    }
    printf("The data is shuffled successfully.\n\n");
}

void standardize_features(double **X_train, double **X_val, int num_train, int num_val, int num_inputs)
{

    /* it performs the same as Python:
    scaler.fit_transform(X_train) and scaler.transform(X_val)
    */
    double *mean = (double *)malloc(num_inputs * sizeof(double));
    double *std_dev = (double *)malloc(num_inputs * sizeof(double));

    // Calculate mean and standard deviation for each feature in X_train
    for (int j = 0; j < num_inputs; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < num_train; i++)
        {
            sum += X_train[i][j];
        }
        mean[j] = sum / num_train;

        double sum_sq = 0.0;
        for (int i = 0; i < num_train; i++)
        {
            sum_sq += pow(X_train[i][j] - mean[j], 2);
        }
        std_dev[j] = sqrt(sum_sq / num_train);

        // To avoid division by zero
        if (std_dev[j] == 0.0)
        {
            std_dev[j] = 1.0;
        }
    }

    // Standardize X_train
    for (int i = 0; i < num_train; i++)
    {
        for (int j = 0; j < num_inputs; j++)
        {
            X_train[i][j] = (X_train[i][j] - mean[j]) / std_dev[j];
        }
    }

    // Standardize X_val using the same mean and std_dev from X_train
    for (int i = 0; i < num_val; i++)
    {
        for (int j = 0; j < num_inputs; j++)
        {
            X_val[i][j] = (X_val[i][j] - mean[j]) / std_dev[j];
        }
    }

    // Free allocated memory
    free(mean);
    free(std_dev);
    printf("Inputs are standardized successfully.\n\n");
}

NeuralNetwork *createModel(int *layers, int nLayers, ActivationConfig **activation_configs, int *config_sizes)
{

    // Initialize the sodium library
    if (sodium_init() < 0)
    {
        printf("Failed to open sodium library.\n");
        exit(-1);
    }
    // srand(time(NULL)); // IMP: we dont need this since we are using sodium

    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = nLayers;
    nn->nInputs = layers[0];
    nn->nOutputs = layers[nLayers - 1];

    nn->numNinLayer = malloc(nLayers * sizeof(nLayers));
    for (int i = 0; i < nLayers; i++)
    {
        nn->numNinLayer[i] = layers[i];
    }

    nn->layers = malloc(nLayers * sizeof(Layer));

    // start from i=1 to skip the input layer
    nn->layers[0].num_neurons = layers[0];
    for (int i = 1; i < nLayers; i++)
    {
        nn->layers[i].num_neurons = layers[i];
        nn->layers[i].neurons = malloc(layers[i] * sizeof(Neuron));

        for (int j = 0; j < layers[i]; j++)
        {
            Neuron *neuron = &nn->layers[i].neurons[j];
            // USER: In case of RNN models output is better to be randomized
            neuron->output = 0;
            // ********************************************************************************
            // TEST
            neuron->bias = random_double(-initial_range, initial_range); // Randomize bias
            // neuron->bias = 0.5; // Randomize bias
            // ********************************************************************************
            neuron->weights = malloc(layers[i - 1] * sizeof(double)); // Allocate weights

            // Randomize weights
            for (int k = 0; k < layers[i - 1]; k++)
            {
                // ********************************************************************************
                // TEST
                neuron->weights[k] = random_double(-initial_range, initial_range);
                // neuron->weights[k] = 0.5;
                // ********************************************************************************
            }
        }

        assign_activation_functions(&nn->layers[i], activation_configs[i - 1], config_sizes[i - 1]);
    }

    return nn;
}

void summary(NeuralNetwork *nn)
{
    int total_params = 0;

    printf("Model Summary:\n");
    printf("=====================================================================\n");
    printf("                                      Activation Functions [%%]    \n");
    printf("                               --------------------------------------\n");
    printf("Layer\tNeurons\tParameters\t(Linear    Sigmoid    Tanh      ReLU)\n");
    printf("=====================================================================\n");

    // Start from layer 1 to skip the input layer
    printf("%d\t%-7d%-19d-\n", 0, nn->layers[0].num_neurons, 0); // Input layer has no activation functions
    for (int i = 1; i < nn->num_layers; i++)
    {
        int num_neurons = nn->layers[i].num_neurons;
        int num_weights = nn->numNinLayer[i - 1] * num_neurons; // Weights = previous layer neurons * current layer neurons
        int num_biases = num_neurons;                           // One bias per neuron
        int layer_params = num_weights + num_biases;            // Total params in this layer

        // Count activation functions in the layer
        int count_linear = 0, count_sigmoid = 0, count_tanh = 0, count_relu = 0;
        for (int j = 0; j < num_neurons; j++)
        {
            if (nn->layers[i].neurons[j].activation_function == linear)
                count_linear++;
            else if (nn->layers[i].neurons[j].activation_function == sigmoid)
                count_sigmoid++;
            else if (nn->layers[i].neurons[j].activation_function == tanh_function)
                count_tanh++;
            else if (nn->layers[i].neurons[j].activation_function == relu)
                count_relu++;
        }

        // Calculate percentages
        double pct_linear = (double)count_linear / num_neurons * 100.0;
        double pct_sigmoid = (double)count_sigmoid / num_neurons * 100.0;
        double pct_tanh = (double)count_tanh / num_neurons * 100.0;
        double pct_relu = (double)count_relu / num_neurons * 100.0;

        // Print layer summary with fixed-width formatting
        printf("%d\t%-7d%-19d%-9.1f%-11.1f%-10.1f%.1f\n",
               i, num_neurons, layer_params, pct_linear, pct_sigmoid, pct_tanh, pct_relu);

        total_params += layer_params;
    }

    printf("=====================================================================\n");
    printf("Total Parameters: %d\n\n", total_params);
}

void freeModel(NeuralNetwork *nn)
{
    // i=1 to skip the input layer
    for (int i = 1; i < nn->num_layers; i++)
    {
        for (int j = 0; j < nn->layers[i].num_neurons; j++)
        {
            free(nn->layers[i].neurons[j].weights); // Free weights
        }
        free(nn->layers[i].neurons); // Free neurons
    }
    free(nn->numNinLayer); // Free layers
    free(nn->layers);      // Free layers
    free(nn);              // Free the network itself
    printf("The memory is de-allocated for ANN model successfully.\n\n");
}

void Evaluation(NeuralNetwork *nn, double *sample)
{
// Forward pass for the first hidden layer
#pragma omp parallel for
    for (int neuron = 0; neuron < nn->layers[1].num_neurons; neuron++)
    {
        double sum = 0.0;

        // Compute weighted sum for the first hidden layer, based on X_train as input
        for (int input = 0; input < nn->nInputs; input++)
        {
            // USER: Accumulated Rounding Errors in Summation
            sum += nn->layers[1].neurons[neuron].weights[input] * sample[input];
        }

        // Add bias and apply activation function
        sum += nn->layers[1].neurons[neuron].bias;
        // USER and NOTE: Precision Loss,
        nn->layers[1].neurons[neuron].output = nn->layers[1].neurons[neuron].activation_function(sum);
        // USER: Propagation of Errors,
    }

    // Forward pass for the remaining layers (2nd hidden layer onward)
    for (int layer = 2; layer < nn->num_layers; layer++)
    {
#pragma omp parallel for
        for (int neuron = 0; neuron < nn->layers[layer].num_neurons; neuron++)
        {
            double sum = 0.0;

            // Compute weighted sum based on the previous layer's outputs
            for (int prev_neuron = 0; prev_neuron < nn->layers[layer - 1].num_neurons; prev_neuron++)
            {
                sum += nn->layers[layer].neurons[neuron].weights[prev_neuron] * nn->layers[layer - 1].neurons[prev_neuron].output;
            }

            // Add bias and apply activation function
            sum += nn->layers[layer].neurons[neuron].bias;
            nn->layers[layer].neurons[neuron].output = nn->layers[layer].neurons[neuron].activation_function(sum);
        }
    }
}

void trainModel(NeuralNetwork *nn, double **X_train, double **Y_train, int num_train, double **X_val, double **Y_val, int num_val, TrainSetting setting)
{

    if (strcmp(setting.outputType, "continuous") == 0)
    {
        metrics = metrics_continuous;
    }
    else if (strcmp(setting.outputType, "binary") == 0)
    {
        metrics = metrics_binary;
    }
    else if (strcmp(setting.outputType, "categorical") == 0)
    {
        metrics = metrics_categorical;
    }
    else
    {
        printf("Error: Unsupported output type '%s'\n", setting.outputType);
        exit(EXIT_FAILURE);
    }

    int nLayersWithOutputs = nn->num_layers - 1; // layers with output (except for input layer)
    double ***outputs = allocate3DArray(nn->numNinLayer, nLayersWithOutputs, setting.batch_size);
    double ***PL = allocate3DArray(nn->numNinLayer, nLayersWithOutputs, setting.batch_size);
    double **Eval_train = allocate2DArray(num_train, nn->nOutputs);
    double **Eval_val = allocate2DArray(num_val, nn->nOutputs);

    printf("Training model initiated with the following settings:\n");
    printf("Batch Size: %d\n", setting.batch_size);
    printf("Epochs: %d\n", setting.epochs);
    printf("Learning Rate: %.6f\n", setting.learning_rate);
    printf("Print Every: %d\n", setting.PRINT_EVERY);
    printf("Metrics: %s\n\n", setting.outputType);

    int num_batches = (num_train + setting.batch_size - 1) / setting.batch_size; // Calculate the number of batches

    bool printFlag = false;
    for (int ep = 1; ep <= setting.epochs; ep++)
    {

        if (ep % setting.PRINT_EVERY == 0 || ep == setting.epochs || ep == 1) // also print first and last 
        {
            printFlag = true;
        }
        else
        {
            printFlag = false;
        }

        for (int batch = 0; batch < num_batches; batch++)
        {
            // Calculate the start and end indices for this batch
            int start_idx = batch * setting.batch_size;
            int end_idx = start_idx + setting.batch_size;
            if (end_idx > num_train)
                end_idx = num_train; // Handle last batch

            // Get the number of samples in the current batch
            int current_batch_size = end_idx - start_idx;

            // Pass the current batch to Forward_Pass and Backward_Pass
            Forward_Pass(nn, &X_train[start_idx], current_batch_size, outputs);
            Backward_Pass(nn, &X_train[start_idx], &Y_train[start_idx], current_batch_size, setting.learning_rate, outputs, PL);
        }

        if (printFlag)
        {
            metrics(nn, num_train, num_val, Eval_train, Eval_val, Y_train, Y_val, X_train, X_val, ep);
        }
    }

    deallocate3DArray(outputs, nn->num_layers - 1, setting.batch_size);
    deallocate3DArray(PL, nn->num_layers - 1, setting.batch_size);
    deallocate2DArray(Eval_train, num_train);
    deallocate2DArray(Eval_val, num_val);
}

void saveModel(NeuralNetwork *nn, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        printf("Error: Unable to open file %s for writing.\n", filename);
        return;
    }

    // Save network structure
    fprintf(file, "%d\n", nn->num_layers);
    for (int i = 0; i < nn->num_layers; i++)
    {
        fprintf(file, "%d ", nn->numNinLayer[i]);
    }
    fprintf(file, "\n");

    // Save weights, biases, and activation functions
    for (int i = 1; i < nn->num_layers; i++)
    {
        Layer *layer = &nn->layers[i];
        for (int j = 0; j < layer->num_neurons; j++)
        {
            Neuron *neuron = &layer->neurons[j];

            // Save biases
            fprintf(file, "%.15lf ", neuron->bias);

            // Save weights
            for (int k = 0; k < nn->numNinLayer[i - 1]; k++)
            {
                fprintf(file, "%.15lf ", neuron->weights[k]);
            }
            fprintf(file, "\n");

            // Save activation function (using identifiers)
            if (neuron->activation_function == sigmoid)
            {
                fprintf(file, "sigmoid\n");
            }
            else if (neuron->activation_function == tanh_function)
            {
                fprintf(file, "tanh\n");
            }
            else if (neuron->activation_function == linear)
            {
                fprintf(file, "linear\n");
            }
            else if (neuron->activation_function == relu)
            {
                fprintf(file, "relu\n");
            }
        }
    }

    fclose(file);
    printf("Model saved to %s successfully.\n\n", filename);
}

NeuralNetwork *loadModel(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Error: Unable to open file %s for reading.\n", filename);
        return NULL;
    }

    int num_layers;
    if (fscanf(file, "%d", &num_layers) != 1)
    {
        printf("Error: Failed to read the number of layers.\n");
        fclose(file);
        return NULL;
    }

    int *layers = malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++)
    {
        if (fscanf(file, "%d", &layers[i]) != 1)
        {
            printf("Error: Failed to read layer size for layer %d.\n", i);
            free(layers);
            fclose(file);
            return NULL;
        }
    }

    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->numNinLayer = layers;
    nn->nInputs = layers[0];
    nn->nOutputs = layers[num_layers - 1];
    nn->layers = malloc(num_layers * sizeof(Layer));

    for (int i = 1; i < num_layers; i++)
    {
        nn->layers[i].num_neurons = layers[i];
        nn->layers[i].neurons = malloc(layers[i] * sizeof(Neuron));
        for (int j = 0; j < layers[i]; j++)
        {
            Neuron *neuron = &nn->layers[i].neurons[j];

            // Load biases
            if (fscanf(file, "%lf", &neuron->bias) != 1)
            {
                printf("Error: Failed to read bias for neuron %d in layer %d.\n", j, i);
                // Add cleanup logic here if needed
                fclose(file);
                return NULL;
            }

            // Load weights
            neuron->weights = malloc(layers[i - 1] * sizeof(double));
            for (int k = 0; k < layers[i - 1]; k++)
            {
                if (fscanf(file, "%lf", &neuron->weights[k]) != 1)
                {
                    printf("Error: Failed to read weight for neuron %d, weight %d in layer %d.\n", j, k, i);
                    // Add cleanup logic here if needed
                    fclose(file);
                    return NULL;
                }
            }

            // Load activation function
            char activation_function[50];
            if (fscanf(file, "%s", activation_function) != 1)
            {
                printf("Error: Failed to read activation function for neuron %d in layer %d.\n", j, i);
                fclose(file);
                return NULL;
            }

            if (strcmp(activation_function, "sigmoid") == 0)
            {
                neuron->activation_function = sigmoid;
                neuron->activation_derivative = sigmoid_derivative;
            }
            else if (strcmp(activation_function, "tanh") == 0)
            {
                neuron->activation_function = tanh_function;
                neuron->activation_derivative = tanh_derivative;
            }
            else if (strcmp(activation_function, "linear") == 0)
            {
                neuron->activation_function = linear;
                neuron->activation_derivative = linear_derivative;
            }
            else if (strcmp(activation_function, "relu") == 0)
            {
                neuron->activation_function = relu;
                neuron->activation_derivative = relu_derivative;
            }
            else
            {
                printf("Error: Unknown activation function %s.\n", activation_function);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    printf("Model loaded from %s successfully.\n\n", filename);
    return nn;
}
