#ifndef SCRATCHANN_H
#define SCRATCHANN_H

// ##########################################################################
// ######################### ANN MLP Architecture ###########################
// ##########################################################################


// INPUT: in binary classification
#define threshhold 0.5

//This represents the range of initial random values weights and biases [-initial_range, +initial_range]. 
#define initial_range 0.25

// Struct for Neurons
typedef struct Neuron {
    // USER: not need to save outputs, unless you need a ANN with memory 
    double output;                           // outputs of each neuron is used in Evaluation function and simplifying the mem allocations
    double bias;                             // Randomized between -0.5 and 0.5
    double *weights;                         // Array of weights, also randomized between -0.5 and 0.5
    double (*activation_function)(double);   // Pointer to activation function
    double (*activation_derivative)(double); // Pointer to derivative of activation function
} Neuron;

// Struct for Layers
typedef struct Layer {
    int num_neurons;
    Neuron *neurons; // Array of neurons
} Layer;

// Struct for Neural Network
typedef struct NeuralNetwork {
    int num_layers;       // Total layers (1*input + number of hidden + 1*output)
    int *numNinLayer;     // array holding the number of neurons in each layer including input and output = layers[] (from inputs)
    // USER: An input layer does not need to have a bias, weights, etc.
    // It can be directely received from input array or It can be defined as follows:
    // double *inputLayer; 
    Layer *layers;        // Array of layers (hidden layer(s) and output layer)
    int nInputs;
    int nOutputs;
} NeuralNetwork;

// Struct for Activation Configuration
typedef struct ActivationConfig {
    double (*activation_function)(double); // Activation function
    double proportion;                     // Proportion of neurons with this activation function
} ActivationConfig;


// Train setting
typedef struct TrainSetting{
    int batch_size;
    int epochs;
    double learning_rate;
    int PRINT_EVERY;
    char outputType[50];
}TrainSetting;


// ##########################################################################
// ################### Activation Function Declarations #####################
// ##########################################################################

double linear(double x);
double sigmoid(double x);
double tanh_function(double x);
double relu(double x);

// ##########################################################################
// ###################### Main Function Declarations ########################
// ##########################################################################

double **allocate2DArray(int rows, int cols);

void deallocate2DArray(double **array, int rows);

// Here write a function to read data.txt and save the values in data[MAX_ROWS][MAX_COLS];
double **ReadDataFromFile(const char *filename, int *MAX_ROWS, int MAX_COLS);

void splitData(double **data, int num_inputs, int num_outputs, int num_train, int num_val,
               double ***X_train, double ***Y_train, double ***X_val, double ***Y_val);

void standardize_features(double **X_train, double **X_val, int num_train, int num_val, int num_inputs);

void shuffleData(double **data, int num_samples, int num_columns);

NeuralNetwork *createModel(int *layers, int nLayers, ActivationConfig **activation_configs, int *config_sizes);

void freeModel(NeuralNetwork *nn);

void Evaluation(NeuralNetwork *nn, double *sample);

void trainModel(NeuralNetwork *nn, double **X_train, double **Y_train, int num_train, double **X_val, double **Y_val, int num_val, TrainSetting setting);

void summary(NeuralNetwork *nn);

void saveModel(NeuralNetwork *nn, const char *filename);

NeuralNetwork* loadModel(const char *filename);








#endif