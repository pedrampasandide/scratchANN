/*
License: IMP: with version

Compile:
`make`

Execute:
`./sample`

MANUAL:
- INPUT: Inputs which are needed to be determinded by the user
- USER : Places that can be imporved in the future
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "scratchANN.h"
#include <omp.h>    // for parallel computation
#include <math.h>








int main()
{

    TrainSetting setting;

    // --------------------------------- INPUTs ---------------------------------------
    // INPUT: the number of layers and neurons in each layer are given by `layers[]`
    int layers[] = {30, 64, 1};

    // INPUT:
    // for example, =0.2 means 20% of data for train and 80% for validation
    double train_split = 0.7;

    // INPUT: Define the batch size (int type)
    setting.batch_size = 32;

    // INPUT (double type)
    setting.learning_rate = 0.01; 

    // INPUT (int type)
    setting.epochs = 2;

    // INPUT: print metrics every PRINT_EVERY epochs (int type)
    setting.PRINT_EVERY = 1;

    // INPUT: type of outputs defines the metrics needed to be printed
    // options: "continuous" or "binary" or "categorical"
    strcpy(setting.outputType, "binary");

    // INPUT: the name of file holding data.
    char filename[] = "breast_cancer_data.txt";

    // INPUT: Set to true to shuffle the data
    bool shuffle = true; // Set to true to shuffle the data

    // INPUT: Set true to standardize inputs (X_train and X_val) to the model
    bool scale_standardize = true; // Set to true to shuffle the data

    // INPUT: the name of a text file to save the model into on disk
    char filnameSave[] = "model.txt";
    
    // INPUT: Number of threads in parallel computation
    omp_set_num_threads(6);
    // --------------------------------------------------------------------------------

    // ---------------------------------- INPUTs --------------------------------------
    //********************************************************************************
    // ActivationConfig layer2_configs[] = {
    //     {relu, 1}};
    // ActivationConfig layer3_configs[] = {
    //     {sigmoid, 1}};
    // int config_sizes[] = {1, 1}; // Manually input the size of each config array.
    // ActivationConfig *activation_configs[] = {layer2_configs, layer3_configs};
    //********************************************************************************
    
    ActivationConfig layer2_configs[] = {
        {tanh_function, 1},{relu, 1},{linear, 1}};
    ActivationConfig layer3_configs[] = {
        {sigmoid, 1}};
    int config_sizes[] = {3, 1}; // Manually input the size of each config array.
    ActivationConfig *activation_configs[] = {layer2_configs, layer3_configs};

    //********************************************************************************
    
    
    // --------------------------------------------------------------------------------

    // ------------------------- Computed based on Inputs -----------------------------
    int nLayers = sizeof(layers) / sizeof(layers[0]);
    int num_inputs = layers[0];
    int num_outputs = layers[nLayers - 1];
    int NUM_FEATURES = num_inputs + num_outputs; // or number of columns in data.txt
    int NUM_SAMPLES = 0;                         // or the number of rows will be read from data.txt
    // --------------------------------------------------------------------------------

    // ------------------------------- Read Data --------------------------------------
    printf("Reading data...\n");
    double **data = ReadDataFromFile(filename, &NUM_SAMPLES, NUM_FEATURES);
    
    // --------------------------------------------------------------------------------

    // ------------------------- Computed based on Inputs -----------------------------
    int num_train = (int)(NUM_SAMPLES * train_split);
    int num_val = NUM_SAMPLES - num_train;
    // --------------------------------------------------------------------------------

    // -------------------------- Shuffle and Split Data ------------------------------
    // Shuffle the data if requested
    if (shuffle)
    {
        shuffleData(data, num_train + num_val, num_inputs + num_outputs);
    }

    double **X_train = NULL, **Y_train = NULL;
    double **X_val = NULL, **Y_val = NULL;

    printf("Splitting data into train and validation dataset...\n");
    splitData(data, num_inputs, num_outputs, num_train, num_val, &X_train, &Y_train, &X_val, &Y_val);

    // USER: based on splitData I won't need the original data
    deallocate2DArray(data, NUM_SAMPLES);

    if (scale_standardize)
    {
        standardize_features(X_train, X_val, num_train, num_val, num_inputs);
    }
    
    // --------------------------------------------------------------------------------

    // ----------------------------- Creat the model ----------------------------------
    NeuralNetwork *nn = createModel(layers, nLayers, activation_configs, config_sizes);
    summary(nn);
    // --------------------------------------------------------------------------------

    // ----------------------------- Train the model ----------------------------------
    // double start, end, cpu_time_used;
    // start = omp_get_wtime();

    trainModel(nn, X_train, Y_train, num_train, X_val, Y_val, num_val, setting);

    // end = omp_get_wtime();
    // cpu_time_used = end - start;
    // printf("Wall Clock Time: %f seconds\n", cpu_time_used);
    // --------------------------------------------------------------------------------

    // ----------------------------------- Save ---------------------------------------
    printf("\nSaving the model on disk...\n");
    saveModel(nn, filnameSave);
    // --------------------------------------------------------------------------------

    // ---------------------------------- Remove --------------------------------------
    printf("De-allocating the memory for ANN model...\n");
    freeModel(nn);
    // --------------------------------------------------------------------------------

    // ----------------------------------- Load ---------------------------------------
    printf("Loading the model from disk...\n");
    NeuralNetwork *loaded_nn = loadModel(filnameSave);
    // --------------------------------------------------------------------------------

    // ----------------------------- Test laoded model --------------------------------
    printf("IMPORTANT: Testing loaded model by training for new settings...\n");

    setting.learning_rate = 0.001;
    setting.epochs = 2;
    trainModel(loaded_nn, X_train, Y_train, num_train, X_val, Y_val, num_val, setting);
    
    // --------------------------------------------------------------------------------

    // ---------------------------------- Remove --------------------------------------
    printf("De-allocating the memory for ANN model...\n");
    freeModel(loaded_nn);
    // --------------------------------------------------------------------------------

    // --------------------------------- Free Data ------------------------------------
    printf("De-allocating train and validation data.\n");
    deallocate2DArray(X_train, num_train);
    deallocate2DArray(Y_train, num_train);
    deallocate2DArray(X_val, num_val);
    deallocate2DArray(Y_val, num_val);
    // --------------------------------------------------------------------------------

    return 0;
}

